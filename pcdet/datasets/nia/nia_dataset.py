import copy
import pickle
import os
import argparse
import yaml
from easydict import EasyDict
import numpy as np
from skimage import io
from pypcd import pypcd
import sys

from . import nia_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_nia, common_utils, object3d_nia
from ..dataset import DatasetTemplate


class NiaDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode] # train or val

        split_dir = os.path.join(self.root_path, 'splits', self.split + '.txt') # data/splits/train.txt or val.txt
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()]
        
        self.lidar_dir = os.path.join(self.root_path, 'lidar_data')
        self.label_dir = os.path.join(self.root_path, 'lidar_data_labeling')
        self.img_dir = os.path.join(self.root_path, 'image_data')
        self.cal_dir = os.path.join(self.root_path, 'calibration')

        self.nia_infos = []
        self.include_nia_data(self.mode)

    def include_nia_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading NIA dataset')
        nia_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = os.path.join(self.root_path, info_path)
            if not os.path.exists(info_path):
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nia_infos.extend(infos)

        self.nia_infos.extend(nia_infos)

        if self.logger is not None:
            self.logger.info('Total samples for NIA dataset: %d' % (len(nia_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = os.path.join(self.root_path, 'splits', self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()]

    def get_lidar(self, idx): # 16_001_006
        scene, scenario, seq_num = map(str, idx.split('_'))
        lidar_file = os.path.join(self.lidar_dir, scene, scenario, 'lidar', f"{idx}.pcd")
        assert os.path.exists(lidar_file)
        pc = pypcd.PointCloud.from_path(lidar_file)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)
        points_ = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
        return points_

    def get_image(self, idx):
        scene, scenario, seq_num = map(str, idx.split('_'))
        img_file = os.path.join(self.img_dir, scene, scenario, 'image0', f"{idx}.jpg")
        assert os.path.exists(img_file)
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        scene, scenario, seq_num = map(str, idx.split('_'))
        img_file = os.path.join(self.img_dir, scene, scenario, 'image0', f"{idx}.jpg")
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        scene, scenario, seq_num = map(str, idx.split('_'))
        label_file = os.path.join(self.label_dir, scene, scenario, 'lidar', f"{idx}.json")
        assert os.path.exists(label_file)
        return object3d_nia.get_objects_from_label(label_file)

    def get_calib(self, idx):
        scene, scenario, seq_num = map(str, idx.split('_'))
        calib_cam0 = os.path.join(self.cal_dir, scene, scenario, 'calib_Camera0.txt')
        calib_lidar2cam = os.path.join(self.cal_dir, scene, scenario, 'calib_CameraToLidar0.txt')
        assert os.path.exists(calib_cam0)
        assert os.path.exists(calib_lidar2cam)
        return calibration_nia.Calibration(calib_cam0, calib_lidar2cam)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        # sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        def process_single_scene(sample_idx):
        
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}    
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': np.array([1860, 2880], dtype=np.int32)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            calib_info = {'cam0': calib.cam0, 'lidar2cam': calib.lidar2cam}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
                annotations['track_id'] = np.array([obj.track_id for obj in obj_list], np.int32)
                annotations['gt_boxes_lidar'] = np.concatenate([obj.box3d.reshape(1, 7) for obj in obj_list], axis=0)
                gt_boxes_lidar = np.concatenate([obj.box3d.reshape(1, 7) for obj in obj_list], axis=0)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('nia_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            try:
                annos = info['annos']
            except:
                continue
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = np.ones((len(pred_boxes), 1))
            pred_dict['bbox'] = np.ones((len(pred_boxes), 4))
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            # if output_path is not None:
            #     cur_det_file = output_path / ('%s.txt' % frame_id)
            #     with open(cur_det_file, 'w') as f:
            #         bbox = single_pred_dict['bbox']
            #         loc = single_pred_dict['location']
            #         dims = single_pred_dict['dimensions']  # lhw -> hwl

            #         for idx in range(len(bbox)):
            #             print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
            #                   % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
            #                      bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
            #                      dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
            #                      loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
            #                      single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.nia_infos[0].keys():
            return None, {}

        from .nia_object_eval_python import eval as nia_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.nia_infos]
        ap_result_str, ap_dict = nia_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.nia_infos) * self.total_epochs

        return len(self.nia_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.nia_infos)

        info = copy.deepcopy(self.nia_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar'].astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict

def convert_nia_format(infos_path, data_path):
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)
    
    cnt = 0
    for idx, info in enumerate(infos):
        scene, scenario, seq_num = map(str, info['image']['image_idx'].split('_'))
        anno = info['annos']
        track_id = anno['track_id']
        labels = anno['name']
        bbox = anno['bbox']
        box_3d = anno['gt_boxes_lidar'] # l w h
        truncated = anno['truncated']
        occluded = anno['occluded']
        alpha = anno['alpha']
        
        out_path = os.path.join(data_path, 'track_label')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file = f"{out_path}/{scene}_{scenario}.txt"
        if os.path.isfile(out_file):
            with open(out_file, 'a') as f:
                for id in range(len(box_3d)):
                    print(
                        '{} {} {} {} {} '
                        '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
                        '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                            str(cnt), track_id[id], labels[id],
                            truncated[id], occluded[id], alpha[id],
                            bbox[id][0], bbox[id][1], bbox[id][2],
                            bbox[id][3], box_3d[id][5], box_3d[id][4],
                            box_3d[id][3], box_3d[id][0], box_3d[id][1],
                            box_3d[id][2], box_3d[id][6]
                            ),
                        file=f)
        else:
            with open(out_file, 'w') as f:
                for id in range(len(box_3d)):
                    print(
                        '{} {} {} {} {} '
                        '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
                        '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                            str(cnt), track_id[id], labels[id],
                            truncated[id], occluded[id], alpha[id],
                            bbox[id][0], bbox[id][1], bbox[id][2],
                            bbox[id][3], box_3d[id][5], box_3d[id][4],
                            box_3d[id][3], box_3d[id][0], box_3d[id][1],
                            box_3d[id][2], box_3d[id][6]
                            ),
                        file=f)
        if int(seq_num) == 199:
            cnt = 0
        else:
            cnt += 1
    return None

def create_nia_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = NiaDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = os.path.join(save_path, f'nia_infos_{train_split}.pkl')
    val_filename = os.path.join(save_path, f'nia_infos_{val_split}.pkl')
    trainval_filename = os.path.join(save_path, f'nia_infos_trainval.pkl')
    test_filename = os.path.join(save_path, 'nia_infos_test.pkl')

    print('---------------Start to generate data infos---------------')
    dataset.set_split(train_split)
    nia_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(train_filename, 'wb') as f:
        pickle.dump(nia_infos_train, f)
    print('Nia info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    nia_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(val_filename, 'wb') as f:
        pickle.dump(nia_infos_val, f)
    convert_nia_format(val_filename, data_path)
    print('Nia info val file is saved to %s' % val_filename)    
    
    with open(trainval_filename, 'wb') as f:
        pickle.dump(nia_infos_train + nia_infos_val, f)
    print('nia info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    nia_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(nia_infos_test, f)
    print('nia info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_nia_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        data_path = sys.argv[3]
        create_nia_infos(
            dataset_cfg=dataset_cfg,
            class_names=['vehicle', 'pedestrian', 'bike'],
            data_path=data_path,
            save_path=data_path
        )
    # import yaml
    # from pathlib import Path
    # from easydict import EasyDict
    # dataset_cfg = EasyDict(yaml.safe_load(open("/workspace/OpenPCDet/tools/cfgs/dataset_configs/nia_dataset.yaml")))
    # data_path = "/workspace/data"
    # create_nia_infos(
    #     dataset_cfg=dataset_cfg,
    #     class_names=['vehicle', 'pedestrian', 'bike'],
    #     data_path=data_path,
    #     save_path=data_path
    # )