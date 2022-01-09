import pickle
import time
import os
import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=result_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    classes = {'vehicle': 1, 'pedestrian': 2, 'bike': 3}
    for class_name in classes.keys():
        cnt = 0
        for i in det_annos:
            scene, scenario, seq_num = map(str, i['frame_id'].split('_'))
            directory = os.path.join(str(result_dir), class_name)
            result_file = os.path.join(directory, f"{scene}_{scenario}.txt")
            if os.path.isfile(result_file):
                with open(result_file, 'a') as f:
                    bbox = i['bbox']
                    box_3d = i['boxes_lidar']
                    alpha = i['alpha']
                    score = i['score']
                    name = i['name']
                    
                    for idx in range(len(bbox)):
                        if name[idx] == class_name:
                            print('{},{},{:.4f},{:.4f},{:.4f},'
                                    '{:.4f},{:.4f},{:.4f},'
                                    '{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                                        str(cnt), classes[name[idx]],
                                        bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                        bbox[idx][3], score[idx], box_3d[idx][5], box_3d[idx][4],
                                        box_3d[idx][3], box_3d[idx][0], box_3d[idx][1],
                                        box_3d[idx][2], box_3d[idx][6], alpha[idx][0]), file=f)
            else:
                os.makedirs(directory, exist_ok=True)
                with open(result_file, 'w') as f:
                    bbox = i['bbox']
                    box_3d = i['boxes_lidar']
                    alpha = i['alpha']
                    score = i['score']
                    name = i['name']
                    
                    for idx in range(len(bbox)):
                        if name[idx] == class_name:
                            print('{},{},{:.4f},{:.4f},{:.4f},'
                                    '{:.4f},{:.4f},{:.4f},'
                                    '{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                                        str(cnt), classes[name[idx]],
                                        bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                        bbox[idx][3], score[idx], box_3d[idx][5], box_3d[idx][4],
                                        box_3d[idx][3], box_3d[idx][0], box_3d[idx][1],
                                        box_3d[idx][2], box_3d[idx][6], alpha[idx][0]), file=f) 
                f.close
            if int(seq_num) == 199:
                cnt = 0

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
