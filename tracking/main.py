import time
import sys
import numpy as np
import os
import argparse
from track_utils import load_files, get_scenario_num
from track_model import AB3DMOT

def start_track():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--detected_file', type=str, default="/workspace/OpenPCDet/output/pointrcnn/detection_result", help='specify the detection result path')
    parser.add_argument('--save_path', type=str, default="/workspace/OpenPCDet/output", help='specify the tracking result save path')
    args = parser.parse_args()
    
    det_classes = ['vehicle', 'pedestrian', 'bike']
    det_id2str = {1:'vehicle', 2:'pedestrian', 3:'bike'}
    
    for cls in det_classes:
        seq_file_list, num_seq = load_files(os.path.join(args.detected_file, cls))
        total_time, total_frames = 0.0, 0
        save_dir = os.path.join(args.save_path, 'track_result', cls)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        eval_dir = os.path.join(save_dir, 'data')
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        
        seq_count = 0
        for seq_file in seq_file_list:
            seq_name = get_scenario_num(seq_file)
            eval_file = os.path.join(eval_dir, seq_name + '.txt'); eval_file = open(eval_file, 'w')
            save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name)
            if not os.path.exists(save_trk_dir):
                os.makedirs(save_trk_dir)
            
            mot_tracker = AB3DMOT()
            seq_dets = np.loadtxt(seq_file, delimiter=',')
            if len(seq_dets.shape) == 1:
                seq_dets = np.expand_dims(seq_dets, axis=0)
            if seq_dets.shape[1] == 0:
                eval_file.close()
                continue
            
            min_frame, max_frame = int(seq_dets[:, 0].min()), int(seq_dets[:, 0].max())
            for frame in range(min_frame, max_frame + 1):
                print_str = 'processing %s: %d/%d, %d/%d    \r' % (seq_name, seq_count, num_seq, frame, max_frame)
                sys.stdout.write(print_str)
                sys.stdout.flush()
                save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame); save_trk_file = open(save_trk_file, 'w')
                
                ori_array = seq_dets[seq_dets[:, 0] == frame, -1].reshape((-1, 1))
                other_array = seq_dets[seq_dets[:, 0] == frame, 1:7]
                additional_info = np.concatenate((ori_array, other_array), axis=1)
                
                dets = seq_dets[seq_dets[:, 0] == frame, 7:14]
                dets_all = {'dets': dets, 'info': additional_info}
                
                start_time = time.time()
                trackers = mot_tracker.update(dets_all)
                cycle_time = time.time() - start_time
                total_time += cycle_time
                
                for d in trackers:
                    bbox3d_tmp = d[0:7]
                    id_tmp = d[7]
                    ori_tmp = d[8]
                    type_tmp = det_id2str[d[9]]
                    bbox2d_tmp_trk = d[10:14]
                    conf_tmp = d[14]
                    
                    str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
                        bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
					    bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp, id_tmp)
                    save_trk_file.write(str_to_srite)
                    
                    str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
	    				type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
		    			bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], 
			    		conf_tmp)
                    eval_file.write(str_to_srite)
                total_frames += 1
                save_trk_file.close()
            seq_count += 1
            eval_file.close()
        print('Total Tracking took: %.3f for %d frames or %.1f FPS' % (total_time, total_frames, total_frames / total_time))
    
if __name__=='__main__':
    start_track()