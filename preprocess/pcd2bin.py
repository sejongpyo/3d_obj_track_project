from pypcd import pypcd
import numpy as np
import os
from tqdm import tqdm
import argparse

def convert():
    parser = argparse.ArgumentParser(description="convert .pcd to .bin")
    parser.add_argument("--scene_path", help='scene path; 1, 2, ...n', type=str)
    parser.add_argument("--out_name", help='out_folder_name', type=str)
    args = parser.parse_args()
    
    scene_path = args.scene_path
    out_folder_name = args.out_name
    
    scene_li = os.listdir(scene_path) # ['1', ..., 'n']
    for scene in scene_li:
        scenario_li = os.listdir(os.path.join(scene_path, scene))

        for scenario in scenario_li:
            try:
                save_dir = os.path.join(scene_path, scene, scenario, out_folder_name)
                os.makedirs(save_dir)
            except FileExistsError:
                print('folder exists -> continue')
                pass

            scenario_pcd_li = list()
            pcd_li = os.listdir(os.path.join(scene_path, scene, scenario, 'lidar'))

            for pcd in pcd_li:
                scenario_pcd_li.append(os.path.join(scene_path, scene, scenario, 'lidar', pcd))

            # convert .pcd to .bin by scenario
            print(f'{scene}_{scenario}_Converting Start!')
            for pcd_file in tqdm(scenario_pcd_li):
                pc = pypcd.PointCloud.from_path(pcd_file)

                np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
                np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
                np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
                np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)

                points_ = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
                
                pcd_dir = pcd_file.split('/')
                save_path = os.path.join(scene_path, pcd_dir[3], pcd_dir[4], out_folder_name,
                                         pcd_dir[-1].replace('.pcd', '.bin'))
                points_.tofile(save_path)

    print('Converting is finished')
    
if __name__=="__main__":
    convert()