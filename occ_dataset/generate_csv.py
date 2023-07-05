import os 
import sys
import csv
import pandas as pd

def path_traverse(root_path:str):
    child_file_names = os.listdir(root_path)
    child_paths = [os.path.join(root_path,name) for name in child_file_names]
    return child_paths

if __name__ == "__main__":
    # occ_cut表示遮挡后的剪影图
    # cut 表示未遮挡的剪影图，即真值gt
    occ_cut_sil_path = '/home/lxc/dataset/CASIA_B/val/silhouettes_occ_cut'
    cut_sil_path = '/home/lxc/dataset/CASIA_B/val/silhouettes_cut'
    all_occ_cut_dataset_path = []
    all_occ_cut_dataset_len = []
    all_cut_dataset_path = []
    all_cut_dataset_len = []
    ids = os.listdir(cut_sil_path)
    occ_cut_id_paths = path_traverse(occ_cut_sil_path)
    cut_id_paths = path_traverse(cut_sil_path)
    for i in range(len(cut_id_paths)):
        occ_cut_type_paths = path_traverse(occ_cut_id_paths[i])
        cut_type_paths = path_traverse(cut_id_paths[i])
        assert len(occ_cut_type_paths)==len(cut_type_paths),'the sil dataset has a difference type number between occ_cut and cut '
        for j in range(len(occ_cut_type_paths)):
            occ_cut_view_paths = path_traverse(occ_cut_type_paths[j])
            cut_view_paths = path_traverse(cut_type_paths[j])
            assert len(occ_cut_view_paths)==len(cut_view_paths),'the sil dataset has a difference view number between occ_cut and cut '
            for k in range(len(occ_cut_view_paths)):
                all_occ_cut_dataset_path.append(occ_cut_view_paths[k])
                all_cut_dataset_path.append(cut_view_paths[k])
                all_occ_cut_dataset_len.append(len(os.listdir(occ_cut_view_paths[k])))
                all_cut_dataset_len.append(len(os.listdir(cut_view_paths[k])))
    
    dataframe = pd.DataFrame({'occ_sil_path':all_occ_cut_dataset_path, 
                            'occ_sil_video_len':all_occ_cut_dataset_len, 
                            'gt_sli_path':all_cut_dataset_path, 
                            'gt_sil_video_len': all_cut_dataset_len
                            })
    dataframe.to_csv('/home/lxc/projects/VideoInpainting/STTN/path_csv/valid_data_path.csv', index=False, sep=',')