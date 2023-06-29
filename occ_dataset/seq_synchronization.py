import os
import sys
def synchronize_seq(seg_dataset_path,ori_dataset_path):
    ids = os.listdir(ori_dataset_path)
    id_paths = [os.path.join(ori_dataset_path,id) for id in ids]
    for id_path in id_paths:
        types = os.listdir(id_path)
        type_paths = [os.path.join(id_path,type_) for type_ in types]
        for type_path in type_paths:
            views = os.listdir(type_path)
            view_paths = [os.path.join(type_path,view) for view in views]
            for i,view_path in enumerate(view_paths):
                view = views[i]
                seg_view_path = view_path.replace(ori_dataset_path,seg_dataset_path)
                seg_file_names = os.listdir(seg_view_path)
                file_names = sorted(os.listdir(view_path))
                file_paths = [os.path.join(view_path,file_name) for file_name in file_names]
                # frame_start = int(file_names[0].split('-')[-1].split('.')[0])
                # frame_end = int(file_names[-1].split('-')[-1].split('.')[0])
                # seg_file_paths = [file_path.replace(ori_dataset_path,seg_dataset_path) for file_path in file_paths]
                for seg_file_name in seg_file_names:
                    if seg_file_name in file_names:
                        continue
                    else:
                        # print(f'{os.path.join(seg_view_path,seg_file_name)} will be delete')
                        if os.path.exists(seg_file_name):
                            os.remove(seg_file_name)
                    
                    
if __name__ == "__main__":
    ori_dataset_path = '/home/lxc/dataset/CASIA_B_silh/GaitDatasetB-silh'
    seg_dataset_path = '/home/lxc/dataset/CASIA_B/silhouettes_seg'
    synchronize_seq(seg_dataset_path,ori_dataset_path)