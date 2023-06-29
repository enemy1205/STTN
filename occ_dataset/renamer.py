import os 
import sys

sil_path = '/home/lxc/dataset/CASIA_B/silhouettes_seg'
ids = os.listdir(sil_path)
id_paths = [os.path.join(sil_path,id) for id in ids]
for id_path in id_paths:
    types = os.listdir(id_path)
    type_paths = [os.path.join(id_path,type_) for type_ in types]
    for type_path in type_paths:
        views = os.listdir(type_path)
        view_paths = [os.path.join(type_path,view) for view in views]
        for i,view_path in enumerate(view_paths):
            view = views[i]
            file_names = os.listdir(view_path)
            file_paths = [os.path.join(view_path,file_name) for file_name in file_names]
            for j in range(len(file_names)):
                frame_num = file_names[j].split('-')[-1]
                revise_name = file_names[j][:-len(frame_num)]+view+'-'+frame_num
                revise_path = os.path.join(view_path,revise_name)
                os.rename(file_paths[j],revise_path)
