from genericpath import exists
import cv2
import sys
sys.path.append('..')
import copy
import numpy as np
import os
import random
from tqdm import tqdm
import config as cfg
from img_cut import get_location, cut_img

class PersonBbox:
    def __init__(self,l,r,t,b) -> None:
        self.left = l
        self.right = r
        self.top = t
        self.bottom = b

class StoragePath:
    def __init__(self,occ,fusion,occ_cut,cut) -> None:
        self.occ_path = occ
        self.fus_path = fusion
        self.occ_cut_path = occ_cut
        self.cut_path = cut

class SilhouettesPath:
    def __init__(self,bg,fg) -> None:
        self.bg_sil_path = bg
        self.fg_sil_path = fg

def random_choice(set_,current_video):
    np.random.seed(random.randint(1,100000))
    random.seed(random.randint(1,100000))
    select_video = random.choice(set_)
    select_id,_,_,select_view = select_video.split('-')
    unsuitable_view = ['000','180']
    current_id,_,_,current_view = current_video.split('-')
    # 防止选择同一人视频或者同一视角(同一视角容易全程A完全遮挡B)
    # 防止前景选择0或180°的，均不太符合透视关系
    if select_id!=current_id and select_view!=current_view and select_view not in unsuitable_view:
        return select_video
    else:
        return random_choice(set_,current_video)

def parse_path_element(path,video_name):
    id,type_,_,view=video_name.split('-')
    type_=type_+'-'+_
    view = view.split('.')[0]
    return os.path.join(path,id,type_,view)

def select_fg(silhouettes_path,bg_video,videos,bg_sil_path,bg_frames):
    max_occ_ratio = 0.
    cnt = 1
    # # 倘若背景剪影图文件夹为空,遍历下一个视频文件名
    # if len(bg_frames)==0:
    #     continue
    # 控制遮挡率,最大遮挡帧必须大于>0.3
    while max_occ_ratio < 0.3 and cnt <200:
        fg_video = random_choice(videos,bg_video)
        fg_sil_path = parse_path_element(silhouettes_path,fg_video)
        # 倘若前景剪影图文件夹为空,循环随机至不为空
        while len(os.listdir(fg_sil_path))==0:
            fg_video = random_choice(videos,bg_video)
            fg_sil_path = parse_path_element(silhouettes_path,fg_video)
        fg_frames = [cv2.imread(os.path.join(fg_sil_path,frame_name),cv2.IMREAD_GRAYSCALE) for frame_name in os.listdir(fg_sil_path)]
        sil_path = SilhouettesPath(bg_sil_path,fg_sil_path)
        for frame_index in range(min(len(bg_frames),len(fg_frames))):
            bg_frame = cv2.resize(bg_frames[frame_index],(cfg.frame_width,cfg.frame_height),interpolation=cv2.INTER_AREA)
            fg_frame = cv2.resize(fg_frames[frame_index],(cfg.frame_width,cfg.frame_height),interpolation=cv2.INTER_AREA)
            occlusion_frame = cv2.subtract(bg_frame,fg_frame)
            occlusion_pixel = (bg_frame.sum()-occlusion_frame.sum())/255
            occlusion_ratio = occlusion_pixel/(bg_frame.sum()/255)
            max_occ_ratio = max(max_occ_ratio,occlusion_ratio)
        # print(f'第{cnt}次选择视频{fg_video},max_occ_ratio为{max_occ_ratio}')
        cnt+=1
    # print(f'最终选择视频{fg_video}')
    return fg_frames,sil_path
    

def map_img(min_frame_num,bg_frames,fg_frames,sil_path:SilhouettesPath,save_path:StoragePath):
    for frame_index in range(min_frame_num):
        bg_frame = cv2.resize(bg_frames[frame_index],(cfg.frame_width,cfg.frame_height),interpolation=cv2.INTER_AREA)
        fg_frame = cv2.resize(fg_frames[frame_index],(cfg.frame_width,cfg.frame_height),interpolation=cv2.INTER_AREA)
        fusion_frame = cv2.add(bg_frame, fg_frame)
        fus_frame_save_path = os.path.join(save_path.fus_path,os.listdir(sil_path.fg_sil_path)[frame_index])
        cv2.imwrite(fus_frame_save_path,fusion_frame)
        # TBD 此处过滤掉了背景中的残缺剪影图，是否对补全数据集有意义
        top_bg, bottom_bg, left_bg, right_bg = get_location(bg_frame.copy())
        if left_bg is None or right_bg is None or top_bg is None or bottom_bg is None: 
            continue
        top_fg, bottom_fg, left_fg, right_fg = get_location(fg_frame.copy())
        if top_fg is None or bottom_fg is None or left_fg is None or right_fg is None: 
            continue
        occlusion_frame = cv2.subtract(bg_frame,fg_frame)
        # occlusion_pixel = (bg_frame.sum()-occlusion_frame.sum())/255
        # occlusion_ratio = occlusion_pixel/(bg_frame.sum()/255)
        occ_frame_save_path = os.path.join(save_path.occ_path,os.listdir(sil_path.bg_sil_path)[frame_index])
        cv2.imwrite(occ_frame_save_path,occlusion_frame)
        # 存储裁剪后的遮挡剪影图
        occ_cut_sil = cut_img(occlusion_frame,top_bg,bottom_bg,left_bg,right_bg)
        occ_cut_sil_save_path = os.path.join(save_path.occ_cut_path,os.listdir(sil_path.bg_sil_path)[frame_index])
        cv2.imwrite(occ_cut_sil_save_path,occ_cut_sil)
        # 存储裁剪后的未遮挡真值图
        cut_sil = cut_img(bg_frame,top_bg,bottom_bg,left_bg,right_bg)
        cut_sil_save_path = os.path.join(save_path.cut_path,os.listdir(sil_path.bg_sil_path)[frame_index])
        cv2.imwrite(cut_sil_save_path,cut_sil)
        
        
def run(silhouettes_path,ids_txt):
    ids = np.loadtxt(ids_txt).astype(int)
    # videos = sorted([f for f in glob.glob(videodir + '/' + "*", recursive=True)])
    # videos_list = sorted(os.listdir(video_dir))
    videos_list = os.listdir(cfg.video_dir)
    videos = [v for v in videos_list if int(v.split('-')[0]) in ids]  # ids中的所有视频
    for bg_video in tqdm(videos):
        occ_cut_save_path = parse_path_element(cfg.save_dir_path,bg_video)
        cut_save_path = parse_path_element(cfg.gt_dir_path,bg_video)
        occ_save_path = parse_path_element(cfg.occ_dir_path,bg_video)
        fus_save_path = parse_path_element(cfg.fusion_dir_path,bg_video)
        save_path = StoragePath(occ_save_path,fus_save_path,occ_cut_save_path,cut_save_path)
        bg_sil_path = parse_path_element(silhouettes_path,bg_video)
        bg_frames = [cv2.imread(os.path.join(bg_sil_path,frame_name),cv2.IMREAD_GRAYSCALE) for frame_name in os.listdir(bg_sil_path)]
        fg_frames , sil_path = select_fg(silhouettes_path,bg_video,videos,bg_sil_path,bg_frames)
        if not os.path.exists(occ_save_path):
            os.makedirs(occ_save_path)
        if not os.path.exists(fus_save_path):
            os.makedirs(fus_save_path)
        if not os.path.exists(occ_cut_save_path):
            os.makedirs(occ_cut_save_path)
        if not os.path.exists(cut_save_path):
            os.makedirs(cut_save_path)
        if len(bg_frames)<len(fg_frames):
            map_img(len(bg_frames),bg_frames,fg_frames,sil_path,save_path)
        else:
            map_img(len(fg_frames),bg_frames,fg_frames,sil_path,save_path)
            # 保存多出来的背景帧
            for frame_index in range(len(fg_frames),len(bg_frames)):
                occ_frame_save_path = os.path.join(occ_save_path,os.listdir(bg_sil_path)[frame_index])
                cv2.imwrite(occ_frame_save_path,bg_frames[frame_index])



if __name__ == "__main__":
    run(cfg.silhouettes_path,cfg.ids_txt)