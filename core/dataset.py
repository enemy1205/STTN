import os
import cv2
import math
import pandas as pd
import random
import xarray as xr
import torch
import os.path as osp
import numpy as np
import torchvision.transforms.functional as tf
import torch.utils.data as tordata

class TestDataset(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0

    def __getitem__(self, index):
        # pose sequence sampling
        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index],

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)


def gaitset_collate_fn(batch):
    sample_type = "all"
    frame_num = 30

    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]
    frame_sets = [batch[i][1] for i in range(batch_size)]
    view = [batch[i][2] for i in range(batch_size)]
    seq_type = [batch[i][3] for i in range(batch_size)]
    label = [batch[i][4] for i in range(batch_size)]
    batch = [seqs, view, seq_type, label, None]

    def select_frame(index):
        sample = seqs[index]
        frame_set = frame_sets[index]
        if sample_type == 'random':
            frame_id_list = random.choices(frame_set, k=frame_num)
            _ = [feature.loc[frame_id_list].values for feature in sample]
        else:
            _ = [feature.values for feature in sample]
        return _

    seqs = list(map(select_frame, range(len(seqs))))

    if sample_type == 'random':
        seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
    else:
        gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)
        batch_frames = [[
                            len(frame_sets[i])
                            for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                            if i < batch_size
                            ] for _ in range(gpu_num)]
        if len(batch_frames[-1]) != batch_per_gpu:
            for _ in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
        seqs = [[
                    np.concatenate([
                                        seqs[i][j]
                                        for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                        if i < batch_size
                                        ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
        seqs = [np.asarray([
                                np.pad(seqs[j][_],
                                        ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                        'constant',
                                        constant_values=0)
                                for _ in range(gpu_num)])
                for j in range(feature_num)]
        batch[4] = np.asarray(batch_frames)

    batch[0] = seqs
    return batch


class MultiGaitDataset(torch.utils.data.Dataset):
    """MultiGaitDataset."""    
    def __init__(self,train_csv_path,video_len,train_id_number):
        self.csv_path = train_csv_path
        self.video_len = video_len
        self.id_list = [i for i in range(1, train_id_number+1)]
        self.occ_cut_silt_path = []
        self.occ_cut_silt_len = []
        self.gt_silt_path = []
        self.gt_pose_path = []
        self.df = pd.read_csv(self.csv_path)
        for i in range(len(self.df)):
            self.occ_cut_silt_path.append(self.df['occ_sil_path'][i])
            self.occ_cut_silt_len.append(self.df['occ_sil_video_len'][i])
            self.gt_silt_path.append(self.df['gt_sli_path'][i])

    def __len__(self):
        return len(self.occ_cut_silt_path)
        
    def loadimg(self, path):
        inImage = cv2.imread(path, 0)  # 以单通道读进来
        info = np.iinfo(inImage.dtype)
        inImage = inImage.astype(np.float32) / info.max  # 归一化到0-1

        iw = inImage.shape[1]
        ih = inImage.shape[0]
        if iw < ih:
            inImage = cv2.resize(inImage, (64, int(64 * ih/iw)))
        else:
            inImage = cv2.resize(inImage, (int(64 * iw / ih), 64))
        inImage = inImage[0:64, 0:64]
        return torch.from_numpy(inImage).unsqueeze(0)
        # 归一化至-1~1
        # return torch.from_numpy(2 * inImage - 1).unsqueeze(0)

    def random_shift(self, image,w_t,h_t):

        image=torch.from_numpy(image)
        image = tf.affine(image.unsqueeze(0), translate=(w_t, h_t), shear=0, angle=0, scale=1)
        return image


    def load_video(self, occ_path, gt_path):
        occ_video = []
        gt_video = []
        for imgs in sorted(os.listdir(occ_path)):
            occ_video.append(self.loadimg(os.path.join(occ_path, imgs)))
        for imgs in sorted(os.listdir(gt_path)):
            gt_video.append(self.loadimg(os.path.join(gt_path, imgs)))
        # 文件夹中剪影图数< 设定视频长度 则用最后一帧去填充空缺,cut和gt应该是一一对应关系，不必二次比较数量
        if len(occ_video) < self.video_len:
            len_broken = len(occ_video)
            # print(len_broken)
            for i in range(self.video_len - len_broken):
                occ_video.append(occ_video[-1])
        if len(gt_video) < self.video_len:
            len_gt = len(gt_video)
            for i in range(self.video_len - len_gt):
                gt_video.append(gt_video[-1])
        mask_length = 10
        # mask_idx = random.randint(0, self.video_len-mask_length)
        # 至于为什么要从10开始，按数组长度正常为[0,self.video_len-mask_length]
        # 但倘若去前面的容易导致时序预测一开始接受的就是mask图
        mask_idx = random.randint(10, self.video_len-mask_length)
        black_mask = torch.zeros(mask_length,1, 64,64).double()
        # 摘取video_len的序列
        split = random.randint(0, len(occ_video)-self.video_len)
        occ_video = occ_video[split:split+self.video_len]
        gt_video = gt_video[split:split+self.video_len]
        occ_video[mask_idx:mask_idx+mask_length] = black_mask
        return torch.stack(occ_video), torch.stack(gt_video)


    '''
    load triplet video id1_rec, id1_gt, id2_gt
    '''
    def load_SiaNet_video(self, occ_cut_path, gt_path):
        # is_flip = random.choice(self.is_flip)
        occ_cut_video = []
        gt_video = []
        neg_gt_video = []
        pos_gt_video = []
        neg_id = pos_id = int(occ_cut_path.split('/')[-3])
        neg_len = 0
        pos_len = 0
        # 获取路径前缀
        common_path = os.path.commonpath([self.gt_silt_path[0], self.gt_silt_path[-1]])
        while neg_id == pos_id or neg_len == 0:
            # 另外选择一个id，如果选中相同则符合while，再次重选
            neg_id = random.choice(self.id_list)
            # print(neg_id)
            # 找出该id人物的所有视频路径序号，随机选择
            neg_idx = random.choice([idx for idx, s in enumerate(self.gt_silt_path) if \
                os.path.join(common_path,'%03d'%neg_id) \
                in s])
            neg_len = self.occ_cut_silt_len[neg_idx]
            if neg_len == 0:
                continue
            neg_video_path = self.gt_silt_path[neg_idx]
        while pos_len == 0:
            # print(neg_id)
            pos_idx = random.choice([idx for idx, s in enumerate(self.gt_silt_path) if \
                os.path.join(common_path,'%03d'%pos_id) \
                in s])
            pos_len = self.occ_cut_silt_len[pos_idx]
            if pos_len == 0:
                continue
            pos_video_path = self.gt_silt_path[pos_idx]

        # print("path:", path_broken, "\n", path_gt, "\n" , neg_video_path, "\n", pos_video_path, "\n")
        for imgs in sorted(os.listdir(occ_cut_path)):
           
            occ_cut_video.append(self.loadimg(os.path.join(occ_cut_path, imgs)))
          
        for imgs in sorted(os.listdir(gt_path)):
            
            gt_video.append(self.loadimg(os.path.join(gt_path, imgs)))
           
        for imgs in sorted(os.listdir(neg_video_path)):
            neg_gt_video.append(self.loadimg(os.path.join(neg_video_path, imgs)))
 
        for imgs in sorted(os.listdir(pos_video_path)):
            pos_gt_video.append(self.loadimg(os.path.join(pos_video_path, imgs)))

        if len(occ_cut_video) < self.video_len:
            len_broken = len(occ_cut_video)
            # print(len_broken)
            for i in range(self.video_len - len_broken):
                occ_cut_video.append(occ_cut_video[-1])
                gt_video.append(gt_video[-1])
        if len(neg_gt_video) < self.video_len:
            len_gt_neg = len(neg_gt_video)
            for i in range(self.video_len - len_gt_neg):
                neg_gt_video.append(neg_gt_video[-1])
        if len(pos_gt_video) < self.video_len:
            len_gt_pos = len(pos_gt_video)
            for i in range(self.video_len - len_gt_pos):
                pos_gt_video.append(pos_gt_video[-1])
        
        
        # print("len:", len(video_broken), "\n", len(video_gt_neg), "\n", len(video_gt_pos), "\n")

        split = random.randint(0, len(occ_cut_video)-self.video_len)
        split_neg = random.randint(0, len(neg_gt_video)-self.video_len)
        split_pos = random.randint(0, len(pos_gt_video)-self.video_len)
        occ_cut_video = occ_cut_video[split:split+self.video_len]
        gt_video = gt_video[split:split+self.video_len]
        neg_gt_video = neg_gt_video[split_neg:split_neg+self.video_len]
        pos_gt_video = pos_gt_video[split_pos:split_pos+self.video_len]
        # print("len_train:", len(video_broken), "\n", len(video_gt_neg), "\n", len(video_gt_pos), "\n")
        return torch.stack(occ_cut_video), torch.stack(gt_video), torch.stack(neg_gt_video),torch.stack(pos_gt_video)
    
    
    
    def load_SiaNet_video_randommask(self, path_broken, path_gt):
        # is_flip = random.choice(self.is_flip)
        video_broken = []
        video_gt = []
        video_gt_neg = []
        video_gt_pos = []
        neg_id = pos_id = int(path_broken.split('/')[-3])
        neg_len = 0
        pos_len = 0
        common_path = os.path.commonpath([self.gt_silt_path[0], self.gt_silt_path[-1]])
        while neg_id == pos_id or neg_len == 0:
            neg_id = random.choice(self.id_list)
            # print(neg_id)
            neg_idx = random.choice([idx for idx, s in enumerate(self.gt_silt_path) if \
                os.path.join(common_path,'%03d'%neg_id) \
                in s])
            neg_len = self.occ_cut_silt_len[neg_idx]
            if neg_len == 0:
                continue
            neg_video_path = self.gt_silt_path[neg_idx]
        while pos_len == 0:
            # print(neg_id)
            pos_idx = random.choice([idx for idx, s in enumerate(self.gt_silt_path) if \
                os.path.join(common_path,'%03d'%pos_id) \
                in s])
            pos_len = self.occ_cut_silt_len[pos_idx]
            if pos_len == 0:
                continue
            pos_video_path = self.gt_silt_path[pos_idx]

        # print("path:", path_broken, "\n", path_gt, "\n" , neg_video_path, "\n", pos_video_path, "\n")
        for imgs in sorted(os.listdir(path_broken)):
           
            video_broken.append(self.loadimg(os.path.join(path_broken, imgs)))
          
        for imgs in sorted(os.listdir(path_gt)):
            
            video_gt.append(self.loadimg(os.path.join(path_gt, imgs)))
           
        for imgs in sorted(os.listdir(neg_video_path)):
            video_gt_neg.append(self.loadimg(os.path.join(neg_video_path, imgs)))
 
        for imgs in sorted(os.listdir(pos_video_path)):
            video_gt_pos.append(self.loadimg(os.path.join(pos_video_path, imgs)))

        if len(video_broken) < self.video_len:
            len_broken = len(video_broken)
            # print(len_broken)
            for i in range(self.video_len - len_broken):
                video_broken.append(video_broken[-1])
        if len(video_gt) < self.video_len:
            len_gt = len(video_gt)
            for i in range(self.video_len - len_gt):
                video_gt.append(video_gt[-1])
        if len(video_gt_neg) < self.video_len:
            len_gt_neg = len(video_gt_neg)
            for i in range(self.video_len - len_gt_neg):
                video_gt_neg.append(video_gt_neg[-1])
        if len(video_gt_pos) < self.video_len:
            len_gt_pos = len(video_gt_pos)
            for i in range(self.video_len - len_gt_pos):
                video_gt_pos.append(video_gt_pos[-1])
        
        
        # print("len:", len(video_broken), "\n", len(video_gt_neg), "\n", len(video_gt_pos), "\n")

        split = random.randint(0, len(video_broken)-self.video_len)
        split_neg = random.randint(0, len(video_gt_neg)-self.video_len)
        split_pos = random.randint(0, len(video_gt_pos)-self.video_len)
        video_broken = video_broken[split:split+self.video_len]
        video_gt = video_gt[split:split+self.video_len]
        video_gt_neg = video_gt_neg[split_neg:split_neg+self.video_len]
        video_gt_pos = video_gt_pos[split_pos:split_pos+self.video_len]

        return torch.stack(video_broken), torch.stack(video_gt), torch.stack(video_gt_neg),torch.stack(video_gt_pos) 

    def __getitem__(self, index):
        # 某段视频不存在帧
        if self.occ_cut_silt_len[index] == 0:
            return None
        occ_cut_silt_video, gt_silt_video, neg_gt_video, pos_gt_video = self.load_SiaNet_video(self.occ_cut_silt_path[index], self.gt_silt_path[index])
     
        return occ_cut_silt_video, gt_silt_video, neg_gt_video, pos_gt_video
    
    """
    load sianet triplet id1_rec, id1_gt_anchor, id2_neg, id1_pos random_mask
    """
    # def __getitem__(self, index):
    #     if self.occ_cut_silt_len[index] == 0:
    #         return None
    #     occ_cut_silt_video, gt_silt_video = self.load_video(self.occ_cut_silt_path[index], self.gt_silt_path[index])

    #     return occ_cut_silt_video, gt_silt_video

