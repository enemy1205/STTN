from .dataset import TestDataset,gaitset_collate_fn
from torch.utils.data import DataLoader,sampler
import torch
import os.path as osp
import os
import numpy as np
from .utils import ts2var,np2var,cuda_dist

def load_data(dataset_path, resolution, dataset, cache=False):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
            continue
        label_path = osp.join(dataset_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)

    pid_list = sorted(list(set(label)))
    test_list = pid_list

    test_source = TestDataset(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label) if l in test_list],
        cache, resolution)

    return test_source


def evaluation(feature, view, seq_type, label, dataset_name):
    dataset = dataset_name.split('-')[0]
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc

class GaitEval:
    def __init__(self,cfg) -> None:
        super(GaitEval, self).__init__()
        self.resolution = cfg['resolution']
        self.dataset = cfg['dataset']
        self.batch_size = cfg['batch_size']
        self.num_workers = cfg['num_workers']
        self.eval_dataset_path = cfg['occ_root_path']
        self.gt_dataset_path = cfg['gt_root_path']
        self.eval_dataset = load_data(self.eval_dataset_path, self.resolution, self.dataset, cache=False)
        self.eval_data_loader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            sampler=sampler.SequentialSampler(self.eval_dataset),
            collate_fn=gaitset_collate_fn,
            num_workers=self.num_workers)
    
    def gt_eval(self,gait_model):
        gt_dataset = load_data(self.gt_dataset_path, self.resolution, self.dataset, cache=False)
        gt_data_loader = DataLoader(
            dataset=gt_dataset,
            batch_size=self.batch_size,
            sampler=sampler.SequentialSampler(gt_dataset),
            collate_fn=gaitset_collate_fn,
            num_workers=self.num_workers)
        # 推理
        gait_model.eval()
        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        for i, x in enumerate(gt_data_loader):
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = np2var(batch_frame).int()
            x = seq, view, seq_type, label, batch_frame
            feature = gait_model(x)
            n, num_bin, _ = feature.size()
            # print(f'{i} ,:{n}')
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label
        feature_list = np.concatenate(feature_list, 0)
        acc_CASIA_B = evaluation(feature_list,view_list,seq_type_list,label_list, self.dataset)
        acc_NM_mean, acc_BG_mean, acc_CL_mean = np.mean(acc_CASIA_B[0, :, :, 0]), np.mean(acc_CASIA_B[1, :, :, 0]), np.mean(acc_CASIA_B[2, :, :, 0])
        print(f"acc_NM_mean:{acc_NM_mean} , acc_BG_mean:{acc_BG_mean} , acc_CL_mean:{acc_CL_mean}")
        
        
    
    def eval(self,rec_model,gait_model):
        # 推理
        gait_model.eval()
        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        for i, x in enumerate(self.eval_data_loader):
            seq, view, seq_type, label, batch_frame = x
            
            for j in range(len(seq)):
                seq[j] = np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = np2var(batch_frame).int()
            x = seq, view, seq_type, label, batch_frame
            feature = gait_model(x)
            n, num_bin, _ = feature.size()
            # print(f'{i} ,:{n}')
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label
        feature_list = np.concatenate(feature_list, 0)
        acc_CASIA_B = evaluation(feature_list,view_list,seq_type_list,label_list, self.dataset)
        acc_NM_mean, acc_BG_mean, acc_CL_mean = np.mean(acc_CASIA_B[0, :, :, 0]), np.mean(acc_CASIA_B[1, :, :, 0]), np.mean(acc_CASIA_B[2, :, :, 0])
        print(f"acc_NM_mean:{acc_NM_mean} , acc_BG_mean:{acc_BG_mean} , acc_CL_mean:{acc_CL_mean}")