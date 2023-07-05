import os
import cv2
import random
import math
import glob
from tqdm import tqdm
import shutil
import importlib
import time
import numpy as np
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from eval import GaitEval
from core.dataset import MultiGaitDataset,TestDataset
from core.loss import AdversarialLoss,gradient_penalty_ori,generate_gei_2


def gaitset_model_init(gaitset_checkpoint_path):
    # 模型初始化
    gait_model = importlib.import_module('model.gaitset').GaitSet().float()
    gait_model.eval()
    gait_model.cuda()
    ckpt = torch.load(gaitset_checkpoint_path)['model']
    gait_model.load_state_dict(ckpt)
    
    return  gait_model

def collate_fn_test(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        batch = [torch.zeros([1, 1, 64, 64], dtype=torch.float64), torch.zeros([1, 1, 64, 64], dtype=torch.float64), torch.zeros([1, 1, 64, 64], dtype=torch.float64)]
    return torch.utils.data.dataloader.default_collate(batch)

def collate_fn(batch):
    # todo 固定值32解决
    batch = list(filter(lambda x: x is not None, batch))
    for i in range(32 - len(batch)):
        batch.append(batch[-1])
    return torch.utils.data.dataloader.default_collate(batch)

class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        if debug:
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # setup data set and data loader
        data_cfg = config['data_loader']
        self.train_dataset = MultiGaitDataset(data_cfg['train_csv_path'],data_cfg['video_len'],data_cfg['train_id_number'])
        self.train_sampler = None
        self.train_args = config['trainer']
        self.eval_args = config['eval']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None), 
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler,
            collate_fn=collate_fn)
        self.test_dataset = TestDataset(data_cfg['valid_csv_path'],data_cfg['video_len'],data_cfg['valid_id_number'])
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_args['batch_size'] // config['world_size'],
            shuffle=False, 
            num_workers=self.eval_args['num_workers'],
            collate_fn=collate_fn_test)
        
        # set eval model and dataset
        eval_cfg = config['eval']
        self.gait_eval = GaitEval(eval_cfg)
        self.gait_model = gaitset_model_init(self.config['gait_model_path'])
        self.gait_model = self.gait_model.to(self.config['device'])

        # set loss functions 
        self.adversarial_loss = torch.nn.BCELoss(size_average=True, reduce=True)
        # self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss().cuda()
        self.l2_loss = nn.MSELoss().cuda()
        
        # setup models including generator and discriminator
        net = importlib.import_module('model.convlstm')
        Generator=net.convlstm_model_64_expand
        self.netG = Generator()
        self.netG = self.netG.to(self.config['device'])
        self.netD = net.Discriminator()
        self.netD = self.netD.to(self.config['device'])
        self.netA = net.NetA(nc=1)
        self.netA = self.netA.to(self.config['device'])
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimA = torch.optim.Adam(
            self.netA.parameters(), 
            lr=config['trainer']['lr_a'])
        self.load()

        if config['distributed']:
            self.netG = DDP(
                self.netG, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netD = DDP(
                self.netD, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netA = DDP(
                self.netA, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.assd_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.assd_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'assd'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    # get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

    # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1**(min(self.iteration,
                          self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0

    # load netG and netD and netA
    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
        if latest_epoch is not None:
            gen_path = os.path.join(
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            if self.config['global_rank'] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    def save_test_fig(self,iter, rec_silt_video, gt_silt_video):
        random_batch_index = random.randint(0,rec_silt_video.shape[0]-1)
        video_len = self.config['data_loader']['video_len']
        vid_rec = torch.Tensor(video_len,1,64,64)
        vid_gt = torch.Tensor(video_len,1,64,64)
        for i in range(rec_silt_video[random_batch_index].shape[0]):
            rec_img = rec_silt_video[random_batch_index][i,:,:,:]
            gt_img = gt_silt_video[random_batch_index][i,:,:,:]
            vid_rec[i,:,:,:] = rec_img
            vid_gt[i,:,:,:] = gt_img
        grid = make_grid(torch.cat((vid_rec,vid_gt)))
        figs_save_path = os.path.join(self.config['save_dir']+'/figs','eval')
        os.makedirs(figs_save_path,exist_ok=True)
        save_image(grid, os.path.join(figs_save_path,f'{iter}.jpg'))

    def save_fig(self,iter, rec_silt_video, gt_silt_video, occ_silt_video):
        # save figs of train-infer imgs
        random_batch_index = random.randint(0,rec_silt_video.shape[0]-1)
        video_len = self.config['data_loader']['video_len']
        vid_rec = torch.Tensor(video_len,1,64,64)
        vid_gt = torch.Tensor(video_len,1,64,64)
        vid_occ = torch.Tensor(video_len,1,64,64)
        for i in range(rec_silt_video[random_batch_index].shape[0]):
            rec_img = rec_silt_video[random_batch_index][i,:,:,:]
            gt_img = gt_silt_video[random_batch_index][i,:,:,:]
            occ_img = occ_silt_video[random_batch_index][i,:,:,:]
            vid_rec[i,:,:,:] = rec_img
            vid_gt[i,:,:,:] = gt_img
            vid_occ[i,:,:,:] = occ_img
        grid = make_grid(torch.cat((vid_rec,vid_gt,vid_occ)))
        figs_save_path = os.path.join(self.config['save_dir']+'/figs',str(self.config["train_id"]))
        os.makedirs(figs_save_path,exist_ok=True)
        save_image(grid, os.path.join(figs_save_path,f'{iter}.jpg'))

    # save parameters every eval_epoch
    def save(self, iter):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(iter).zfill(5)))
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(iter).zfill(5)))
            assd_path = os.path.join(
                self.config['save_dir'], 'assd_{}.pth'.format(str(iter).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
                netA = self.netA.module
            else:
                netG = self.netG
                netD = self.netD
                netA = self.netA
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'netA': netA.state_dict()}, assd_path)
            os.system('echo {} > {}'.format(str(iter).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))


    # train entry
    def train(self):
        
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
        
        # self.gait_eval.gt_eval(self.gait_model)
        
        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)
            # self._valid_epoch()
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        device = self.config['device']
        for occ_silt_video, gt_silt_video, gt_pos_video, gt_neg_video in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1

            occ_silt_video, gt_silt_video = occ_silt_video.to(device).float(),gt_silt_video.to(device).float()
            gt_pos_video , gt_neg_video = gt_pos_video.to(device).float(),gt_neg_video.to(device).float()
            b, t, c, h, w = occ_silt_video.size()
            pred_silt_video = self.netG(occ_silt_video).permute(0,2,1,3,4)
            gt_silt_video = gt_silt_video.permute(0,2,1,3,4)
            gt_pos_video = gt_pos_video.permute(0,2,1,3,4)
            gt_neg_video = gt_neg_video.permute(0,2,1,3,4)
            gen_loss = 0
            # discriminator adversarial loss
            real_vid_feat = self.netD(gt_silt_video)
            fake_vid_feat = self.netD(pred_silt_video.detach())
            # dis_real_loss = torch.mean(real_vid_feat+1)
            # dis_fake_loss = torch.mean(fake_vid_feat+1)
            # dis_loss = (dis_fake_loss - dis_real_loss)/2
            dis_real_loss = torch.mean(real_vid_feat)
            dis_fake_loss = torch.mean(fake_vid_feat)
            dis_loss = dis_fake_loss - dis_real_loss

            
            grad_penalty = gradient_penalty_ori(self.config['losses']["gp_weight"], self.netD, gt_silt_video, pred_silt_video)
            gp_losses = grad_penalty
            
            dis_loss += gp_losses
            
            self.add_summary(
                self.dis_writer, 'loss/gp_losses', gp_losses.item())
            self.add_summary(
                self.dis_writer, 'loss/dis_loss', dis_loss.item())
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            rec_gei = generate_gei_2(pred_silt_video.detach())
            gt_video_pos_gei = generate_gei_2(gt_pos_video).detach()
            gt_video_neg_gei = generate_gei_2(gt_neg_video).detach()
            gt_video_gei = generate_gei_2(gt_silt_video).detach()
            
            associated = torch.cat((gt_video_gei, gt_video_pos_gei), 1)
            no_associated = torch.cat((gt_video_gei, gt_video_neg_gei), 1)
            faked = torch.cat((gt_video_gei, rec_gei), 1)
            
            out_assd = self.netA(associated)
            out_noassd = self.netA(no_associated)
            fake_digit = self.netA(faked)
            lossA_assd = F.binary_cross_entropy(out_assd, torch.ones_like(out_assd))
            lossA_noassd = F.binary_cross_entropy(out_noassd, torch.zeros_like(out_noassd))
            lossA_faked = F.binary_cross_entropy(fake_digit, torch.zeros_like(fake_digit))
            
            lossDA = lossA_assd + lossA_noassd + lossA_faked
            
            self.add_summary(self.assd_writer,'loss/lossA_assd',lossA_assd.item())
            self.add_summary(self.assd_writer,'loss/lossA_noassd',lossA_noassd.item())
            self.add_summary(self.assd_writer,'loss/lossA_faked',lossA_faked.item())
            
            self.optimA.zero_grad()
            lossDA.backward()
            self.optimA.step()
            
            # generator adversarial loss
            gen_vid_feat = self.netD(pred_silt_video)
            # gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
            gan_loss = -torch.mean(gen_vid_feat)
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_loss
            self.add_summary(
                self.gen_writer, 'loss/gan_loss', gan_loss.item())

            # generator img loss
            if self.config['losses']["imgloss_type"] == 'L2':
                valid_loss = self.l2_loss(pred_silt_video, gt_silt_video)
            else:
                valid_loss = self.l1_loss(pred_silt_video, gt_silt_video)
            valid_loss = valid_loss * self.config['losses']['valid_weight']
            # lossGA = F.binary_cross_entropy(fake_digit, torch.ones_like(fake_digit))
            # gen_loss =gen_loss+ valid_loss + 0.1*lossGA
            gen_loss =gen_loss+ valid_loss
            self.add_summary(
                self.gen_writer, 'loss/valid_loss', valid_loss.item())
            
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()
            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    # f"d: {dis_loss.item():.3f}; assd: {lossDA.item():.3f};"
                    f"d: {dis_loss.item():.3f};"
                    f"gen: {gen_loss.item():.3f}")
                )
            # saving test figures
            if self.iteration % self.train_args['save_fig_freq'] == 0:
                self.save_fig(self.iteration,pred_silt_video.view(b,t, c, h, w),\
                    gt_silt_video.view(b,t, c, h, w),occ_silt_video)
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
                self.gait_eval.eval(self.netG,self.gait_model)
            if self.iteration > self.train_args['iterations']:
                break
        
    def _valid_epoch(self):
        device = self.config['device']
        for iter ,(occ_silt_video , gt_silt_video , rec_silt_paths) in  enumerate(self.test_loader):
            occ_silt_video, gt_silt_video = occ_silt_video.to(device).float(),gt_silt_video.to(device).float()
            b, t, c, h, w = occ_silt_video.size()
            rec_silt_video = self.netG(occ_silt_video)
            if iter%40 == 0:
                self.save_test_fig(iter,rec_silt_video.view(b,t, c, h, w),gt_silt_video)

