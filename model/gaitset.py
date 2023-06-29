import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        # n*s ,c ,h ,w
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


class GaitSet(nn.Module):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
        Arxiv:  https://arxiv.org/abs/1811.06186
        Github: https://github.com/AbnerHqC/GaitSet
    """

    def __init__(self):
        super(GaitSet, self).__init__()
        in_c=[1, 32, 64, 128]
        self.set_block1=nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2=nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3=nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True))

        self.gl_block2=copy.deepcopy(self.set_block2)
        self.gl_block3=copy.deepcopy(self.set_block3)

        self.set_block1=SetBlockWrapper(self.set_block1)
        self.set_block2=SetBlockWrapper(self.set_block2)
        self.set_block3=SetBlockWrapper(self.set_block3)

        self.set_pooling=PackSequenceWrapper(torch.max)

        self.Head=SeparateFCs(parts_num=62, in_channels=128, out_channels=256)

        self.HPP=HorizontalPoolingPyramid()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # sils [c, b*t 求和, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)    # sils [c,1, b*t 求和, h, w]

        del ipts
        outs = self.set_block1(sils)    # outs [c,num_bin=32, b*t 求和, h/2, w/2]
        gl = self.set_pooling(outs, seqL, options={"dim": 2})[0] # gl [b,num_bin=32, h, w]
        gl = self.gl_block2(gl) # gl [b,num_bin=64, h/2, w/2]

        outs = self.set_block2(outs)     # outs [c,num_bin=64, b*t 求和, h/4, w/4]
        gl = gl + self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs = self.Head(feature)

        return embs
    
    def infer(self, inputs):
        seqL=None
        inputs = inputs.unsqueeze(1)
        x=inputs.permute(2,1,0,3,4)
        del inputs

        outs=self.set_block1(x)
        gl=self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl=self.gl_block2(gl)

        outs=self.set_block2(outs)
        gl=gl+self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl=self.gl_block3(gl)

        outs=self.set_block3(outs)
        outs=self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl=gl+outs

        # Horizontal Pooling Matching, HPM
        feature1=self.HPP(outs)  # [n, c, p]
        feature2=self.HPP(gl)  # [n, c, p]
        feature=torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs=self.Head(feature)

        return embs
    
    
class TemporalAssd(nn.Module):
    def __init__(self):
        super(TemporalAssd, self).__init__() 
        in_c=[1, 32, 64, 128]
        self.set_block1=nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2=nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3=nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True))

        self.gl_block2=copy.deepcopy(self.set_block2)
        self.gl_block3=copy.deepcopy(self.set_block3)

        self.set_block1=SetBlockWrapper(self.set_block1)
        self.set_block2=SetBlockWrapper(self.set_block2)
        self.set_block3=SetBlockWrapper(self.set_block3)

        self.set_pooling=PackSequenceWrapper(torch.max)

        self.Head=SeparateFCs(parts_num=62, in_channels=128, out_channels=256)

        self.HPP=HorizontalPoolingPyramid()

    def forward(self, inputs):
        seqL=None
        inputs = inputs.unsqueeze(1)
        x=inputs.permute(2,1,0,3,4)
        del inputs

        outs=self.set_block1(x)
        gl=self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl=self.gl_block2(gl)

        outs=self.set_block2(outs)
        gl=gl+self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl=self.gl_block3(gl)

        outs=self.set_block3(outs)
        outs=self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl=gl+outs

        # Horizontal Pooling Matching, HPM
        feature1=self.HPP(outs)  # [n, c, p]
        feature2=self.HPP(gl)  # [n, c, p]
        feature=torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs=self.Head(feature)

        return embs