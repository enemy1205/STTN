from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 16, [4,4,4], [2,2,2], [1,1,1])

        self.conv2 = nn.Conv3d(16, 32, [4,4,4], [2,2,2], [1,1,1])
        # self.bn2 = nn.BatchNorm3d(128)
        self.ln2 = nn.LayerNorm([32, 16, 16])

        self.conv3 = nn.Conv3d(32, 64, [4,4,4], [2,2,2], [1,1,1])
        # self.bn3 = nn.BatchNorm3d(256)
        self.ln3 = nn.LayerNorm([64, 8, 8])

        self.conv4 = nn.Conv3d(64, 64, [4,4,4], [2,2,2], [1,1,1])
        # self.bn4 = nn.BatchNorm3d(512)
        self.ln4 = nn.LayerNorm([64, 4, 4])

        self.conv5 = nn.Conv3d(64, 1, [2,4,4], [1,1,1])

        # self.act = nn.Sigmoid()
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        print('initialize Disc 128 model layernorm parameter done!')



    def forward(self, x):
        c1 = self.conv1(x)
        x = F.leaky_relu(c1, 0.2)
        c2 = self.conv2(x)
        l2 = self.ln2(c2.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x = F.leaky_relu(l2, 0.2)
        c3 = self.conv3(x)
        l3 = self.ln3(c3.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x = F.leaky_relu(l3, 0.2)
        c4 = self.conv4(x)
        l4 = self.ln4(c4.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x = F.leaky_relu(l4, 0.2)
        # x = self.act(self.conv5(x))
        x = self.conv5(x)
        return x 



class NetA(nn.Module):
    def __init__(self, nc=3, ndf=96):
        super(NetA, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(nc*2, ndf, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x.view(-1, 1)




class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.encoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        print('init encoder param')
        

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        outputs = [x]
        for layer in self.layers:
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            x = getattr(self, layer)(x)
            if 'conv_' in layer: x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            if 'convlstm' in layer: outputs.append(x)
        return outputs

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.decoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        print('init decoder param')

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh': layers.append(nn.Tanh())
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        elif type == 'deconv':
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, encoder_outputs):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        idx = len(encoder_outputs)-1
        for layer in self.layers:
            if 'conv_' in layer or 'deconv_' in layer:
                x = encoder_outputs[idx]
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
                x = getattr(self, layer)(x)
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            elif 'convlstm' in layer:
                idx -= 1
                x = torch.cat([encoder_outputs[idx], x], dim=2)
                x = getattr(self, layer)(x)
                encoder_outputs[idx] = x
        return x

class ConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Config:
    encoder = [('conv', 'leaky', 1, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 128, 3, 1, 2),
             ('convlstm', '', 128, 128, 3, 1, 1)]
    decoder = [('deconv', 'leaky', 128, 64, 4, 1, 2),
               ('convlstm', '', 128, 64, 3, 1, 1),
               ('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 32, 4, 1, 2),
               ('convlstm', '', 33, 32, 3, 1, 1),
               ('conv', 'tanh', 32, 1, 1, 0, 1)]

class Config_64:
    encoder = [('conv', 'leaky', 1, 16, 3, 1, 2),
             ('convlstm', '', 16, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1)]
    decoder = [('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 16, 4, 1, 2),
               ('convlstm', '', 32, 16, 3, 1, 1),
               ('deconv', 'leaky', 16, 16, 4, 1, 2),
               ('convlstm', '', 17, 16, 3, 1, 1),
               ('conv', 'tanh', 16, 1, 1, 0, 1)]


""""
加深网络.128
"""
class Config_128_myself:
    
    encoder = [('conv', 'leaky', 1, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 128, 3, 1, 2),
            #  ('conv', 'leaky', 128, 128, 3, 1, 2),
             ('convlstm', '', 128, 128, 3, 1, 1)]
    decoder = [
               ('deconv', 'leaky', 64, 64, 4, 1, 1),
               ('deconv', 'leaky', 128, 64, 4, 1, 2),
               ('convlstm', '', 128, 64, 3, 1, 1),
               ('deconv', 'leaky', 32, 32, 4, 1, 1),
               ('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 32, 4, 1, 2),
            #    ('deconv', 'leaky', 32, 32, 4, 1, 1),
               ('convlstm', '', 33, 32, 3, 1, 1),
               ('conv', 'tanh', 32, 1, 1, 0, 1)]
""""
加深网络,64, 加深卷积不影响性能
"""
class Config_64_myself:
    
    encoder = [
             ('conv', 'leaky', 1, 16, 3, 1, 2),
             ('conv', 'leaky', 16, 16, 3, 1, 1),
             ('convlstm', '', 16, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('conv', 'leaky', 32, 32, 3, 1, 1),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('conv', 'leaky', 64, 64, 3, 1, 1),
             ('convlstm', '', 64, 64, 3, 1, 1)]
    decoder = [('deconv', 'leaky', 64, 64, 4, 1, 1),
               ('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 32, 4, 1, 1),
               ('deconv', 'leaky', 32, 16, 4, 1, 2),
               ('convlstm', '', 32, 16, 3, 1, 1),
               ('deconv', 'leaky', 16, 16, 4, 1, 1),
               ('deconv', 'leaky', 16, 16, 4, 1, 2),
               ('convlstm', '', 17, 16, 3, 1, 1),
               ('conv', 'tanh', 16, 1, 1, 0, 1)]

""""
在64基础上增大feature dim 128
"""

class Config_64expand:
    
    encoder = [('conv', 'leaky', 1, 16, 3, 1, 2),
             ('convlstm', '', 16, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 128, 3, 1, 1),
             ('convlstm', '', 128, 128, 3, 1, 1)]
    decoder = [
               ('deconv', 'leaky', 128, 64, 3, 1, 1),
               ('convlstm', '', 128, 64, 3, 1, 1),
               ('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 16, 4, 1, 2),
               ('convlstm', '', 32, 16, 3, 1, 1),
               ('deconv', 'leaky', 16, 16, 4, 1, 2),
               ('convlstm', '', 17, 16, 3, 1, 1),
               ('conv', 'tanh', 16, 1, 1, 0, 1)]

""""
在64基础上增大feature dim 256
"""

class Config_64expand256:
    
    encoder = [('conv', 'leaky', 1, 16, 3, 1, 2),
             ('convlstm', '', 16, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 128, 3, 1, 1),
             ('convlstm', '', 128, 128, 3, 1, 1),
             ('conv', 'leaky', 128, 256, 3, 1, 1),
             ('convlstm', '', 256, 256, 3, 1, 1)]
    decoder = [
               ('deconv', 'leaky', 256, 128, 3, 1, 1),
               ('convlstm', '', 256, 128, 3, 1, 1),
               ('deconv', 'leaky', 128, 64, 3, 1, 1),
               ('convlstm', '', 128, 64, 3, 1, 1),
               ('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 16, 4, 1, 2),
               ('convlstm', '', 32, 16, 3, 1, 1),
               ('deconv', 'leaky', 16, 16, 4, 1, 2),
               ('convlstm', '', 17, 16, 3, 1, 1),
               ('conv', 'tanh', 16, 1, 1, 0, 1)]

class Config_64expand512:
    
    encoder = [('conv', 'leaky', 1, 8, 3, 1, 2),
             ('convlstm', '', 8, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 64, 3, 1, 2),
             ('convlstm', '', 64, 128, 3, 1, 1),
             ('conv', 'leaky', 128, 128, 3, 1, 2),
             ('convlstm', '', 128, 256, 3, 1, 1),
             ('conv', 'leaky', 256, 256, 3, 1, 1),
             ('convlstm', '', 256, 512, 3, 1, 1),
            ]
            #  ('conv', 'leaky', 128, 256, 3, 1, 1),
            #  ('convlstm', '', 256, 256, 3, 1, 1)]
    decoder = [
               ('deconv', 'leaky', 512, 256, 3, 1, 1),
               ('convlstm', '', 512, 256, 3, 1, 1),
               ('deconv', 'leaky', 256, 128, 4, 1, 2),
               ('convlstm', '', 256, 128, 3, 1, 1),
               ('deconv', 'leaky', 128, 64, 4, 1, 2),
               ('convlstm', '', 128, 64, 3, 1, 1),
               ('deconv', 'leaky', 64, 8, 4, 1, 2),
               ('convlstm', '', 9, 8, 3, 1, 1),
            #    ('deconv', 'leaky', 32, 32, 3, 1, 1),
            #    ('convlstm', '', 33, 32, 3, 1, 1),
               ('conv', 'tanh', 8, 1, 1, 0, 1)]
config = Config()
config_64 = Config_64()
config_128_myself = Config_64()
config_64_myself = Config_64_myself()
config_64_expand = Config_64expand()
config_64_expand256 = Config_64expand256()
config_64_expand512 = Config_64expand512()


"""
可以选择embedding size为128或者64的模型
"""
def convlstm_model(config = config):
    return ConvLSTM(config)

def convlstm_model_64(config = config_64):
    return ConvLSTM(config)
def convlstm_model_64_myself(config = config_64_myself):
    return ConvLSTM(config)
def convlstm_model_64_expand(config = config_64_expand):
    print("using convlstm64 model expand 128 version")
    return ConvLSTM(config)
def convlstm_model_64_expand256(config = config_64_expand256):
    print("using convlstm64 model expand 256 version")
    return ConvLSTM(config)
def convlstm_model_64_expand512(config = config_64_expand512):
    print("using convlstm64 model expand 512 version")
    return ConvLSTM(config)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']= '1'

    model = convlstm_model_64_expand()
    inp = torch.Tensor(20, 32, 1, 64, 64)

    out = model(inp)
    print(out.shape)
