import torch.nn as nn
import torch.optim
from lf_denoising.multi_path_abstract import MultiPathAbstract
from sub_network import SRDTrans
from lf_denoising.cbam_fusion import ResGroup
from lf_denoising.rstb_fusion import SwinIR


class LFDenoising(MultiPathAbstract):

    def __init__(self,
                 img_dim,
                 img_time,
                 in_channel=1,
                 embedding_dim=128,
                 num_heads=8,
                 hidden_dim=128*4,
                 window_size=7,
                 num_transBlock=1,
                 attn_dropout_rate=0.1,
                 lr=1e-4,
                 b1=0.5,
                 b2=0.999,
                 f_maps=[16, 32, 64],
                 input_dropout_rate=0.1,
                 step_size=5,
                 gamma=0.3
                 ):
        net_list = []
        optim_list = []
        scheduler_list = []
        for _ in range(2):
            net = SRDTrans(
                img_dim, img_time, in_channel, embedding_dim, window_size, num_heads, hidden_dim, num_transBlock,
                attn_dropout_rate, f_maps, input_dropout_rate
            )
            net_list.append(net)
            op = torch.optim.Adam(net.parameters(),
                                  lr=lr, betas=(b1, b2))
            optim_list.append(op)
            scheduler_list.append(torch.optim.lr_scheduler.StepLR(op, step_size=step_size, gamma=gamma))

        super().__init__(net_list, optim_list, scheduler_list)

class FusionModule(nn.Module):
    def __init__(self, inplanes=4 * 81, planes=81, stride=1, depths=None):
        super(FusionModule, self).__init__()
        self.ln1 = nn.LayerNorm(inplanes)
        self.conv_begin = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=(1, 1))

        self.ResGroup_1 = ResGroup(planes, planes)
        self.ResGroup_2 = ResGroup(planes, planes)
        self.ResGroup_3 = ResGroup(planes, planes)
        self.ResGroup_4 = ResGroup(planes, planes)
        self.ResGroup_5 = ResGroup(planes, planes)

        self.final = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(1, 1))

        if depths is None:
            if inplanes % 6 == 0:
                depths_n_heads = [6, 6, 6, 6, 6, 6, 6, 6]
                im_size = 48
            elif inplanes % 49 == 0:
                depths_n_heads = [7, 7, 7, 7, 7, 7]
                im_size = 64
            else:
                depths_n_heads = [8, 8, 8, 8, 8, 8]
                im_size = 64
        else:
            depths_n_heads = depths
            im_size = 48
        self.model = SwinIR(upscale=2, in_chans=inplanes, img_size=im_size, window_size=8,
                         img_range=1., depths=depths_n_heads, embed_dim=inplanes,
                         num_heads=depths_n_heads,
                         mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, xs, yt):
        x = torch.cat([xs, yt], dim=1)
        swinir = self.model(x)
        out = self.conv_begin(x)

        out = self.ResGroup_1(out)
        out = self.ResGroup_2(out)
        out = self.ResGroup_3(out)
        out = self.ResGroup_4(out)
        out = self.ResGroup_5(out)

        out = self.final(out)

        out = out + self.conv_begin(swinir)

        return out
