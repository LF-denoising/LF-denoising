"""
Train the LF-denoising model
The Code is created based on the method described in the following paper
    ZHI LU, WENTAO CHEN etc.
    Self-supervised light-field denoising empowers high-sensitivity fast 3D fluorescence imaging without
    temporal dependency
    Submitted, 2024.
    Contact: ZHI LU (luzhi@tsinghua.edu.cn)
"""

from utils.sampling import generate_mask_pair, generate_subimages
from utils.save_params import save_yaml
from utils.data_process import train_preprocess_lessMemoryMulStacks, Trainset2Net
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import os
import numpy as np
import torch.cuda
from torch import optim
from lf_denoising import LFDenoising, FusionModule
from utils.sampling import generate_mask_pair, generate_subimages
from utils.round_view import RoundView


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='test_model',
                    help="name of trained model")
parser.add_argument('--GPU', type=str, default='0',
                    help="the index of GPU you will use for computation (e.g. '0')")
parser.add_argument('--patch_s', type=int, default=128,
                    help="the spatial size of 3D patches (patch size in x and y dimension)")
parser.add_argument('--total_view', type=int, default=169,
                    help="the angular size of LF images, should be a square of an odd integer. default 169 (13*13)")
parser.add_argument('--radius', type=int, default=5,
                    help="the radius of selected LF images, default 5")
parser.add_argument('--overlap_factor', type=float, default=0.5,
                    help="the overlap factor between two adjacent patches, default 0.5")
parser.add_argument('--batch_size', type=int,
                    default=1, help="the batch_size, limited by your VRAM, default 1")
parser.add_argument("--n_epochs", type=int, default=30,
                    help="number of training epochs")
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--step_size', type=float, default=5,
                    help="step size for training, default 5")
parser.add_argument('--step_gamma', type=float, default=0.3,
                    help="gamma for training, default 0.3")
parser.add_argument("--b1", type=float, default=0.5,
                    help="Adam: bata1, default 0.5")
parser.add_argument("--b2", type=float, default=0.999,
                    help="Adam: bata2, default 0.999")
parser.add_argument('--mask_fusion_loss', type=bool, default=True,
                    help="use mask loss for fusion layers training, default True")
parser.add_argument('--scale_factor', type=int, default=1,
                    help='the factor for image intensity scaling, default 1')
parser.add_argument('--datasets_path', type=str, default='datasets',
                    help="dataset root path")
parser.add_argument('--datasets_folder', type=str, default='example',
                    help="A folder containing files for training")
parser.add_argument('--output_dir', type=str,
                    default='results', help="output directory")
parser.add_argument('--pth_path', type=str,
                    default='pth', help="pth file root path")
parser.add_argument('--key_word', type=str, default='',
                    help="suffix appended to model name")
parser.add_argument('--select_img_num', type=int, default=50,
                    help='select the number of images used for training')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU

one_side_total_view = np.sqrt(opt.total_view)
assert one_side_total_view ** 2 == opt.total_view

# use isotropic patch size by default
rv = RoundView(opt.radius, total_view=int(one_side_total_view))
# for u-net compatibility, patch_y is the dimension of selected views
opt.patch_y = rv.size
opt.patch_x = opt.patch_s
opt.patch_t = opt.patch_s

opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))
opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
opt.batch_size = opt.batch_size
print('\033[1;31mTraining parameters -----> \033[0m')
print(opt)


if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)
current_time = opt.datasets_folder + '_' + opt.key_word + \
    '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
if opt.model_name != "test_model":
    pth_path = opt.pth_path + '/' + opt.model_name + '_' + current_time
else:
    pth_path = opt.pth_path + '/' + opt.model_name
print("ckp is saved in {}".format(pth_path))
if not os.path.exists(pth_path):
    os.makedirs(pth_path)


train_name_list, train_noise_img, train_coordinate_list, stack_index = train_preprocess_lessMemoryMulStacks(
    opt, rv)

yaml_name = pth_path + '//para.yaml'
save_yaml(opt, yaml_name)

L1_pixelwise = nn.L1Loss()
L2_pixelwise = nn.MSELoss()

denoise_generator = LFDenoising(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128 * 4,
    window_size=7,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    lr=opt.lr,
    b1=opt.b1,
    b2=opt.b2,
    f_maps=[16, 32, 64],
    input_dropout_rate=0,
    step_size=opt.step_size,
    gamma=opt.step_gamma
)

fusion_layer = FusionModule(inplanes=2 * opt.patch_y, planes=opt.patch_y)
fusion_optim = optim.Adam(fusion_layer.parameters(),
                          lr=opt.lr, betas=(opt.b1, opt.b2))
fusion_scheduler = optim.lr_scheduler.StepLR(
    fusion_optim, step_size=opt.step_size, gamma=opt.step_gamma)

param_num = sum([param.nelement() for param in denoise_generator.parameters()])
param_num += sum([param.nelement() for param in fusion_layer.parameters()])
print('\033[1;31mParameters of model is {:.2f}M. \033[0m'.format(
    param_num / 1e6))

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU
    denoise_generator.cuda()
    fusion_layer = fusion_layer.cuda()
    L2_pixelwise.cuda()
    L1_pixelwise.cuda()

time_start = time.time()


def train_epoch():
    train_data = Trainset2Net(
        train_name_list, train_coordinate_list, train_noise_img, stack_index, rv)
    trainloader = DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    total_loss_list = []

    prev_time = time.time()

    for iteration, (
            xs_noise, yt_noise, xs_i, xs_t, yt_i, yt_t, start) in enumerate(trainloader):

        xs_i, xs_t, yt_i, yt_t = (xs_i.detach().cuda(), xs_t.detach().cuda(),
                                  yt_i.detach().cuda(), yt_t.detach().cuda())

        xs_noise = xs_noise.permute(0, 1, 3, 2, 4)
        xs_masks_multi = generate_mask_pair(xs_noise)
        multi_target1 = generate_subimages(
            xs_noise, xs_masks_multi[0]).squeeze(1).cuda()

        multi_target2 = generate_subimages(
            xs_noise, xs_masks_multi[1]).squeeze(1).cuda()

        multi_target3 = xs_t[..., 1 -
                             start::2].squeeze(1).permute(0, 2, 1, 3).cuda()

        denoise_generator.train(True)
        denoise_generator.zero_grad()

        xs_predict, yt_predict = \
            denoise_generator.forward([xs_i, yt_i])

        loss_1 = (0.5 * L2_pixelwise(xs_predict, xs_t) +
                  0.5 * L1_pixelwise(xs_predict, xs_t))
        loss_2 = (0.5 * L2_pixelwise(yt_predict, yt_t) +
                  0.5 * L1_pixelwise(yt_predict, yt_t))

        loss_1.backward()
        loss_2.backward()
        denoise_generator.step()
        denoise_generator.train(False)

        xs_predict = torch.squeeze(xs_predict.detach(), 1)[..., start::2]
        yt_predict = torch.squeeze(yt_predict.detach(), 1)[..., start::2]

        xs_predict_from_yt = yt_predict.permute(
            0, 3, 2, 1)[:, :, rv.cxt_2_cxs_index_list, :]

        xs_predict = xs_predict.permute(0, 2, 1, 3)
        xs_predict_from_yt = xs_predict_from_yt.permute(0, 2, 1, 3)

        fusion_layer.train(True)
        fusion_optim.zero_grad()

        predict = fusion_layer(
            xs_predict, xs_predict_from_yt)

        loss_7 = ((0.5 * L2_pixelwise(predict, xs_predict) + 0.5 * L1_pixelwise(predict, xs_predict)) +
                  (0.5 * L2_pixelwise(predict, xs_predict_from_yt) + 0.5 * L1_pixelwise(predict, xs_predict_from_yt))) / 2
        if opt.mask_fusion_loss:
            loss_8 = ((0.5 * L2_pixelwise(predict, multi_target1) + 0.5 * L1_pixelwise(predict, multi_target1)) +
                      (0.5 * L2_pixelwise(predict, multi_target2) + 0.5 * L1_pixelwise(predict, multi_target2)) +
                      (0.5 * L2_pixelwise(predict, multi_target3) + 0.5 * L1_pixelwise(predict, multi_target3))) / 3
        else:
            loss_8 = 0.5 * \
                L2_pixelwise(predict, multi_target3) + 0.5 * \
                L1_pixelwise(predict, multi_target3)

        total_loss = loss_7 * 0.2 + loss_8 * 0.8
        total_loss.backward()
        fusion_optim.step()
        fusion_layer.train(False)

        Total_loss = (loss_1 + loss_2 + total_loss) / 3

        batches_done = epoch * len(trainloader) + iteration
        batches_left = opt.n_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=int(
            batches_left * (time.time() - prev_time)))
        prev_time = time.time()

        total_loss_list.append(Total_loss.item())

        if iteration % 1 == 0:
            time_end = time.time()
            print(
                '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f] [Iter loss: %.2f %.2f - %.2f] [ETA: %s] [Time cost: %.2d s] '
                % (
                    epoch + 1,
                    opt.n_epochs,
                    iteration + 1,
                    len(trainloader),
                    np.mean(total_loss_list),
                    loss_1.item(),
                    loss_2.item(),
                    total_loss.item(),
                    time_left,
                    time_end - time_start
                ), end=' ')

        if (iteration + 1) % len(trainloader) == 0:
            print('\n', end=' ')

        # save model
        if (iteration + 1) % (len(trainloader)) == 0:
            model_save_name = pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(
                4)
            denoise_generator.save(model_save_name)
            torch.save(fusion_layer.state_dict(), model_save_name + "_fl.pth")


if __name__ == "__main__":
    for epoch in range(0, opt.n_epochs):
        train_epoch()
        denoise_generator.scheduler_step()
        fusion_scheduler.step()
