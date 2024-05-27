"""
Test the LF-denoising model
The Code is created based on the method described in the following paper
    ZHI LU, WENTAO CHEN etc.
    Self-supervised light-field denoising empowers high-sensitivity fast 3D fluorescence imaging without
    temporal dependency
    Submitted, 2024.
    Contact: ZHI LU (luzhi@tsinghua.edu.cn)
"""
from lf_denoising import LFDenoising, FusionModule
from utils.sampling import *
from utils.data_process import LF_2_buck, im_resize, test_preprocess, testset, singlebatch_test_save, multibatch_test_save
import os
import torch
import argparse
import time
import datetime
import numpy as np
import tifffile as tiff
import re
from torch.utils.data import DataLoader
from utils.save_params import save_yaml
from utils.round_view import RoundView


def parse_idx_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        vals = list(range(int(m.group(1)), int(m.group(2))+1))
    else:
        vals = s.split(',')
    return [f"test_No{x}.tif" for x in vals]


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100,
                    help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0',
                    help="the index of GPU you will use for computation (e.g. '0')")
parser.add_argument('--patch_s', type=int, default=128,
                    help="the spatial size of 3D patches (patch size in x and y dimension)")
parser.add_argument('--total_view', type=int, default=169,
                    help="the angular size of LF images, should be a square of an odd integer. default 169 (13*13)")
parser.add_argument('--overlap_factor', type=float, default=0.25,
                    help="the overlap factor between two adjacent patches, default 0.25")
parser.add_argument('--batch_size', type=int,
                    default=1, help="the batch_size, limited by your VRAM, default 1")
parser.add_argument('--radius', type=int, default=5,
                    help="the radius of selected LF images, default 5")
parser.add_argument('--scale_factor', type=int, default=1,
                    help='the factor for image intensity scaling, default 1')
parser.add_argument('--datasets_path', type=str,
                    default='datasets', help="dataset root path")
parser.add_argument('--datasets_folder', type=str, default='example_test',
                    help="A folder containing files to be tested")
parser.add_argument('--output_dir', type=str,
                    default='results', help="output directory")
parser.add_argument('--pth_path', type=str,
                    default='pth', help="pth file root path")
parser.add_argument('--ckp_idx', type=int, default=25,
                    help="idx of checkpoint")
parser.add_argument('--img_list', type=parse_idx_range, default='0-5',
                    help="the number of images to be denoised (e.g. '0', '0,1,3,5' or '0-50')")
parser.add_argument('--denoise_model', type=str, default='test_model',
                    help='A folder containing models to be tested')
parser.add_argument('--subname', type=str, default='',
                    help='A suffix append to output directory')
opt = parser.parse_args()

img_list = opt.img_list

opt.cut_uv = 2
opt.upscale = 1
opt.export_uint16 = True

one_side_total_view = np.sqrt(opt.total_view)
assert one_side_total_view ** 2 == opt.total_view

# use isotropic patch size by default
rv = RoundView(opt.radius, total_view=int(one_side_total_view))
# for u-net compatibility, patch_y is the dimension of selected views
opt.patch_y = rv.size
opt.patch_x = opt.patch_s
opt.patch_t = opt.patch_s

opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))

print('\033[1;31mParameters -----> \033[0m')
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
model_path = opt.pth_path + '//' + opt.denoise_model

model_list = list(os.walk(model_path, topdown=False))[-1][-1]
model_list.sort()

# read paremeters from file
for i in range(len(model_list)):
    aaa = model_list[i]
    if '.yaml' in aaa:
        yaml_name = model_list[i]
        del model_list[i]


model_name_phrase = model_list[0].split("_")

model_list = []
for i in range(opt.ckp_idx, opt.ckp_idx + 1):
    ckp_idx = "%02d" % (i)
    model_name_phrase[1] = ckp_idx
    model_list.append("_".join(model_name_phrase))

print('\033[1;31mStacks for processing -----> \033[0m')
print('Total stack umber -----> ', len(img_list))


if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
output_path = opt.output_dir + '//' + opt.denoise_model
if opt.subname:
    output_path += "_" + opt.subname
if not os.path.exists(output_path):
    os.mkdir(output_path)

yaml_name = output_path + '//para.yaml'
save_yaml(opt, yaml_name)

denoise_generator = LFDenoising(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128 * 4,
    lr=1e-5,
    b1=0.5,
    b2=0.999,
    window_size=7,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

fusion_layer = FusionModule(inplanes=2 * opt.patch_y, planes=opt.patch_y)


def test():
    # Start processing
    for pth_index in range(len(model_list)):
        aaa = model_list[pth_index]
        if '.pth' in aaa:
            pth_name = model_list[pth_index]

            # load model
            model_name = opt.pth_path + '//' + opt.denoise_model + \
                '//' + pth_name.replace('_0.pth', '')
            denoise_generator.load(model_name)
            fusion_layer.load_state_dict(torch.load(
                model_name + "_fl.pth", map_location="cpu"))
            denoise_generator.train(False)
            fusion_layer.train(False)

            denoise_generator.cuda()
            fusion_layer.cuda()

            opt.img_list = img_list
            # test all stacks
            for N in range(len(img_list)):
                print("Input img is " + img_list[N])
                name_list, noise_img, coordinate_list = test_preprocess(
                    opt, N, rv)

                prev_time = time.time()
                time_start = time.time()

                denoise_img = np.zeros(noise_img.shape)

                test_data = testset(name_list, coordinate_list, noise_img, rv)
                testloader = DataLoader(
                    test_data, batch_size=opt.batch_size, shuffle=False)
                with torch.no_grad():
                    for iteration, (xs_volume, yt_volume, single_coordinate) in enumerate(
                            testloader):
                        xs_volume = xs_volume.cuda()
                        yt_volume = yt_volume.cuda()

                        xs_predict, yt_predict = denoise_generator.forward(
                            [xs_volume, yt_volume])

                        xs_predict = torch.squeeze(xs_predict, 1)
                        xs_predict_from_yt = torch.squeeze(
                            yt_predict, 1).permute(0, 3, 2, 1)

                        xs_predict_from_yt = xs_predict_from_yt[:,
                                                                :, rv.cxt_2_cxs_index_list, :]

                        xs_predict = xs_predict.permute(0, 2, 1, 3)
                        xs_predict_from_yt = xs_predict_from_yt.permute(
                            0, 2, 1, 3)

                        predict = fusion_layer(xs_predict, xs_predict_from_yt)
                        predict = predict.permute(0, 2, 1, 3)

                        batches_done = iteration
                        batches_left = 1 * len(testloader) - batches_done
                        time_left_seconds = int(
                            batches_left * (time.time() - prev_time))

                        prev_time = time.time()

                        if iteration % 1 == 0:
                            time_end = time.time()
                            time_cost = time_end - time_start
                            print(
                                '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                                % (
                                    pth_index + 1,
                                    len(model_list),
                                    pth_name,
                                    N + 1,
                                    len(img_list),
                                    img_list[N],
                                    iteration + 1,
                                    len(testloader),
                                    time_cost,
                                    time_left_seconds
                                ), end=' ')

                        if (iteration + 1) % len(testloader) == 0:
                            print('\n', end=' ')

                        output_image = np.squeeze(
                            predict.cpu().detach().numpy())

                        raw_image = np.squeeze(
                            xs_volume.cpu().detach().numpy())

                        if output_image.ndim == 3:
                            stack_cnt = 1
                        else:
                            stack_cnt = output_image.shape[0]

                        if stack_cnt == 1:
                            stack_img, _, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                                single_coordinate, output_image, raw_image)
                            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                                = stack_img
                        else:
                            for i in range(stack_cnt):
                                stack_img, _, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                                    single_coordinate, i, output_image, raw_image)
                                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                                    = stack_img

                    del noise_img

                    output_img = denoise_img.squeeze().astype(np.float32) * opt.scale_factor

                    output_img = output_img.transpose(
                        (1, 0, 2))[rv.cxs_2_o_index_list]

                    output_img = rv.rv_stack_2_lf(output_img)

                    if opt.upscale != 1:
                        output_img = im_resize(output_img, 1 / opt.upscale)

                    if opt.export_uint16:
                        output_img = np.clip(
                            output_img, 0, 65535).astype(np.uint16)

                    if output_img.ndim == 4:
                        output_img = LF_2_buck(output_img)

                    result_name = output_path + '//' + img_list[N]
                    tiff.imsave(result_name, output_img)

                    print("test result saved in:", result_name)


if __name__ == "__main__":
    test()
