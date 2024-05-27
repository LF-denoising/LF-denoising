import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from glob import glob
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from utils.round_view import RoundView


def im_resize(stack_4d, upscale_factor):
    assert stack_4d.ndim == 4
    new_shape = (round(stack_4d.shape[2] * upscale_factor),
                 round(stack_4d.shape[3] * upscale_factor))
    input = torch.from_numpy(stack_4d.astype(np.float32))
    res = interpolate(input, size=new_shape,
                      mode="bilinear", align_corners=True)

    if stack_4d.dtype == np.uint16:
        res = torch.clamp(res, 0, 65535)

    return res.numpy().astype(stack_4d.dtype)


def random_transform(input):
    p_trans = random.randrange(8)
    if p_trans == 0:  # no transformation
        input = input
    elif p_trans == 1:  # left rotate 90
        input = np.rot90(input, k=1, axes=(0, 2))
    elif p_trans == 2:  # left rotate 180
        input = np.rot90(input, k=2, axes=(0, 2))
    elif p_trans == 3:  # left rotate 270
        input = np.rot90(input, k=3, axes=(0, 2))
    elif p_trans == 4:  # horizontal flip
        input = input[:, :, ::-1]
    elif p_trans == 5:  # horizontal flip & left rotate 90
        input = input[:, :, ::-1]
        input = np.rot90(input, k=1, axes=(0, 2))
    elif p_trans == 6:  # horizontal flip & left rotate 180
        input = input[:, :, ::-1]
        input = np.rot90(input, k=2, axes=(0, 2))
    elif p_trans == 7:  # horizontal flip & left rotate 270
        input = input[:, :, ::-1]
        input = np.rot90(input, k=3, axes=(0, 2))
    return input


def get_input_target_single(raw, start=None):
    if start is None:
        start = random.randint(0, 1)
    assert raw.shape[
        -3] % 2 == 0, f"The raw volume cannot be equally divided (The shape is {raw.shape})"

    input = raw[..., start::2, :, :]
    target = raw[..., 1 - start::2, :, :]
    return input, target


def get_input_target_pair(vol1, vol2, start=None):
    if start is None:
        start = random.randint(0, 1)
    i1, t1 = get_input_target_single(vol1, start=start)
    i2, t2 = get_input_target_single(vol2, start=start)
    return i1, t1, i2, t2, start


def get_multi_target(xs_volume, y_start, x_start):
    target = xs_volume[1 - x_start::2, :, 1 - y_start::2]
    LF = np.transpose(target, (1, 0, 2))

    return LF


class trainset(Dataset):
    def __init__(
            self, name_list, coordinate_list,
            noise_img_all, stack_index, rv: RoundView
    ):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index
        self.rv = rv

    def __getitem__(self, index):
        # fn = self.images[index]
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]

        noise_img = random_transform(noise_img)

        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        xs_volume = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        xt_volume = xs_volume[:, self.rv.cxs_2_cxt_index_list]

        xs_volume = torch.from_numpy(np.expand_dims(xs_volume, 0).copy())
        xt_volume = torch.from_numpy(np.expand_dims(xt_volume, 0).copy())

        return xs_volume, xt_volume

    def __len__(self):
        return len(self.name_list)


class Trainset2Net(trainset):

    def get_input_target_pair(self, raw, start=None):
        if start is None:
            start = random.randint(0, 1)
        if raw.shape[1] % 2 == 1:
            raw = raw[:, 1:] if start == 0 else raw[:, :-1]

        input = raw[:, start::2]
        target = raw[:, 1 - start::2]
        return input, target, start

    def __getitem__(self, index):
        xs_volume, xt_volume = super().__getitem__(index)
        yt_volume = xt_volume.permute((0, 3, 2, 1))
        xs_i, xs_t, start = self.get_input_target_pair(xs_volume)
        yt_i, yt_t, _ = self.get_input_target_pair(yt_volume, start=start)
        # xt_i, xt_t, start = self.get_input_target_pair(xt_volume, start=start)
        return xs_volume, yt_volume, xs_i, xs_t, yt_i, yt_t, start


class testset(Dataset):
    def __init__(self, name_list, coordinate_list, noise_img, rv: RoundView):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img = noise_img
        self.rv = rv

    def __getitem__(self, index):
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        xs_volume = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        xt_volume = xs_volume[:, self.rv.cxs_2_cxt_index_list]
        yt_volume = xt_volume.transpose((2, 1, 0))

        xs_volume = torch.from_numpy(np.expand_dims(xs_volume, 0).copy())
        yt_volume = torch.from_numpy(np.expand_dims(yt_volume, 0).copy())

        return xs_volume, yt_volume, single_coordinate

    def __len__(self):
        return len(self.name_list)


def buck_169_2_LF(im):
    assert im.ndim == 3
    assert im.shape[0] == 169
    return im.reshape(13, 13, *im.shape[1:])


def LF_2_buck(im):
    assert im.ndim == 4
    assert im.shape[0] == im.shape[1]
    return im.reshape(-1, *im.shape[2:])


def train_preprocess_lessMemoryMulStacks(args, rv: RoundView):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t = args.patch_t
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t = args.gap_t

    im_folder = args.datasets_path + '/' + args.datasets_folder
    im_list = glob(im_folder + "/*.tif")

    noise_im_all = []

    stack_num = len(im_list)
    print('Total stack number -----> ', stack_num)

    from tqdm import tqdm
    ratio = stack_num / args.select_img_num

    if ratio <= 1:
        preread = tqdm(np.arange(0, stack_num, ratio).astype(int))
    else:
        max_start = int(args.select_img_num * (ratio - 1) * 0.5)
        start = random.randint(0, max_start - 1)
        new_ratio = (stack_num - start) / args.select_img_num
        preread = np.arange(start, stack_num, new_ratio).astype(int)
        print(f"Random first img: {start}")
        preread = preread[:args.select_img_num]

    tr = tqdm(preread)

    for i in tr:

        im_dir = im_list[i]
        im_name = os.path.basename(im_dir)

        noise_im = tiff.imread(im_dir)
        if noise_im.ndim == 4 and noise_im.shape[0] == noise_im.shape[1]:
            noise_im = LF_2_buck(noise_im)

        noise_im = noise_im[rv.lf_2_rv_index_list]
        noise_im = noise_im[rv.o_2_cxs_index_list]
        noise_im = noise_im.transpose((1, 0, 2))

        noise_im_all.append(noise_im.astype(np.float32))

    assert gap_y >= 0 and gap_x >= 0 and gap_t >= 0, "train gap size is negative"

    name_list = []
    coordinate_list = {}
    stack_index = []
    ind = 0
    for noise_im in noise_im_all:

        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_t = noise_im.shape[0]

        num_h = math.ceil(
            (whole_y - patch_y + gap_y) / gap_y)
        num_w = math.ceil(
            (whole_x - patch_x + gap_x) / gap_x)
        num_s = math.ceil(
            (whole_t - patch_t + gap_t) / gap_t)

        for x in range(0, num_h):
            for y in range(0, num_w):
                for z in range(0, num_s):
                    single_coordinate = {
                        'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                    if y != num_w - 1:
                        init_w = gap_x * y
                        end_w = gap_x * y + patch_x
                    else:
                        init_w = whole_x - patch_x
                        end_w = whole_x
                    if z != num_s - 1:
                        init_s = gap_t * z
                        end_s = gap_t * z + patch_t
                    else:
                        init_s = whole_t - patch_t
                        end_s = whole_t
                    if x != num_h - 1:
                        init_h = gap_y * x
                        end_h = gap_y * x + patch_y
                    else:
                        init_h = whole_y - patch_y
                        end_h = whole_y

                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s

                    patch_name = args.datasets_folder + '_' + \
                        im_name.replace('.tif', '') + '_x' + \
                        str(x) + '_y' + str(y) + '_z' + str(z)

                    name_list.append(patch_name)

                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind = ind + 1
    return name_list, noise_im_all, coordinate_list, stack_index


def singlebatch_test_save(single_coordinate, output_image, raw_image):
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    aaaa = output_image[patch_start_s:patch_end_s,
                        patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    bbbb = raw_image[patch_start_s:patch_end_s,
                     patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def multibatch_test_save(single_coordinate, id, output_image, raw_image):
    stack_start_w_id = single_coordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w'].numpy()
    stack_end_w = int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w'].numpy()
    patch_start_w = int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w'].numpy()
    patch_end_w = int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate['stack_start_h'].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate['stack_end_h'].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate['patch_end_h'].numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = single_coordinate['stack_start_s'].numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = single_coordinate['stack_end_s'].numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = single_coordinate['patch_start_s'].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = single_coordinate['patch_end_s'].numpy()
    patch_end_s = int(patch_end_s_id[id])

    output_image_id = output_image[id]
    raw_image_id = raw_image[id]
    aaaa = output_image_id[patch_start_s:patch_end_s,
                           patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    bbbb = raw_image_id[patch_start_s:patch_end_s,
                        patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def test_preprocess(args, N, rv: RoundView):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t2 = args.gap_t
    cut_w = (patch_x - gap_x) / 2
    # cut_h = (patch_y - gap_y) / 2
    cut_h = 0
    cut_s = (patch_t2 - gap_t2) / 2

    assert cut_w >= 0 and cut_h >= 0 and cut_s >= 0, "test cut size is negative!"
    im_folder = args.datasets_path + '//' + args.datasets_folder

    name_list = []
    coordinate_list = {}

    im_name = args.img_list[N]

    im_dir = im_folder + '//' + im_name
    noise_im = tiff.imread(im_dir)

    if noise_im.ndim == 3 and noise_im.shape[0] == 169:
        noise_im = buck_169_2_LF(noise_im)

    if args.upscale != 1:
        noise_im = im_resize(noise_im, args.upscale)

    noise_im = LF_2_buck(noise_im)

    noise_im = noise_im[rv.lf_2_rv_index_list][rv.o_2_cxs_index_list].transpose(
        (1, 0, 2))

    noise_im = noise_im.astype(np.float32) / args.scale_factor

    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]

    num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)

    for z in range(0, num_s):
        for x in range(0, num_h):
            for y in range(0, num_w):
                single_coordinate = {'init_h': 0, 'end_h': 0,
                                     'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h - 1):
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                elif x == (num_h - 1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w - 1):
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                elif y == (num_w - 1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s - 1):
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                elif z == (num_s - 1):
                    init_s = whole_t - patch_t2
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y * gap_x
                    single_coordinate['stack_end_w'] = y * \
                        gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x - cut_w
                elif y == num_w - 1:
                    single_coordinate['stack_start_w'] = whole_x - \
                        patch_x + cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y * gap_x + cut_w
                    single_coordinate['stack_end_w'] = y * \
                        gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x - cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x * gap_y
                    single_coordinate['stack_end_h'] = x * \
                        gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y - cut_h
                elif x == num_h - 1:
                    single_coordinate['stack_start_h'] = whole_y - \
                        patch_y + cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x * gap_y + cut_h
                    single_coordinate['stack_end_h'] = x * \
                        gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y - cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z * gap_t2
                    single_coordinate['stack_end_s'] = z * \
                        gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s
                elif z == num_s - 1:
                    single_coordinate['stack_start_s'] = whole_t - \
                        patch_t2 + cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2
                else:
                    single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
                    single_coordinate['stack_end_s'] = z * \
                        gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s

                patch_name = args.datasets_folder + '_x' + \
                    str(x) + '_y' + str(y) + '_z' + str(z)

                name_list.append(patch_name)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list
