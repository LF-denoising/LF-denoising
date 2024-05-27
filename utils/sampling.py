import torch
from einops import rearrange
import numpy as np
# import unfoldNd

operation_seed_counter = 0

def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, t, h, w = img.shape
    mask1 = torch.zeros(size=(n * t * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * t * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask3 = torch.zeros(size=(n * t * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor([
        [0, 1, 2], [0, 2, 1], 
        [1, 0, 3], [1, 3, 0], 
        [2, 0, 3], [2, 3, 0],
        [3, 2, 1], [3, 1, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * t * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * t * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    # [n * h // 2 * w // 2, ]
    rd_pair_idx = idx_pair[rd_idx]
    # [n * t * h // 2 * w // 2, 2]

    rd_pair_idx += torch.arange(start=0,
                                end=n * t * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    mask3[rd_pair_idx[:, 2]] = 1
    return mask1, mask2, mask3


def generate_subimages(img, mask):
    n, c, t, h, w = img.shape
    img = rearrange(img, 'b c s h w -> (b s) c h w')
    subimage = torch.zeros(n*t,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)   # nt c=1*4 h w
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)   # nt h w c=1*4
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n*t, h // 2, w // 2, c).permute(0, 3, 1, 2)     # nt c=1 h w

    subimage = rearrange(subimage, '(n t) c h w -> n c t h w', n=n, t=t)
    return subimage


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    # cuda
    g_cuda_generator = torch.Generator(device="cpu")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)
