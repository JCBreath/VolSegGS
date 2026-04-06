from utils_psnr import peak_signal_noise_ratio, Logger
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray
import cv2
import numpy as np
import lpips
import os
import sys

res = int(sys.argv[1])

gt_path = './data/fivejets_time_rgb_rand_n1000_r1024/test'
# vn_path = './trash/render_test_visnerf'
vn_path = './trash/render_test_kplane'


img_files = os.listdir('{}'.format(gt_path))
print(img_files)
img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[1]))
# img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[3]))
# img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[4]))

# inf_img_files = os.listdir('{}'.format(vn_path))
# inf_img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[1]))
# inf_img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[2]))
# inf_img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[3]))

loss_fn = lpips.LPIPS(net='alex',version='0.1')
loss_fn.cuda()

psnr_scores = []
lpips_scores = []
ssim_scores = []


dim = (res, res)

for idx, filename in enumerate(img_files):
    # var,p1, theta, phi, x, y, z = filename.split('.png')[0].split('_')
    # var,p1,p2, theta, phi, x, y, z = filename.split('.png')[0].split('_')
    # var,p1,p2,p3, theta, phi, x, y, z = filename.split('.png')[0].split('_')
    # p1 = float(p1)
    # p2 = float(p2)
    # theta = float(theta)
    # phi = float(phi)
    # x = float(x)
    # y = float(y)
    # z = float(z)
    # print(filename, idx)
    # continue
    # vn_img_path = '{}/300000-test-{}.png'.format(vn_path, idx)
    # vn_img_path = '{}/{:03d}.png'.format(vn_path, idx)
    # vn_img_path = '{}/{}.png'.format(vn_path, idx)
    # vn_img_path = '{}/test{:03d}.png'.format(vn_path, idx)
    if "gaussian-splatting" in vn_path:
        vn_img_path = f'{vn_path}/{idx:05d}.png'
    elif "instant-ngp" in vn_path or "K-Planes" in vn_path:
        vn_img_path = f'{vn_path}/{idx}.png'
    elif "ViSNeRF" in vn_path or "eg3d" in vn_path or "stylegan" in vn_path or "coordnet" in vn_path or 'insitunet' in vn_path:
        vn_img_path = f'{vn_path}/{idx:03d}.png'
    elif "HexPlane" in vn_path:
        vn_img_path = f'{vn_path}/test{idx:03d}.png'
    else:
        vn_img_path = f'{vn_path}/r_{idx}.png'

    # print(inf_img_files[idx], filename)
    # vn_img_path = '{}/{}'.format(vn_path, inf_img_files[idx])
    gt_img_path = '{}/{}'.format(gt_path, filename)

    vn_img = resize(imread(vn_img_path)[:,:,:3], dim)
    gt_img = imread(gt_img_path)
    # print(gt_img.shape)
    if gt_img.shape[2] == 4:  # Check if the image has an alpha channel
        alpha_channel = gt_img[:, :, 3] / 255.0
        rgb_channels = gt_img[:, :, :3]
        white_background = np.ones_like(rgb_channels) * 255.0
        
        gt_img = rgb_channels * alpha_channel[..., None] + white_background * (1 - alpha_channel[..., None])
    # gt_img = resize(imread(gt_img_path)[:,:,:3], dim)
    # gt_img = lpips.load_image(vn_img_path)
    # print(gt_img)
    imsave(f'./trash/test_img/gt_{idx}.png', gt_img.astype(np.uint8))

    gt_img_path = f'./trash/test_img/gt_{idx}.png'
    gt_img = resize(imread(gt_img_path)[:,:,:3], dim)

    vn_tensor = lpips.im2tensor(cv2.resize(lpips.load_image(vn_img_path), dim)).cuda()
    gt_tensor = lpips.im2tensor(cv2.resize(lpips.load_image(gt_img_path), dim)).cuda()

    # print(resize(lpips.load_image(vn_img_path), dim))

    lpips_score = loss_fn.forward(vn_tensor, gt_tensor).item()
    lpips_scores.append(lpips_score)

    psnr_score = peak_signal_noise_ratio(vn_img, gt_img)
    psnr_scores.append(psnr_score)

    # print(rgb2gray(vn_img).shape, vn_img.min())
    # ssim_score = structuralSimilarityIndex(rgb2gray(vn_img), rgb2gray(gt_img))
    ssim_score = structural_similarity(rgb2gray(vn_img), rgb2gray(gt_img), data_range=1.0)
    ssim_scores.append(ssim_score)

    # print(sum(psnr_scores)/len(psnr_scores), end='\r')
    print(f'{idx} {gt_img_path} {vn_img_path} {psnr_score:.2f} {ssim_score:.4f} {lpips_score:.4f}')

print(sum(psnr_scores)/len(psnr_scores))
print(sum(ssim_scores)/len(ssim_scores))
print(sum(lpips_scores)/len(lpips_scores))
