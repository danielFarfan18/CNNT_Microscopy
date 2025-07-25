"""
Extra utilities for CNNT
"""

import os
import cv2
import json
import wandb
import torch
import logging
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

# -------------------------------------------------------------------------------------------------

def simulate_gaussian_blur(image, sigma_range=(1, 8), return_sigma=False):
    """
    Apply Gaussian blur with random sigma from the given range
    
    @args:
        - image (np.ndarray): Input image to blur
        - sigma_range (tuple): Min and max sigma values for blur
        - return_sigma (bool): Whether to return the used sigma value
        
    @returns:
        - blurred_image (np.ndarray): Blurred image
        - sigma (float, optional): Used sigma value if return_sigma=True
    """
    if isinstance(sigma_range, (int, float)):
        sigma = sigma_range
    else:
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    
    if image.ndim == 2:
        blurred = gaussian_filter(image, sigma=sigma)
    elif image.ndim == 3:
        # For time series data (T, H, W)
        blurred = np.zeros_like(image)
        for t in range(image.shape[0]):
            blurred[t] = gaussian_filter(image[t], sigma=sigma)
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    if return_sigma:
        return blurred.astype(image.dtype), sigma
    return blurred.astype(image.dtype)

def simulate_variable_blur(image, sigma_map=None, sigma_range=(1, 8)):
    """
    Apply spatially varying blur to simulate real microscopy conditions
    
    @args:
        - image (np.ndarray): Input image to blur
        - sigma_map (np.ndarray, optional): Spatial map of sigma values
        - sigma_range (tuple): Min and max sigma values if sigma_map is None
        
    @returns:
        - blurred_image (np.ndarray): Spatially varying blurred image
        - sigma_map (np.ndarray): Used sigma map
    """
    if image.ndim == 2:
        H, W = image.shape
        time_series = False
    elif image.ndim == 3:
        T, H, W = image.shape
        time_series = True
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    if sigma_map is None:
        # Create a smooth sigma map with random variations
        base_sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        sigma_map = np.full((H, W), base_sigma)
        
        # Add spatial variations (common in microscopy)
        num_spots = np.random.randint(3, 8)
        for _ in range(num_spots):
            center_x = np.random.randint(0, H)
            center_y = np.random.randint(0, W)
            radius = np.random.randint(min(H, W)//8, min(H, W)//3)
            sigma_variation = np.random.uniform(-2, 2)
            
            y, x = np.ogrid[:H, :W]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            sigma_map[mask] = np.clip(sigma_map[mask] + sigma_variation, 
                                    sigma_range[0], sigma_range[1])
    
    if time_series:
        blurred = np.zeros_like(image)
        for t in range(T):
            blurred[t] = apply_variable_blur_2d(image[t], sigma_map)
    else:
        blurred = apply_variable_blur_2d(image, sigma_map)
    
    return blurred.astype(image.dtype), sigma_map

def apply_variable_blur_2d(image, sigma_map):
    """
    Apply spatially varying Gaussian blur to a 2D image
    """
    H, W = image.shape
    blurred = np.zeros_like(image)
    
    # Use a grid approach for efficiency
    sigma_levels = np.unique(np.round(sigma_map, 1))
    
    for sigma in sigma_levels:
        mask = np.abs(sigma_map - sigma) < 0.05
        if np.any(mask):
            # Create a temporary image with only the relevant region
            temp_image = np.zeros_like(image)
            temp_image[mask] = image[mask]
            
            # Apply blur to the entire image
            blurred_temp = gaussian_filter(temp_image, sigma=sigma)
            
            # Copy back only the relevant region
            blurred[mask] = blurred_temp[mask]
    
    return blurred

def create_multi_scale_blur_dataset(clean_images, blur_config):
    """
    Create a dataset with multiple blur levels for training
    
    @args:
        - clean_images (list): List of clean images
        - blur_config (dict): Configuration for blur simulation
            - 'sigma_ranges': List of sigma ranges for different difficulty levels
            - 'probabilities': Probability of each difficulty level
            - 'variable_blur_prob': Probability of using variable blur
            
    @returns:
        - blurred_images (list): List of blurred images
        - blur_info (list): List of blur parameters used
    """
    blurred_images = []
    blur_info = []
    
    sigma_ranges = blur_config.get('sigma_ranges', [(1, 2), (2, 4), (4, 6), (6, 8)])
    probabilities = blur_config.get('probabilities', [0.4, 0.3, 0.2, 0.1])
    variable_blur_prob = blur_config.get('variable_blur_prob', 0.3)
    
    # Normalize probabilities
    probabilities = np.array(probabilities) / np.sum(probabilities)
    
    for clean_img in clean_images:
        # Choose blur difficulty level
        difficulty_idx = np.random.choice(len(sigma_ranges), p=probabilities)
        sigma_range = sigma_ranges[difficulty_idx]
        
        # Decide between uniform and variable blur
        if np.random.random() < variable_blur_prob:
            blurred_img, sigma_map = simulate_variable_blur(clean_img, sigma_range=sigma_range)
            blur_info.append({
                'type': 'variable',
                'sigma_range': sigma_range,
                'sigma_map': sigma_map,
                'difficulty': difficulty_idx
            })
        else:
            blurred_img, sigma = simulate_gaussian_blur(clean_img, sigma_range=sigma_range, return_sigma=True)
            blur_info.append({
                'type': 'uniform',
                'sigma': sigma,
                'sigma_range': sigma_range,
                'difficulty': difficulty_idx
            })
        
        blurred_images.append(blurred_img)
    
    return blurred_images, blur_info

# -------------------------------------------------------------------------------------------------

def add_shared_args(parser=argparse.ArgumentParser("Argument parser for CNNT")):
    """
    Add shared arguments between training and testing

    @args:
        - parser (argparse): parser object. Defaults to argparse.ArgumentParser("Argument parser for CNNT").

    @rets:
        - parser (argparse): modified parser
    """

    # Model arguments
    parser.add_argument('--blocks', nargs='+', type=int, default=[32, 64, 96], help='number of channels in each resolution layer')
    parser.add_argument("--blocks_per_set", type=int, default=4, help='number of transformer blocks to use per set')
    parser.add_argument("--n_head", type=int, default=8, help='number of transformer heads')
    parser.add_argument("--kernel_size", type=int, default=3, help='size of the square kernel for CNN')
    parser.add_argument("--stride", type=int, default=1, help='stride for CNN (equal x and y)')
    parser.add_argument("--padding", type=int, default=1, help='padding for CNN (equal x and y)')
    parser.add_argument("--dropout_p", type=float, default=0.1, help='pdrop regulization in transformer')
    parser.add_argument("--norm_mode", type=str, default="instance", help='normalization mode, layer or batch or instance or mixed')
    parser.add_argument("--with_mixer", type=int, default=1, help='1 or 0 for having the mixer in CNNT module')
    parser.add_argument("--use_conv_3D", action="store_true", help='if set, 3D convolution is used')

    # Optimization arguments
    parser.add_argument("--loss", nargs='+', type=str, default=["mse", "ssim"], help='What loss to use, mse or ssim or sobel or combinations such as mse_ssim_sobel')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[0.1, 1.0], help='to balance multiple losses, weights can be supplied')

    parser.add_argument("--optim", type=str, default="adamw", help='what optimizer to use, adamw, nadam, sgd')
    parser.add_argument("--global_lr", type=float, default=5e-4, help='step size for the optimizer')
    parser.add_argument("--weight_decay", type=float, default=0.1, help='weight decay for the optimizer')
    parser.add_argument("--beta1", type=float, default=0.90, help='beta1 for the default optimizer')
    parser.add_argument("--beta2", type=float, default=0.95, help='beta2 for the default optimizer')
    parser.add_argument("--no_w_decay", action="store_true", help='option of having batchnorm and bias params not have weight decay on lr')
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help='gradient clip norm, if <=0, no clipping')
    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau", help='ReduceLROnPlateau, StepLR, or OneCycleLR')

    # train specific args
    parser.add_argument("--per_scaling", action="store_true", help='if present uses percent scaling instead of hard values')
    parser.add_argument("--im_value_scale", type=float, nargs='+', default=[0,65536], help='min max values to scale with respect to the scaling type')
    parser.add_argument("--valu_thres", type=float, default=0.002, help='threshold of pixel value between background and foreground')
    parser.add_argument("--area_thres", type=float, default=0.25, help='percentage threshold of area that needs to be foreground')

    parser.add_argument("--run_name", type=str, default=None, help='run name for wandb')
    parser.add_argument("--run_notes", type=str, default=None, help='notes for the current run')

    parser.add_argument("--skip_LSUV", action="store_true", help='skip LSUV for testing')
    parser.add_argument("--no_residual", action="store_true", help='skip long term residual connection or not? (predict image or noise?)')

    parser.add_argument("--train_only", action="store_true", help='no val or dev. used to time training')
    parser.add_argument("--fine_samples", type=int, default=-1, help='samples to use for finetuning. If <=0 then use ratio arg instead')
    parser.add_argument("--time_scale", type=int, default=0, help='range of time for time series data. 0: input is not time data. >0: the range to use. <0 random between 1-32')

    return parser

# -------------------------------------------------------------------------------------------------

def save_model(model, config, epoch):
    """
    wrapper around save model to cover DP
    """

    if config.dp:
        model.module.save(epoch)
    else:
        model.save(epoch)


def save_results(train_loss, val_loss, config, epoch):
    """
    compare and save results if we find a better version
    @args:
        - train_loss: train_loss at the given epoch
        - val_loss: val_loss at the given epoch
        - config: config used for the model
        - epoch: the epoch for the results
    @rets:
        - True: if these results are better and have been saved
        - False: if saved results are better and these results are ignored
    """

    best_path = os.path.join(config.result_path, "best.json")

    if os.path.exists(best_path):
        f = open(best_path, 'r')
        best = json.load(f)

        if val_loss < best["val_loss"]:

            logging.info("Found better results. Saving the config")

            f = open(best_path, 'w')
            json.dump({'train_loss':train_loss,
                    'val_loss':val_loss,
                    'epoch':epoch,
                    'config':dict(config)}, f)

            return True

        return False
    else:

        f = open(best_path, 'w')
        json.dump({'train_loss':train_loss,
                'val_loss':val_loss,
                'epoch':epoch,
                'config':dict(config)}, f)
        return True

# -------------------------------------------------------------------------------------------------

def compute_composed_image(x, y, pred):
    """
    @args:
        - x: the noisy image of shape [T, C, H, W]
        - y: ground truth image of shape [T, C, R*H, R*W], can be upsampled
        - pred: the predicted clean image of shape [T, C, R*H, R*W]
    """

    B, T, C, H, W = y.shape
    composed_res = np.zeros((T, B*H, 3*W))

    x = normalize_image(x, percentiles=(0,100))
    # pred_real = normalize_image(pred_real, values=(np.percentile(y_real, 0), np.percentile(y_real, 100)), clip=True)
    # y_real = normalize_image(y_real, percentiles=(0,100))

    for b in range(B):
        composed_res[:,b*H:(b+1)*H,0:W] = x[b,:,0]
        composed_res[:,b*H:(b+1)*H,W:2*W] = y[b,:,0]
        composed_res[:,b*H:(b+1)*H,2*W:3*W] = pred[b,:,0]

    temp = np.zeros_like(composed_res)
    composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

    return composed_res

def wandb_log_vid(tag, x, y, pred):
    """
    Logs the given ground truth, and predicted clean pair
    @args:
        - tag: string the prepend to the captions
        - x: the noisy image of shape [T, C, H, W]
        - y: ground truth image of shape [T, C, H, W]
        - pred: the predicted clean image of shape [T, C, H, W]
    """
    pre_tag="Nosiy_Pred_GT"

    tag = pre_tag + tag

    composed_res = compute_composed_image(x, y, pred)

    wandb.log({"video": wandb.Video(np.repeat(composed_res[:,np.newaxis], 3, axis=1).astype('uint8'), fps=8, format="gif")})

def normalize_image(image, percentiles=None, values=None, clip=True):
    """
    Normalizes image locally.
    @args:
        - image (np.ndarray or torch.Tensor): the image to normalize
        - percentiles (2 tuple int): pair of percentiles ro normalize with
        - values (2 tuple int): pair of values normalize with
        - clip (bool): whether to clip the resulting values to [0,1] or not
        NOTE: only one of percentiles and values is required
    @rets:
        - n_img (same as input image): the image normalized wrt given params.
    """

    assert (percentiles==None and values!=None) or (percentiles!=None and values==None)

    if type(image)==torch.Tensor:
        image_c = image.cpu().detach().numpy()
    else:
        image_c = image

    if percentiles != None:
        i_min = np.percentile(image_c, percentiles[0])
        i_max = np.percentile(image_c, percentiles[1])
    if values != None:
        i_min = values[0]
        i_max = values[1]

    n_img = (image - i_min)/(i_max - i_min)

    if clip:
        return torch.clip(n_img, 0, 1) if type(n_img)==torch.Tensor else np.clip(n_img, 0, 1)

    return n_img

# -------------------------------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# -------------------------------------------------------------------------------------------------
