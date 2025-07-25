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

class AdvancedDefocusSimulator:
    """
    Advanced defocus simulator for microscopy images with multi-sigma capabilities
    Integrates your existing implementation with improvements for CNNT training
    """
    
    def __init__(self):
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    def create_multi_sigma_mask(self, image_shape, sigma_config):
        """
        Create a mask with multiple sigma values for different regions
        
        @args:
            - image_shape (tuple): Shape of the image (height, width)
            - sigma_config (dict): Configuration for sigma distribution
                - 'sigma_ranges': List of (min_sigma, max_sigma) tuples for different regions
                - 'num_regions': (min, max) number of regions
                - 'region_size_range': (min, max) size range for regions
                - 'overlap_allowed': Whether regions can overlap
                
        @returns:
            - sigma_map (np.ndarray): Map where each pixel has a sigma value
            - region_info (list): Information about each region
        """
        height, width = image_shape
        sigma_map = np.zeros(image_shape, dtype=np.float32)
        region_info = []
        
        # Configuration
        sigma_ranges = sigma_config.get('sigma_ranges', [(1, 2), (2, 4), (4, 6), (6, 8)])
        num_regions_range = sigma_config.get('num_regions', (2, 5))
        region_size_range = sigma_config.get('region_size_range', (50, 200))
        overlap_allowed = sigma_config.get('overlap_allowed', True)
        
        num_regions = np.random.randint(num_regions_range[0], num_regions_range[1] + 1)
        region_centers = []
        
        for i in range(num_regions):
            # Choose sigma level for this region
            sigma_range_idx = np.random.randint(0, len(sigma_ranges))
            sigma_range = sigma_ranges[sigma_range_idx]
            sigma = np.random.uniform(sigma_range[0], sigma_range[1])
            
            # Region parameters
            region_size = np.random.randint(region_size_range[0], region_size_range[1])
            
            # Find valid position
            max_attempts = 50
            valid_position = False
            
            for attempt in range(max_attempts):
                center_y = np.random.randint(region_size//2, height - region_size//2)
                center_x = np.random.randint(region_size//2, width - region_size//2)
                
                # Check overlap if not allowed
                if not overlap_allowed:
                    valid_position = True
                    for prev_center, prev_size in region_centers:
                        dist = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                        if dist < (region_size + prev_size) * 0.3:
                            valid_position = False
                            break
                else:
                    valid_position = True
                
                if valid_position:
                    break
            
            if not valid_position:
                continue
            
            region_centers.append(((center_x, center_y), region_size))
            
            # Create region mask with smooth transitions
            transition_width = region_size * 0.2
            y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
            dist = np.sqrt(x*x + y*y)
            region_mask = 1 / (1 + np.exp((dist - region_size/2) / transition_width))
            
            # Update sigma map (use maximum for overlapping regions)
            current_sigma_region = region_mask * sigma
            sigma_map = np.maximum(sigma_map, current_sigma_region)
            
            region_info.append({
                'center': (center_x, center_y),
                'size': region_size,
                'sigma': sigma,
                'sigma_range_idx': sigma_range_idx
            })
        
        return sigma_map, region_info
    
    def apply_multi_sigma_blur(self, image, sigma_map, max_sigma=8):
        """
        Apply blur with spatially varying sigma values
        More efficient than your original implementation
        
        @args:
            - image (np.ndarray): Input image
            - sigma_map (np.ndarray): Map of sigma values for each pixel
            - max_sigma (float): Maximum sigma value for efficiency
            
        @returns:
            - blurred_image (np.ndarray): Spatially varying blurred image
        """
        if image.ndim == 2:
            return self._apply_blur_2d(image, sigma_map, max_sigma)
        elif image.ndim == 3:
            # For time series data
            result = np.zeros_like(image)
            for t in range(image.shape[0]):
                result[t] = self._apply_blur_2d(image[t], sigma_map, max_sigma)
            return result
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    def _apply_blur_2d(self, image, sigma_map, max_sigma):
        """Apply 2D blur with spatially varying sigma"""
        result = image.copy().astype(np.float32)
        
        # Quantize sigma map for efficiency
        sigma_levels = np.unique(np.round(sigma_map * 10) / 10)
        sigma_levels = sigma_levels[sigma_levels > 0.1]  # Skip very small sigmas
        
        for sigma in sigma_levels:
            if sigma > max_sigma:
                sigma = max_sigma
            
            # Create mask for this sigma level
            tolerance = 0.05
            level_mask = np.abs(sigma_map - sigma) < tolerance
            
            if not np.any(level_mask):
                continue
            
            # Calculate appropriate kernel size
            kernel_size = max(3, int(2 * np.ceil(2 * sigma) + 1))
            kernel_size = min(kernel_size, 21)  # Reasonable upper limit
            
            # Apply Gaussian blur
            if kernel_size >= 3:
                blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
                result[level_mask] = blurred[level_mask]
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    def create_curriculum_blur_sample(self, clean_image, difficulty_level='random', curriculum_stage=0):
        """
        Create blur samples following a curriculum learning approach
        
        @args:
            - clean_image (np.ndarray): Clean input image
            - difficulty_level (str or int): 'easy', 'medium', 'hard', 'extreme', or 0-3
            - curriculum_stage (int): Current stage in curriculum (0=start, higher=more advanced)
            
        @returns:
            - blurred_image (np.ndarray): Blurred version
            - blur_info (dict): Information about applied blur
        """
        
        # Define curriculum stages
        curriculum_configs = {
            0: {  # Easy - single small blur regions
                'sigma_ranges': [(0.5, 1.5), (1.0, 2.0)],
                'num_regions': (1, 2),
                'region_size_range': (80, 150),
                'overlap_allowed': False,
                'probabilities': [0.7, 0.3]
            },
            1: {  # Medium - multiple moderate blur regions
                'sigma_ranges': [(0.5, 1.5), (1.5, 3.0), (2.5, 4.0)],
                'num_regions': (2, 3),
                'region_size_range': (60, 180),
                'overlap_allowed': True,
                'probabilities': [0.4, 0.4, 0.2]
            },
            2: {  # Hard - multiple strong blur regions
                'sigma_ranges': [(1.0, 2.5), (2.5, 4.5), (4.0, 6.0)],
                'num_regions': (2, 4),
                'region_size_range': (50, 200),
                'overlap_allowed': True,
                'probabilities': [0.3, 0.4, 0.3]
            },
            3: {  # Extreme - very strong blur, many regions
                'sigma_ranges': [(2.0, 4.0), (4.0, 6.0), (6.0, 8.0)],
                'num_regions': (3, 5),
                'region_size_range': (40, 220),
                'overlap_allowed': True,
                'probabilities': [0.2, 0.4, 0.4]
            }
        }
        
        # Select configuration based on difficulty or curriculum stage
        if difficulty_level == 'random':
            # Bias towards easier levels early in training
            if curriculum_stage <= 0.25:  # First quarter of training
                config_probs = [0.6, 0.3, 0.1, 0.0]
            elif curriculum_stage <= 0.5:  # Second quarter
                config_probs = [0.3, 0.4, 0.2, 0.1]
            elif curriculum_stage <= 0.75:  # Third quarter
                config_probs = [0.2, 0.3, 0.4, 0.1]
            else:  # Final quarter
                config_probs = [0.1, 0.2, 0.3, 0.4]
            
            difficulty_idx = np.random.choice(4, p=config_probs)
        elif isinstance(difficulty_level, str):
            difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2, 'extreme': 3}
            difficulty_idx = difficulty_map.get(difficulty_level, 1)
        else:
            difficulty_idx = max(0, min(3, int(difficulty_level)))
        
        config = curriculum_configs[difficulty_idx]
        
        # Create sigma map
        sigma_map, region_info = self.create_multi_sigma_mask(
            clean_image.shape[-2:] if clean_image.ndim == 3 else clean_image.shape,
            config
        )
        
        # Apply blur
        blurred_image = self.apply_multi_sigma_blur(clean_image, sigma_map)
        
        blur_info = {
            'difficulty_level': difficulty_idx,
            'difficulty_name': ['easy', 'medium', 'hard', 'extreme'][difficulty_idx],
            'curriculum_stage': curriculum_stage,
            'num_regions': len(region_info),
            'sigma_ranges_used': [info['sigma'] for info in region_info],
            'avg_sigma': np.mean([info['sigma'] for info in region_info]) if region_info else 0,
            'max_sigma': np.max([info['sigma'] for info in region_info]) if region_info else 0,
            'region_info': region_info
        }
        
        return blurred_image, blur_info

def simulate_gaussian_blur(image, sigma_range=(1, 8), return_sigma=False):
    """
    Apply Gaussian blur with random sigma from the given range
    Backward compatibility function for existing code
    """
    simulator = AdvancedDefocusSimulator()
    
    if isinstance(sigma_range, (int, float)):
        sigma = sigma_range
    else:
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    
    # Create uniform sigma map
    if image.ndim == 2:
        shape = image.shape
    elif image.ndim == 3:
        shape = image.shape[-2:]
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    sigma_map = np.full(shape, sigma, dtype=np.float32)
    blurred = simulator.apply_multi_sigma_blur(image, sigma_map)
    
    if return_sigma:
        return blurred, sigma
    return blurred

def simulate_variable_blur(image, sigma_map=None, sigma_range=(1, 8)):
    """
    Apply spatially varying blur to simulate real microscopy conditions
    Backward compatibility function for existing code
    """
    simulator = AdvancedDefocusSimulator()
    
    if sigma_map is None:
        # Create default configuration
        config = {
            'sigma_ranges': [sigma_range],
            'num_regions': (2, 4),
            'region_size_range': (50, 150),
            'overlap_allowed': True
        }
        
        if image.ndim == 2:
            shape = image.shape
        elif image.ndim == 3:
            shape = image.shape[-2:]
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")
        
        sigma_map, _ = simulator.create_multi_sigma_mask(shape, config)
    
    blurred = simulator.apply_multi_sigma_blur(image, sigma_map)
    return blurred, sigma_map

# -------------------------------------------------------------------------------------------------

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

    # Blur simulation arguments
    parser.add_argument("--enable_blur_simulation", action="store_true", help='enable blur simulation during training for better generalization')
    parser.add_argument("--blur_difficulty", type=str, default=None, choices=['easy', 'medium', 'hard', 'extreme'], help='fixed difficulty level for blur simulation (None for curriculum learning)')
    parser.add_argument("--blur_sigma_ranges", nargs='+', type=str, default=['0.5,1.5', '1.5,3.0', '3.0,5.0', '5.0,8.0'], help='sigma ranges for different difficulty levels (format: "min,max")')
    parser.add_argument("--blur_probabilities", nargs='+', type=float, default=[0.4, 0.3, 0.2, 0.1], help='probabilities for each difficulty level')
    parser.add_argument("--variable_blur_prob", type=float, default=0.2, help='probability of using spatially variable blur')
    parser.add_argument("--no_blur_prob", type=float, default=0.1, help='probability of using clean image as input (no blur)')

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
