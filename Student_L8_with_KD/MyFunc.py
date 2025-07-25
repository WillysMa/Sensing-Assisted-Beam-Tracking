#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: MyFunc.py
@time: 2025/5/26 16:09
"""
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transf
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import _LRScheduler
from collections.abc import Iterable
import torch.nn as nn
import torch.nn.functional as F
from math import log, cos, pi, floor
import random

def set_seed(seed=42):
    """Set all random seeds for reproducible training"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_best_gpu():
    if not torch.cuda.is_available():
        return ""
    
    best_gpu = 0
    min_memory_used = float('inf')
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        memory_used = torch.cuda.memory_allocated(i)
        if memory_used < min_memory_used:
            min_memory_used = memory_used
            best_gpu = i
    
    return str(best_gpu)


def calculate_weight_update_interval(epoch, initial_interval=20, min_interval=3, beta=0.8, start_epoch=10):
    """
    Calculate the interval for weight updates based on current epoch.
    Starts weight updates at specified epoch, then decreases interval by beta factor after each update cycle.
    
    Args:
        epoch: Current epoch number
        initial_interval: Initial interval between weight updates (default: 20)
        min_interval: Minimum interval between weight updates (default: 3)
        beta: Time decay coefficient applied to reduce interval (default: 0.8)
        start_epoch: Epoch to start weight updates (default: 20)
    
    Returns:
        current_interval: Current interval for weight updates
                         Returns -1 if before start_epoch (indicating no updates)
    """
    # No weight updates before start_epoch
    if epoch < start_epoch:
        return 10000
    
    # Calculate which interval generation we're in
    epochs_since_start = epoch - start_epoch
    current_interval = initial_interval
    accumulated_epochs = 0
    
    # Find the current interval by simulating the decay process
    while accumulated_epochs + current_interval <= epochs_since_start:
        accumulated_epochs += current_interval
        # Apply decay coefficient to get next interval
        current_interval = int(current_interval * beta)
        # Ensure minimum interval
        current_interval = max(current_interval, min_interval)
    
    return current_interval

class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999, warmup_steps=1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Store EMA parameters
        self.ema_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA parameters"""
        self.step_count += 1
        
        # Calculate dynamic decay with warmup
        if self.step_count <= self.warmup_steps:
            decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        else:
            decay = self.decay
        
        # Update EMA parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_params:
                self.ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_to_model(self, model):
        """Apply EMA parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_params:
                param.data.copy_(self.ema_params[name])
    
    def store_model_params(self, model):
        """Store current model parameters"""
        stored_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                stored_params[name] = param.data.clone()
        return stored_params
    
    def restore_model_params(self, model, stored_params):
        """Restore model parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in stored_params:
                param.data.copy_(stored_params[name])
    
    def state_dict(self):
        """Return EMA state dict"""
        return {
            'ema_params': self.ema_params,
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.ema_params = state_dict['ema_params']
        self.decay = state_dict['decay']
        self.warmup_steps = state_dict['warmup_steps']
        self.step_count = state_dict['step_count']

    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_topk_accuracy(outputs, labels, k_values=[1, 2, 3, 5, 10]):
    """Calculate top-k accuracy for given k values"""
    num_pred = labels.shape[1]
    topk_correct = {k: np.zeros((num_pred,)) for k in k_values}
    total = torch.sum(labels != -100, dim=0).cpu().numpy()
    
    _, idx = torch.topk(outputs, max(k_values), dim=-1)
    idx = idx.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for i in range(labels.shape[1]):  # for each time step
        for j in range(labels.shape[0]):  # examine all samples
            for k in k_values:
                topk_correct[k][i] += np.isin(labels[j, i], idx[j, i, :k])
    
    # Calculate accuracy
    topk_acc = {}
    for k in k_values:
        topk_acc[k] = topk_correct[k] / (total + 1e-8)  # Add small epsilon to avoid division by zero
    
    return topk_acc, total

def calculate_dba_score(outputs, labels, delta=5):
    """Calculate DBA (Distance-Based Accuracy) score"""
    num_pred = labels.shape[1]
    dba_score = np.zeros((num_pred,))
    valid_count = np.zeros((num_pred,))
    
    _, idx = torch.topk(outputs, 3, dim=-1)  # top-3 predictions for DBA
    idx = idx.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for t in range(labels.shape[1]):
        for b in range(labels.shape[0]):
            gt = labels[b, t]
            if gt == -100:
                continue  # skip invalid label
            
            preds = idx[b, t, :3]  # top-3 predictions
            norm_dists = np.minimum(np.abs(preds - gt) / delta, 1.0)
            min_norm_dist = np.min(norm_dists)
            
            dba_score[t] += min_norm_dist
            valid_count[t] += 1
    
    # Avoid division by zero
    valid_count[valid_count == 0] = 1
    dba_score = 1 - (dba_score / valid_count)
    
    return dba_score

def save_checkpoint(state, save_path, filename='checkpoint.pth'):
    """Save training checkpoint"""
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(save_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint_path = os.path.join(save_path, 'Final_model.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return start_epoch, checkpoint.get('test_loss', 0.0)
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, 0.0
    

    

def plot_training_curves(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, save_path, 
            train_task_loss_hist=None, train_distill_loss_hist=None, div_history=None):
    """Plot and save training curves including knowledge distillation losses"""
    epochs = len(train_acc_hist)
    
    # Learning rate schedule
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.grid(True)
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(save_path, 'LR_schedule.png'))
    plt.close()
    
    # Accuracy curves
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_acc_hist, '-o', label='Train')
    plt.plot(np.arange(1, epochs + 1), test_acc_hist, '-o', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Test Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Accuracy_curves.png'))
    plt.close()
    
    # Loss curves
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_loss_hist, '-o', label='Train Total')
    plt.plot(np.arange(1, epochs + 1), test_loss_hist, '-o', label='Test')
    
    # Add knowledge distillation loss components if available
    if train_task_loss_hist is not None:
        plt.plot(np.arange(1, epochs + 1), train_task_loss_hist, '--', label='Train Task Loss')
    if train_distill_loss_hist is not None:
        plt.plot(np.arange(1, epochs + 1), train_distill_loss_hist, ':', label='Train Distillation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Components (Knowledge Distillation)')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Loss_curves_KD.png'))
    plt.close()

    # KL divergence curves for each time slot
    if div_history is not None:
        div_history = np.array(div_history)
        
        # Check if div_history has meaningful data (not empty and not all zeros)
        if div_history.size > 0 and not np.all(div_history == 0):
            plt.figure(figsize=(12, 8))
            
            # If div_history is 2D (epochs x time_slots), plot each time slot
            if div_history.ndim == 2:
                num_time_slots = div_history.shape[1]
                epochs_range = np.arange(1, epochs + 1)
                
                for slot in range(num_time_slots):
                    plt.plot(epochs_range, div_history[:, slot], '-o', 
                            label=f'Time Slot {slot}', markersize=3)
            else:
                # If 1D, assume it's for a single time slot or average
                plt.plot(np.arange(1, epochs + 1), div_history, '-o', 
                        label='KL Divergence', markersize=3)
            
            plt.xlabel('Epoch')
            plt.ylabel('KL Divergence')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title('KL Divergence per Time Slot vs Epochs')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'KL_divergence_vs_epochs.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # Also create a heatmap showing KL divergence evolution
            if div_history.ndim == 2:
                plt.figure(figsize=(12, 6))
                im = plt.imshow(div_history.T, aspect='auto', cmap='viridis', 
                               interpolation='nearest')
                plt.colorbar(im, label='KL Divergence')
                plt.xlabel('Epoch')
                plt.ylabel('Time Slot')
                plt.title('KL Divergence Heatmap: Time Slots vs Epochs')
                plt.xticks(np.arange(0, epochs, max(1, epochs//10)), 
                          np.arange(1, epochs+1, max(1, epochs//10)))
                plt.yticks(np.arange(num_time_slots), 
                          [f'Slot {i}' for i in range(num_time_slots)])
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'KL_divergence_heatmap.png'), 
                           bbox_inches='tight', dpi=300)
                plt.close()
        else:
            print("Skipping KL divergence plots: No meaningful KL divergence data available (KD mode may not be 1 or teacher model not used)")

def create_samples(root, portion=1.):
    f = pd.read_csv(root, na_values='')
    f = f.fillna(-99)
    Total_Num = len(f)
    num_data = int(Total_Num * portion)
    data_samples_rgb = []
    # data_samples_radar = []
    pred_beam = []
    inp_beam = []
    for idx, row in f.head(num_data).iterrows():

        # Dynamic approach: get all image columns
        vision_cols = [col for col in f.columns if col.startswith('camera')]
        vision_cols.sort()  # Ensure consistent ordering (future_beam1, future_beam2, etc.)
        vision_data = row[vision_cols].tolist() # get the future_beam columns

        data_samples_rgb.append(vision_data)
        # radar_data = row['radar1':'radar8'].tolist()
        # data_samples_radar.append(radar_data)

        # Dynamic approach: get all future_beam columns
        future_beam_cols = [col for col in f.columns if col.startswith('future_beam')]
        future_beam_cols.sort()  # Ensure consistent ordering (future_beam1, future_beam2, etc.)
        future_beam = row[future_beam_cols].tolist()
        pred_beam.append(future_beam)
        # future_beam_id = np.asarray([np.argmax(np.loadtxt(pwr)) for pwr in future_beam])
        # pred_beam.append(future_beam_id)

        input_beam_cols = [col for col in f.columns if col.startswith('beam')]
        input_beam_cols.sort()  # Ensure consistent ordering (future_beam1, future_beam2, etc.)
        input_beam = row[input_beam_cols].tolist()
        # input_beam_id = np.asarray([np.argmax(np.loadtxt(pwr)) for pwr in input_beam]) # start with 0
        inp_beam.append(input_beam)

    print('list is ready')
    return data_samples_rgb, inp_beam, pred_beam


class DataFeed(Dataset):
    def __init__(self, data_root, root_csv, seq_len, transform=None,  portion=1.):

        self.data_root = data_root
        self.samples_rgb, self.inp_val, self.pred_val = create_samples(root_csv, portion=portion)
        self.seq_len = seq_len
        self.transform = transform


    def __len__(self):
        return len(self.samples_rgb)

    def __getitem__(self, idx):
        samples_rgb = self.samples_rgb[idx]
        # samples_radar = self.samples_radar[idx]
        beam_val = self.pred_val[idx]
        input_beam = self.inp_val[idx]

        sample_rgb = samples_rgb[-self.seq_len:]
        # sample_radar = samples_radar[-self.seq_len:]
        input_beam1 = input_beam[-self.seq_len:]

        # out_beam = torch.zeros((3,))
        image_val = np.zeros((self.seq_len, 224,224))
        image_dif = np.zeros((self.seq_len-1, 224, 224))
        image_motion_masks = np.zeros((self.seq_len - 1, 224, 224))

        beam_past = []
        for i in range(self.seq_len):
            beam_past_i = int(np.argmax(np.loadtxt(self.data_root + input_beam1[i][1:]))) 
            beam_past.append(beam_past_i)
            # Load the image
            img = self.transform(io.imread(self.data_root + sample_rgb[i][1:]))

            img = rgb2gray(img)  # Convert to grayscale

            # Apply Gaussian filtering
            img_smoothed = gaussian_filter(img, sigma=1)  # Adjust sigma for smoothing strength

            # Store the smoothed image
            image_val[i, ...] = img_smoothed

            # Compute the difference with the previous frame
            if i >= 1:
                diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
                image_dif[i - 1, ...] = diff

                # Calculate the dynamic threshold as 10% of the maximum pixel value in the difference image
                max_pixel_value = np.max(diff)
                threshold_value = 0.1 * max_pixel_value

                # Generate binary mask of significant changes
                motion_mask = (diff > threshold_value).astype(np.uint8)
                image_motion_masks[i - 1, ...] = motion_mask

        image_masks = torch.tensor(image_motion_masks,dtype=torch.float32)
        # radar_masks = torch.tensor(radar_motion_masks,dtype=torch.float32)

  

        beam_future = []
        for i in range(len(beam_val)):
            beam_future_i = int(np.argmax(np.loadtxt(self.data_root + beam_val[i][1:]))) 
            beam_future.append(beam_future_i)


        input_beam = torch.tensor(beam_past, dtype=torch.int64)
        out_beam = torch.tensor(beam_future, dtype=torch.int64)

        pass
        return image_masks, input_beam.long(), torch.squeeze(out_beam.long())



class CyclicCosineDecayLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 init_decay_epochs,
                 min_decay_lr,
                 restart_interval=None,
                 restart_interval_multiplier=None,
                 restart_lr=None,
                 warmup_epochs=None,
                 warmup_start_lr=None,
                 last_epoch=-1,
                 verbose=False):
        """
        Initialize new CyclicCosineDecayLR object.
        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_decay_epochs: (int) - Number of initial decay epochs.
        :param min_decay_lr: (float or iterable of floats) - Learning rate at the end of decay.
        :param restart_interval: (int) - Restart interval for fixed cycles.
            Set to None to disable cycles. Default: None.
        :param restart_interval_multiplier: (float) - Multiplication coefficient for geometrically increasing cycles.
            Default: None.
        :param restart_lr: (float or iterable of floats) - Learning rate when cycle restarts.
            If None, optimizer's learning rate will be used. Default: None.
        :param warmup_epochs: (int) - Number of warmup epochs. Set to None to disable warmup. Default: None.
        :param warmup_start_lr: (float or iterable of floats) - Learning rate at the beginning of warmup.
            Must be set if warmup_epochs is not None. Default: None.
        :param last_epoch: (int) - The index of the last epoch. This parameter is used when resuming a training job. Default: -1.
        :param verbose: (bool) - If True, prints a message to stdout for each update. Default: False.
        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError("init_decay_epochs must be positive integer, got {} instead".format(init_decay_epochs))

        if isinstance(min_decay_lr, Iterable) and len(min_decay_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(min_decay_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_decay_lr), len(optimizer.param_groups)))

        if restart_interval is not None and (not isinstance(restart_interval, int) or restart_interval < 1):
            raise ValueError("restart_interval must be positive integer, got {} instead".format(restart_interval))

        if restart_interval_multiplier is not None and \
                (not isinstance(restart_interval_multiplier, float) or restart_interval_multiplier <= 0):
            raise ValueError("restart_interval_multiplier must be positive float, got {} instead".format(
                restart_interval_multiplier))

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups)))

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(type(warmup_epochs)))

            if warmup_start_lr is None:
                raise ValueError("warmup_start_lr must be set when warmup_epochs is not None")

            if not (isinstance(warmup_start_lr, float) or isinstance(warmup_start_lr, Iterable)):
                raise ValueError("warmup_start_lr must be either float or iterable of floats, got {} instead".format(
                    warmup_start_lr))

            if isinstance(warmup_start_lr, Iterable) and len(warmup_start_lr) != len(optimizer.param_groups):
                raise ValueError("Expected len(warmup_start_lr) to be equal to len(optimizer.param_groups), "
                                 "got {} and {} instead".format(len(warmup_start_lr), len(optimizer.param_groups)))

        group_num = len(optimizer.param_groups)
        self._warmup_start_lr = [warmup_start_lr] * group_num if isinstance(warmup_start_lr, float) else warmup_start_lr
        self._warmup_epochs = 0 if warmup_epochs is None else warmup_epochs
        self._init_decay_epochs = init_decay_epochs
        self._min_decay_lr = [min_decay_lr] * group_num if isinstance(min_decay_lr, float) else min_decay_lr
        self._restart_lr = [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        self._restart_interval = restart_interval
        self._restart_interval_multiplier = restart_interval_multiplier
        try:
                # PyTorch 1.4+ (with verbose parameter)
                super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch, verbose=verbose)
        except TypeError:
            try:
                # PyTorch 1.1-1.3 (without verbose parameter)
                super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch)
            except TypeError:
                # Very old PyTorch versions
                super(CyclicCosineDecayLR, self).__init__(optimizer)

    def get_lr(self):

        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return self._calc(self.last_epoch,
                              self._warmup_epochs,
                              self._warmup_start_lr,
                              self.base_lrs)

        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(self.last_epoch - self._warmup_epochs,
                              self._init_decay_epochs,
                              self.base_lrs,
                              self._min_decay_lr)
        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (self.last_epoch - self._init_decay_epochs - self._warmup_epochs) % self._restart_interval
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      self._restart_interval,
                                      lrs,
                                      self._min_decay_lr)
                else:
                    n = self._get_n(self.last_epoch - self._warmup_epochs - self._init_decay_epochs)
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = self.last_epoch - sn_prev - self._warmup_epochs - self._init_decay_epochs
                    interval = self._restart_interval * self._restart_interval_multiplier ** n
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      interval,
                                      lrs,
                                      self._min_decay_lr)
            else:
                return self._min_decay_lr

    def _calc(self, t, T, lrs, min_lrs):
        return [min_lr + (lr - min_lr) * ((1 + cos(pi * t / T)) / 2)
                for lr, min_lr in zip(lrs, min_lrs)]

    def _get_n(self, epoch):
        _t = 1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval
        return floor(log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n):
        return self._restart_interval * (1 - self._restart_interval_multiplier ** n) / (
                    1 - self._restart_interval_multiplier)


class SmoothCosineDecayLR(_LRScheduler):
    """
    Smooth Cosine Decay Learning Rate Scheduler with Warmup and Gradually Reducing Peak LR
    
    This scheduler provides:
    1. Warmup phase: Linear increase from warmup_start_lr to base_lr
    2. Smooth cosine decay with gentle restarts
    3. Each restart cycle has a gradually reduced peak learning rate
    4. No sharp jumps between cycles
    """
    def __init__(self,
                 optimizer,
                 warmup_epochs=10,
                 warmup_start_lr=1e-5,
                 first_cycle_epochs=50,
                 min_lr=1e-5,
                 cycle_decay_factor=0.8,
                 cycle_length_multiplier=1.2,
                 last_epoch=-1,
                 verbose=False):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Learning rate at the beginning of warmup
            first_cycle_epochs: Length of the first cosine cycle after warmup
            min_lr: Minimum learning rate (bottom of each cycle)
            cycle_decay_factor: Factor by which peak LR is reduced each cycle (e.g., 0.8 means 20% reduction)
            cycle_length_multiplier: Factor by which cycle length increases each cycle (e.g., 1.2 means 20% longer)
            last_epoch: The index of the last epoch
            verbose: If True, prints a message to stdout for each update
        """
        
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative integer")
        
        if not isinstance(first_cycle_epochs, int) or first_cycle_epochs < 1:
            raise ValueError("first_cycle_epochs must be positive integer")
        
        if cycle_decay_factor <= 0 or cycle_decay_factor >= 1:
            raise ValueError("cycle_decay_factor must be between 0 and 1")
        
        if cycle_length_multiplier <= 0:
            raise ValueError("cycle_length_multiplier must be positive")
        
        # Store parameters
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.first_cycle_epochs = first_cycle_epochs
        self.min_lr = min_lr
        self.cycle_decay_factor = cycle_decay_factor
        self.cycle_length_multiplier = cycle_length_multiplier
        
        # Store initial learning rate for calculations (before parent init)
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        # Initialize with base learning rates from optimizer
        try:
                # PyTorch 1.4+ (with verbose parameter)
                super(SmoothCosineDecayLR, self).__init__(optimizer, last_epoch, verbose)
        except TypeError:
            try:
                # PyTorch 1.1-1.3 (without verbose parameter)
                super(SmoothCosineDecayLR, self).__init__(optimizer, last_epoch)
            except TypeError:
                # Very old PyTorch versions
                super(SmoothCosineDecayLR, self).__init__(optimizer)
    
    def get_lr(self):
        """Calculate learning rate for current epoch"""
        
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 0:
                return self.base_lrs
            
            # Linear warmup
            warmup_progress = self.last_epoch / self.warmup_epochs
            current_lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * warmup_progress
            return [current_lr] * len(self.base_lrs)
        
        # Cosine decay with smooth restarts
        epoch_after_warmup = self.last_epoch - self.warmup_epochs
        
        # Determine which cycle we're in and position within that cycle
        cycle_num = 0
        cycle_start_epoch = 0
        current_cycle_length = self.first_cycle_epochs
        
        while cycle_start_epoch + current_cycle_length <= epoch_after_warmup:
            cycle_start_epoch += current_cycle_length
            cycle_num += 1
            current_cycle_length = int(current_cycle_length * self.cycle_length_multiplier)
        
        # Position within current cycle (0 to 1)
        cycle_progress = (epoch_after_warmup - cycle_start_epoch) / current_cycle_length
        
        # Calculate peak LR for current cycle (gradually decreasing)
        current_peak_lr = self.initial_lr * (self.cycle_decay_factor ** cycle_num)
        
        # Smooth transition from previous cycle's end
        if cycle_num > 0:
            # Previous cycle's peak LR
            prev_peak_lr = self.initial_lr * (self.cycle_decay_factor ** (cycle_num - 1))
            # Previous cycle ended at min_lr, so we smoothly increase from there
            cycle_start_lr = self.min_lr + (current_peak_lr - self.min_lr) * 0.1  # Gentle start
        else:
            cycle_start_lr = current_peak_lr
        
        # Cosine decay within cycle with smooth start
        if cycle_progress <= 0.1:  # First 10% of cycle: smooth increase to peak
            smoothing_factor = cycle_progress / 0.1
            current_lr = cycle_start_lr + (current_peak_lr - cycle_start_lr) * smoothing_factor
        else:  # Remaining 90%: cosine decay from peak to min
            adjusted_progress = (cycle_progress - 0.1) / 0.9
            cosine_factor = 0.5 * (1 + cos(pi * adjusted_progress))
            current_lr = self.min_lr + (current_peak_lr - self.min_lr) * cosine_factor
        
        return [max(current_lr, self.min_lr)] * len(self.base_lrs)
    
    def get_cycle_info(self):
        """Get information about current cycle (useful for debugging/logging)"""
        if self.last_epoch < self.warmup_epochs:
            return {"phase": "warmup", "progress": self.last_epoch / self.warmup_epochs}
        
        epoch_after_warmup = self.last_epoch - self.warmup_epochs
        cycle_num = 0
        cycle_start_epoch = 0
        current_cycle_length = self.first_cycle_epochs
        
        while cycle_start_epoch + current_cycle_length <= epoch_after_warmup:
            cycle_start_epoch += current_cycle_length
            cycle_num += 1
            current_cycle_length = int(current_cycle_length * self.cycle_length_multiplier)
        
        cycle_progress = (epoch_after_warmup - cycle_start_epoch) / current_cycle_length
        current_peak_lr = self.initial_lr * (self.cycle_decay_factor ** cycle_num)
        
        return {
            "phase": "cosine_decay",
            "cycle_num": cycle_num,
            "cycle_progress": cycle_progress,
            "current_peak_lr": current_peak_lr,
                         "cycle_length": current_cycle_length
         }


class StaircaseDecayLR(_LRScheduler):
    """
    Staircase (Step) Decay Learning Rate Scheduler with Warmup
    
    This scheduler provides:
    1. Warmup phase: Linear increase from warmup_start_lr to base_lr
    2. Staircase decay: LR is multiplied by decay_factor at specified epochs
    3. Constant LR between decay steps
    """
    def __init__(self,
                 optimizer,
                 warmup_epochs=10,
                 warmup_start_lr=1e-5,
                 min_lr=1e-5,
                 decay_epochs=None,
                 decay_factor=0.5,
                 last_epoch=-1,
                 verbose=False):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Learning rate at the beginning of warmup
            decay_epochs: List of epochs at which to decay LR (e.g., [30, 60, 90])
            decay_factor: Factor by which to multiply LR at decay epochs (e.g., 0.5 for 50% reduction)
            last_epoch: The index of the last epoch
            verbose: If True, prints a message to stdout for each update
        """
        
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative integer")
        
        if decay_epochs is None:
            decay_epochs = [30, 60, 90, 120, 150]  # Default decay schedule
        
        if not isinstance(decay_epochs, (list, tuple)):
            raise ValueError("decay_epochs must be a list or tuple of integers")
        
        if decay_factor <= 0 or decay_factor >= 1:
            raise ValueError("decay_factor must be between 0 and 1")
        
        # Store parameters
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.decay_epochs = sorted(decay_epochs)  # Ensure epochs are sorted
        self.decay_factor = decay_factor
        
        # Store initial learning rate for calculations (before parent init)
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        # Initialize with base learning rates from optimizer
        try:
                # PyTorch 1.4+ (with verbose parameter)
                super(StaircaseDecayLR, self).__init__(optimizer, last_epoch, verbose)
        except TypeError:
            try:
                # PyTorch 1.1-1.3 (without verbose parameter)
                super(StaircaseDecayLR, self).__init__(optimizer, last_epoch)
            except TypeError:
                # Very old PyTorch versions
                super(StaircaseDecayLR, self).__init__(optimizer)
    
    def get_lr(self):
        """Calculate learning rate for current epoch"""
        
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 0:
                return self.base_lrs
            
            # Linear warmup
            warmup_progress = self.last_epoch / self.warmup_epochs
            current_lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * warmup_progress
            return [current_lr] * len(self.base_lrs)
        
        # Staircase decay phase
        # Count how many decay epochs have passed
        num_decays = sum(1 for decay_epoch in self.decay_epochs if self.last_epoch >= decay_epoch)
        
        # Calculate current LR by applying decay factor for each passed decay epoch
        current_lr = self.min_lr + (self.initial_lr - self.min_lr) * (self.decay_factor ** num_decays)
        
        return [current_lr] * len(self.base_lrs)
    
    def get_decay_info(self):
        """Get information about current decay state (useful for debugging/logging)"""
        if self.last_epoch < self.warmup_epochs:
            return {
                "phase": "warmup", 
                "progress": self.last_epoch / self.warmup_epochs,
                "current_lr": self.get_lr()[0]
            }
        
        # Find next decay epoch
        next_decay_epoch = None
        for decay_epoch in self.decay_epochs:
            if self.last_epoch < decay_epoch:
                next_decay_epoch = decay_epoch
                break
        
        num_decays = sum(1 for decay_epoch in self.decay_epochs if self.last_epoch >= decay_epoch)  
        current_lr = self.min_lr + (self.initial_lr - self.min_lr) * (self.decay_factor ** num_decays)
        
        return {
            "phase": "staircase_decay",
            "num_decays": num_decays,
            "current_lr": current_lr,
            "next_decay_epoch": next_decay_epoch,
            "epochs_to_next_decay": next_decay_epoch - self.last_epoch if next_decay_epoch else None
        }
     
if __name__ == "__main__":
    num_classes = 64
    batch_size = 4
    val_batch_size = 64
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    data_root = parent_dir + '/dataset/scenario9'
    train_dir = data_root + '/train_seqs.csv'

    seq_len = 8
    img_resize = transf.Resize((224, 224))
    # img_norm = transf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    proc_pipe = transf.Compose(
        [transf.ToPILImage(),
         img_resize]
    )
    FFT_TUPLE = (64, 64, 16) #FFT_ANGLE, FFT_RANGE, FFT_VELOCITY
    DATASET_PCT = 0.1
    train_loader = DataLoader(DataFeed(data_root, train_dir, seq_len, proc_pipe, portion=DATASET_PCT, fft_tuple=FFT_TUPLE),
                              batch_size=batch_size, shuffle=True)
    data = next(iter(train_loader))
    print('done')

    # Path to the CSV file



    # Apply thresholding to both maps

    # filtered_range_angle_map = threshold_map(range_angle_map, percentile=60)
    # filtered_range_velocity_map = threshold_map(range_velocity_map, percentile=60)
    #
    # plt.figure(figsize=(10, 5))
    # plt.imshow(filtered_range_angle_map.T, aspect='auto', cmap='jet', origin='lower')
    # plt.xlabel("Angle Index")
    # plt.ylabel("Range Index")
    # plt.title("Range-Angle Map")
    # plt.colorbar(label="Power")
    # plt.savefig(target_path + sample_name+'_RA_refined.jpg')  # Save the figure to the target path
    # plt.show()
    #
    # # Plot Range-Velocity Map
    # plt.figure(figsize=(10, 5))
    # plt.imshow(filtered_range_velocity_map, aspect='auto', cmap='jet', origin='lower')
    # plt.xlabel("Velocity Index")
    # plt.ylabel("Range Index")
    # plt.title("Range-Velocity Map")
    # plt.colorbar(label="Power")
    # plt.savefig(target_path + sample_name + '_RV_refined.jpg')  # Save the figure to the target path
    # plt.show()