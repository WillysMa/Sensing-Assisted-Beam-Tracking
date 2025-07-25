import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import random
import matplotlib.pyplot as plt

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

def plot_training_curves(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, save_path):
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
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Components (Knowledge Distillation)')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Loss_curves.png'))
    plt.close()


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

def create_samples(root, portion=1.):
    f = pd.read_csv(root, na_values='')
    f = f.fillna(-99)
    Total_Num = len(f)
    num_data = int(Total_Num * portion)
    data_samples_bbox = []
    # data_samples_radar = []
    pred_beam = []
    inp_beam = []
    for idx, row in f.head(num_data).iterrows():

        # Dynamic approach: get all image columns
        vision_cols = [col for col in f.columns if col.startswith('bbox')]
        vision_cols.sort()  # Ensure consistent ordering (future_beam1, future_beam2, etc.)
        vision_data = row[vision_cols].tolist() # get the future_beam columns

        data_samples_bbox.append(vision_data)
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
    return data_samples_bbox, inp_beam, pred_beam


class DataFeed(Dataset):
    def __init__(self, data_root, root_csv, seq_len,  portion=1.):

        self.data_root = data_root
        self.samples_bbox, self.inp_val, self.pred_val = create_samples(root_csv, portion=portion)
        self.seq_len = seq_len


    def __len__(self):
        return len(self.samples_bbox)

    def __getitem__(self, idx):

        samples_bbox = self.samples_bbox[idx]
        beam_val = self.pred_val[idx]
        inp_val = self.inp_val[idx]

        sample_bbox = samples_bbox[-self.seq_len:]
        # sample_radar = samples_radar[-self.seq_len:]
        input_beam = inp_val[-self.seq_len:]

        bbox_seq = np.zeros((self.seq_len, 4))
        beam_past = []
        for i in range(self.seq_len):
            beam_past_i = int(np.argmax(np.loadtxt(self.data_root + input_beam[i][1:]))) 
            beam_past.append(beam_past_i)
            bbox = np.loadtxt(self.data_root + sample_bbox[i][1:])
            if bbox.ndim == 1:
                bbox_seq[i, :] = bbox[1:]
            else:
                rows_with_zero = bbox[bbox[:, 0] == 0]
                bbox_seq[i, :] = rows_with_zero[0, 1:]  # Take the first row starting with 0, excluding the first element
                # print('rows_with_zero', rows_with_zero)


        bbox_seq = torch.tensor(bbox_seq, dtype=torch.float32)

        beam_future = []
        for i in range(len(beam_val)):
            beam_future_i = int(np.argmax(np.loadtxt(self.data_root + beam_val[i][1:]))) 
            beam_future.append(beam_future_i)

        input_beam = torch.tensor(beam_past, dtype=torch.int64)
        out_beam = torch.tensor(beam_future, dtype=torch.int64)

        ccc =1
        return bbox_seq, input_beam.long(), torch.squeeze(out_beam.long())
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    num_classes = 64
    batch_size = 4
    val_batch_size = 64
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    data_root = parent_dir + '/dataset/scenario9'
    train_dir = data_root + '/train_seqs_6_bbox.csv'

    seq_len = 8
    DATASET_PCT = 0.1
    train_loader = DataLoader(DataFeed(data_root, train_dir, seq_len, portion=DATASET_PCT),
                            batch_size=batch_size, shuffle=False)
    data = next(iter(train_loader))
    print('done')