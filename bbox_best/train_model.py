import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import os
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from tqdm import tqdm
import sys
import datetime
import json
import shutil
from Funcs import *
from model import *
import argparse
import time
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single Modality Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['crossentropy', 'focal'], 
                        help='Loss function type')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--use_early_stopping', action='store_true', default=True, help='Enable early stopping')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum change to qualify as improvement')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    
    # Model parameters
    parser.add_argument('--feature_size', type=int, default=64, help='Feature size')
    parser.add_argument('--gru_hidden_size', type=int, default=64, help='GRU hidden size')
    parser.add_argument('--gru_num_layers', type=int, default=1, help='Number of GRU layers for teacher')
    parser.add_argument('--num_classes', type=int, default=64, help='Number of classes')
    parser.add_argument('--seq_length', type=int, default=8, help='Sequence length for teacher model')
    parser.add_argument('--num_pred', type=int, default=6, help='Number of predictions')


    # Data parameters
    parser.add_argument('--data_root', type=str, default='../dataset/scenario9', help='Data root directory')
    parser.add_argument('--dataset_pct', type=float, default=1.0, help='Dataset percentage to use')
    parser.add_argument('--train_csv_name', type=str, default='train_seqs_6_bbox.csv', help='Train csv name')
    parser.add_argument('--test_csv_name', type=str, default='test_seqs_6_bbox.csv', help='Test csv name')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Training settings
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--save_dir', type=str, default='saved_folder_train', help='Save directory')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode (saves to saved_folder_debug)')

    # lr scheduler parameters
    parser.add_argument('--T_0', type=int, default=10, help='Number of iterations for the first restart')
    parser.add_argument('--T_mult', type=int, default=2, help='A factor increases T_i after a restart')
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate')
    
    
    return parser.parse_args()

def train(net, dataloaders, optimizer, scheduler, criterion, device, args, save_directory):
    """Training function for one epoch"""
    net.train()
    start_time = time.time()
    print('Start training...', flush=True)
    best_test_loss = float('inf')
    epochs_without_improvement = 0
    train_acc_hist = []
    train_loss_hist = []
    val_loss_hist = []
    val_acc_hist = []
    lrs = []
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        running_acc = 1.0
        lrs.append(optimizer.param_groups[0]["lr"])
        with tqdm(dataloaders['train'], unit="batch", file=sys.stdout) as tepoch:
            for i, (bbox, beam, label) in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                d1, d2, d3 = bbox.shape
                bbox_batch = torch.cat([bbox[:, 1-args.seq_length:, ...], torch.zeros(d1, args.num_pred, d3)], dim=1).to(device)
                label_batch = torch.cat([beam[..., -1:], label[:, :args.num_pred]], dim=-1).to(device)

                optimizer.zero_grad()

                # h = net.initHidden(d1).to(device)
                outputs = net(bbox_batch)
                outputs = outputs[:, -(args.num_pred + 1):, :]
                loss = criterion(outputs.reshape(-1, args.num_classes), label_batch.flatten())
                prediction = torch.argmax(outputs, dim=-1)
                acc = (prediction == label_batch).sum().item() / int(
                    torch.sum(label != -100).cpu()
                )
                loss.backward()
                optimizer.step()
                
                # print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_acc = (acc + i * running_acc) / (i + 1)
                log = OrderedDict()
                log["loss"] = running_loss
                log["acc"] = running_acc
                tepoch.set_postfix(log)
        scheduler.step()
         # Validation
        val_loss, topk_acc, dba_score = validate(net, dataloaders['test'], criterion, device, args, epoch, save_directory)
        # Set model back to training mode after validation
        
        # Combined model saving and early stopping logic

        improvement_threshold = best_test_loss - (args.min_delta if args.use_early_stopping else 0)
        epochs_without_improvement = 0
        
        if val_loss < improvement_threshold:
            # Validation loss improved - save model and reset early stopping counter
            best_test_loss = val_loss
            torch.save(net.state_dict(), os.path.join(save_directory, 'model_best.pth'))
            print(f"New best model saved! Validation loss: {best_test_loss:.4f}")
        
            # Reset early stopping counter
            if args.use_early_stopping:
                epochs_without_improvement = 0
        else:
            # No improvement
            if args.use_early_stopping:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs (best: {best_test_loss:.4f})")
                
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                    print(f"Best validation loss: {best_test_loss:.4f}")
                    break

        train_acc_hist.append(running_acc)
        train_loss_hist.append(running_loss)             
        val_loss_hist.append(val_loss)
        val_acc_hist.append(topk_acc[1].mean())

    time_elapsed = time.time() - start_time
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, lrs


def validate(net, val_loader, criterion, device, args, epoch, save_directory):
    """Validation function"""
    net.eval()
    val_loss = 0
    all_outputs = []
    all_labels = []
    
    for (bbox, beam, label) in val_loader:
        d1, d2, d3 = bbox.shape
        bbox_batch = torch.cat([bbox[:, 1-args.seq_length:, ...], torch.zeros(d1, args.num_pred, d3)], dim=1).to(device)
        label_batch = torch.cat([beam[..., -1:], label[:, :args.num_pred]], dim=-1).to(device)

        with torch.no_grad():
            # h = net.initHidden(bbox_batch.shape[1]).to(device)
            outputs = net(bbox_batch)
        outputs = outputs[:, -(args.num_pred + 1):, :]
        val_loss += criterion(outputs.reshape(-1, args.num_classes), label_batch.flatten()).item()

        all_outputs.append(outputs)
        all_labels.append(label_batch)

    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    topk_acc, total = calculate_topk_accuracy(all_outputs, all_labels)
    dba_score = calculate_dba_score(all_outputs, all_labels)
    
    val_loss /= len(val_loader)
    
    param_info = f" (seq_len={args.seq_length}, num_pred={args.num_pred})"
    
    print(f'Epoch {epoch} Test Loss{param_info}: {val_loss:.4f}', flush=True)
    print("DBA-Score (Top-3):", dba_score)
    print('Top-K Accuracy:', flush=True)
    for k, acc in topk_acc.items():
        print(f'Top-{k}: {acc}', flush=True)
    
    # Save results
    with open(os.path.join(save_directory, 'test_results.txt'), "a") as f:
        f.write(f"Epoch {epoch} Results Summary{param_info}\n\n")
        f.write(f"Test Loss: {val_loss:.4f}\n\n")
        
        # Write DBA-Score
        dba_str = ", ".join([f"{x:.4f}" for x in dba_score])
        f.write(f"DBA-Score (Top-3): [{dba_str}]\n\n")
        
        # Write Top-K Accuracy
        f.write("Top-K Accuracy Per Time Slot:\n")
        for k, acc in topk_acc.items():
            acc_str = ", ".join([f"{a:.4f}" for a in acc])
            f.write(f"Top-{k} Accuracy: [{acc_str}]\n")
        f.write("=" * 50 + "\n\n")
    
    return val_loss, topk_acc, dba_score




def test(net, val_loader, criterion, device, args, save_directory):
    """Test function"""
    net.eval()
    val_loss = 0
    all_outputs = []
    all_labels = []
    
    
    for (bbox, beam, label) in val_loader:
        d1, d2, d3 = bbox.shape
        bbox_batch = torch.cat([bbox[:, 1-args.seq_length:, ...], torch.zeros(d1, args.num_pred, d3)], dim=1).to(device)
        label_batch = torch.cat([beam[..., -1:], label[:, :args.num_pred]], dim=-1).to(device)

        with torch.no_grad():
            # h = net.initHidden(bbox_batch.shape[1]).to(device)
            outputs = net(bbox_batch)
        outputs = outputs[:, -(args.num_pred + 1):, :]
        val_loss += criterion( outputs.reshape(-1, args.num_classes), label_batch.flatten() ).item()

        all_outputs.append(outputs)
        all_labels.append(label_batch)

    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    topk_acc, total = calculate_topk_accuracy(all_outputs, all_labels)
    dba_score = calculate_dba_score(all_outputs, all_labels)
    
    val_loss /= len(val_loader)
    
    print(f'Test Loss: {val_loss:.4f}', flush=True)
    print("DBA-Score (Top-3):", dba_score)
    print('Top-K Accuracy:', flush=True)
    for k, acc in topk_acc.items():
        print(f'Top-{k}: {acc}', flush=True)
    

     # Save results
    with open(os.path.join(save_directory, 'test_results.txt'), "a") as f:
        f.write("Test Results Summary\n\n")
        f.write(f"Test Loss: {val_loss:.4f}\n\n")
        
        # Write DBA-Score
        dba_str = ", ".join([f"{x:.4f}" for x in dba_score])
        f.write(f"DBA-Score (Top-3): [{dba_str}]\n\n")
        
        # Write Top-K Accuracy
        f.write("Top-K Accuracy Per Time Slot:\n")
        for k, acc in topk_acc.items():
            acc_str = ", ".join([f"{a:.4f}" for a in acc])
            f.write(f"Top-{k} Accuracy: [{acc_str}]\n")
        f.write("=" * 50 + "\n\n")
    
    return val_loss, topk_acc, dba_score

def main():
    """Main function with overall parameters"""
    # Set random seed for reproducibility
    args = parse_args()
    

    # Set random seeds
    set_seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # Setup directories
    dayTime = datetime.datetime.now().strftime('%m-%d-%Y')
    hourTime = datetime.datetime.now().strftime('%H_%M')
    current_dir = os.path.dirname(__file__)
    
    # Set base save directory based on debug mode
    base_save_dir = 'saved_folder_debug' if args.debug else 'saved_folder_train'
    save_directory = os.path.join(current_dir, base_save_dir, f'bbox_{dayTime}_{hourTime}')
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    print(f'Saving to {save_directory}')

    with open(os.path.join(save_directory, 'params.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # Copy source files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file in ['train_model.py', 'Funcs.py', 'model.py']:
        if os.path.exists(os.path.join(script_dir, file)):
            shutil.copy(os.path.join(script_dir, file), save_directory)


    # Setup data paths
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_root = parent_dir + '/dataset/scenario9'
    
    train_csv_name = args.train_csv_name
    test_csv_name = args.test_csv_name
    
    train_dir = os.path.join(data_root, train_csv_name)
    test_dir = os.path.join(data_root, test_csv_name)
    
    # Setup data loaders
    train_loader = DataLoader(
        DataFeed(data_root, train_dir, args.seq_length, portion=args.dataset_pct), 
        batch_size=args.train_batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(
        DataFeed(data_root, test_dir, args.seq_length, portion=args.dataset_pct), 
        batch_size=args.test_batch_size, shuffle=False, generator=generator
    )
    print(f'TrainDataSize: {len(train_loader.dataset)}, TestDataSize: {len(val_loader.dataset)}')
    dataloaders = {'train': train_loader, 'test': val_loader}
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model
    net = GruModelSimple_v1(num_classes=args.num_classes, num_layers=args.gru_num_layers, hidden_size=args.gru_hidden_size, embed_size=args.feature_size)
    net.to(device)
    
    # Setup optimizer and scheduler
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
    print(f"Using CosineAnnealingWarmRestarts scheduler with T_0=10, T_mult=2, eta_min=0.0001")
    
    bbox_input = torch.randn(1,args.seq_length, 4).to(device)
    # h = net.initHidden(1).to(device)
    print(summary(net, bbox_input))

     # Save model summary and parameters to file
    with open(os.path.join(save_directory, 'params.txt'), 'a') as f:
        f.write("\n\nModel Architecture Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Capture model summary
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # Print model summary using student model
        try:
            print(summary(net, bbox_input))
        except Exception as e:
            print(f"Model summary could not be generated: {str(e)}")
            # Fallback: just print basic model info
            print(f"Model: {net.name}")
            print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
            print(f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
        
        sys.stdout = old_stdout
        model_summary = buffer.getvalue()
        
        f.write(model_summary)
        f.write("\n" + "=" * 50 + "\n")
    # Training
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, lrs = train(net, dataloaders, optimizer, scheduler, criterion, device, args, save_directory)       

        
    training_outputs = {
        'train_acc_hist': np.array(train_acc_hist),
        'train_loss_hist': np.array(train_loss_hist),
        'val_acc_hist': np.array(val_acc_hist),
        'val_loss_hist': np.array(val_loss_hist),
        'learning_rates': np.array(lrs)
    }
    
    # Save all outputs in a single numpy file
    np.savez(os.path.join(save_directory, 'training_outputs.npz'), **training_outputs)
    print(f"=====Combined Training outputs saved to numpy files in {save_directory}=====")


    # Plot training curves
    plot_training_curves(train_acc_hist, train_loss_hist, val_acc_hist, val_loss_hist, lrs, save_directory)
    
    print(f"Training completed. Results saved to: {save_directory}")

    # Final testing
    print("Starting final test...")
    net.load_state_dict(torch.load(os.path.join(save_directory, 'model_best.pth')))
    test_loss, test_acc, dba_score = test(net, val_loader, criterion, device, args, save_directory)
    

    print("Finished Training")
    print("Final test accuracy:", test_acc)
    

if __name__ == "__main__":
    main()
    
