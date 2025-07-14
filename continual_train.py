import argparse
from typing import List
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import os
import csv
from datetime import datetime
import copy

import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models.nf_model as nfs
from models.model import HGAD
from models.utils import save_model
from datasets.mvtec import MVTEC, MVTEC_CLASS_NAMES
from datasets.btad import BTAD, BTAD_CLASS_NAMES
from datasets.mvtec_3d import MVTEC3D, MVTEC3D_CLASS_NAMES
from datasets.visa import VISA, VISA_CLASS_NAMES
from datasets.union import UnionDataset
from utils import adjust_learning_rate, warmup_learning_rate, onehot


# MVTecAD 클래스들을 5개의 task로 분할
CONTINUAL_TASKS = [
    ['bottle', 'cable', 'capsule'],           # Task 1
    ['carpet', 'grid', 'hazelnut'],           # Task 2
    ['leather', 'metal_nut', 'pill'],         # Task 3
    ['screw', 'tile', 'toothbrush'],          # Task 4
    ['transistor', 'wood', 'zipper']          # Task 5
]

# Create global class mapping
ALL_CLASSES = []
for task in CONTINUAL_TASKS:
    ALL_CLASSES.extend(task)

GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}


def save_training_log(log_dir, task_id, epoch, sub_epoch, iteration, losses, lr):
    """Save training iteration logs to CSV file"""
    log_file = os.path.join(log_dir, f"training_log_task_{task_id}.csv")
    
    # Check if file exists to write header
    write_header = not os.path.exists(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(['timestamp', 'task_id', 'epoch', 'sub_epoch', 'iteration', 'L_g', 'L_mi', 'L_e', 'L_g_intra', 'L_z', 'lr'])
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp, task_id, epoch, sub_epoch, iteration,
            losses['L_g'].item(), losses['L_mi'].item(), losses['L_e'].item(),
            losses['L_g_intra'].item() if isinstance(losses['L_g_intra'], torch.Tensor) else losses['L_g_intra'],
            losses['L_z'].item() if isinstance(losses['L_z'], torch.Tensor) else losses['L_z'],
            lr
        ])

def save_evaluation_results(result_dir, task_id, epoch, sub_epoch, class_names, img_aucs, pixel_aucs):
    """Save evaluation results to CSV file"""
    result_file = os.path.join(result_dir, f"evaluation_results_task_{task_id}.csv")
    
    # Create expected header for current task
    expected_header = ['timestamp', 'task_id', 'epoch', 'sub_epoch']
    for class_name in class_names:
        expected_header.extend([f'{class_name}_img_auc', f'{class_name}_pixel_auc'])
    expected_header.extend(['mean_img_auc', 'mean_pixel_auc'])
    
    # Check if file exists and validate header compatibility
    write_header = True
    if os.path.exists(result_file):
        # Read existing header to check compatibility
        with open(result_file, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
                # Check if headers match
                if existing_header == expected_header:
                    write_header = False
                else:
                    # Headers don't match, need to rewrite file
                    print(f"Warning: Header mismatch in {result_file}. Expected: {expected_header}, Found: {existing_header}")
                    print("Recreating file with correct header...")
            except StopIteration:
                # Empty file, need to write header
                write_header = True
    
    # Open file in appropriate mode
    file_mode = 'w' if write_header else 'a'
    
    with open(result_file, file_mode, newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(expected_header)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, task_id, epoch, sub_epoch]
        for i, class_name in enumerate(class_names):
            row.extend([img_aucs[i], pixel_aucs[i]])
        row.extend([np.mean(img_aucs), np.mean(pixel_aucs)])
        writer.writerow(row)

def save_continual_results(result_dir, task_id, current_classes, all_classes, img_aucs, pixel_aucs):
    """Save continual learning results including all previous tasks"""
    result_file = os.path.join(result_dir, f"continual_results_after_task_{task_id}.csv")
    
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['class_name', 'task_id', 'img_auc', 'pixel_auc', 'is_current_task']
        writer.writerow(header)
        
        # Write results for each class
        for i, class_name in enumerate(all_classes):
            task_idx = -1
            for t_id, task_classes in enumerate(CONTINUAL_TASKS):
                if class_name in task_classes:
                    task_idx = t_id
                    break
            
            is_current = class_name in current_classes
            writer.writerow([class_name, task_idx, img_aucs[i], pixel_aucs[i], is_current])
        
        # Write summary
        writer.writerow(['Average', '', np.mean(img_aucs), np.mean(pixel_aucs), ''])
        
        # Write task-wise averages
        for t_id, task_classes in enumerate(CONTINUAL_TASKS[:task_id + 1]):
            task_indices = [i for i, cls in enumerate(all_classes) if cls in task_classes]
            if task_indices:
                task_img_aucs = [img_aucs[i] for i in task_indices]
                task_pixel_aucs = [pixel_aucs[i] for i in task_indices]
                writer.writerow([f'Task_{t_id}_Average', t_id, np.mean(task_img_aucs), np.mean(task_pixel_aucs), ''])

def save_unified_evaluation_results(result_dir, task_id, epoch, sub_epoch, all_classes, img_aucs, pixel_aucs):
    """Save evaluation results for all tasks in a unified file"""
    result_file = os.path.join(result_dir, "unified_evaluation_results.csv")
    
    # Check if file exists to write header
    write_header = not os.path.exists(result_file)
    
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            # Create header with all possible classes
            header = ['timestamp', 'current_task_id', 'epoch', 'sub_epoch']
            for class_name in ALL_CLASSES:
                header.extend([f'{class_name}_img_auc', f'{class_name}_pixel_auc'])
            header.extend(['mean_img_auc', 'mean_pixel_auc'])
            writer.writerow(header)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, task_id, epoch, sub_epoch]
        
        # Create a mapping from class name to AUC values
        class_to_img_auc = {}
        class_to_pixel_auc = {}
        
        for i, class_name in enumerate(all_classes):
            class_to_img_auc[class_name] = img_aucs[i]
            class_to_pixel_auc[class_name] = pixel_aucs[i]
        
        # Add AUC values for all classes (use -1 for not evaluated classes)
        for class_name in ALL_CLASSES:
            if class_name in class_to_img_auc:
                row.extend([class_to_img_auc[class_name], class_to_pixel_auc[class_name]])
            else:
                row.extend([-1, -1])  # Not evaluated yet
        
        row.extend([np.mean(img_aucs), np.mean(pixel_aucs)])
        writer.writerow(row)


def create_task_dataset(args, task_classes, train=True):
    """Create dataset for specific task classes"""
    if args.dataset == 'mvtec':
        # Create individual datasets for each class in the task
        filtered_data = []
        
        for class_name in task_classes:
            # Create dataset for specific class
            class_dataset = MVTEC(args.data_path, class_name=class_name, train=train,
                                img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
            
            # Add all data from this class to filtered_data
            for i in range(len(class_dataset)):
                image, label, mask, name, path = class_dataset[i]
                # Use global class ID instead of task-specific ID
                global_class_id = GLOBAL_CLASS_TO_IDX[class_name]
                filtered_data.append((image, global_class_id, mask, name, path))
    else:
        raise ValueError(f'Dataset {args.dataset} not supported for continual learning')
    
    # Create new dataset with filtered data
    class TaskDataset:
        def __init__(self, data, class_names, class_to_idx):
            self.data = data
            self.class_names = class_names
            self.class_to_idx = class_to_idx
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Create global class mapping
    global_class_to_idx = {cls: GLOBAL_CLASS_TO_IDX[cls] for cls in task_classes}
    
    return TaskDataset(filtered_data, task_classes, global_class_to_idx)


def train_single_task(model, task_id, task_classes, args):
    """Train model on a single task"""
    print(f"\n=== Training Task {task_id}: {task_classes} ===")
    
    # Create task dataset
    train_dataset = create_task_dataset(args, task_classes, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=6, pin_memory=True, drop_last=True)
    
    # Update model parameters for current task
    # For continual learning, use the total number of classes seen so far
    total_classes_so_far = (task_id + 1) * args.classes_per_task
    
    # If this is not the first task, expand the model to accommodate new classes
    if task_id > 0:
        model.expand_for_new_classes(len(task_classes))
    
    plot_columns = ['epoch', 'iteration', 'L_g', 'L_mi', 'L_e', 'L_g_intra', 'L_z', 'lr']
    train_loss_names = [column for column in plot_columns if column[0] == 'L']
    header_fmt = '{:<15}{:<15}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}' 
    
    output_fmt_live = '{:04d}/{:04d}      {:04d}/{:04d}    '
    fmts = ['{:10.5f}', '{:10.5f}', '{:10.5f}', '{:15.5f}', '{:10.5f}', '{:10.5f}']
    for i, name in enumerate(plot_columns[2:]):
        output_fmt_live += fmts[i]

    best_img_aucs, best_pixel_aucs = [0]*len(task_classes), [0]*len(task_classes)
    best_mean_img_auc, best_mean_pixel_auc = 0, 0
    
    for epoch in range(args.meta_epochs):
        adjust_learning_rate(args, model.optimizer, epoch)
        I = len(train_loader)
        for sub_epoch in range(args.sub_epochs):
            print(header_fmt.format(*plot_columns))
            
            for idx, (image, label, _, _, _) in enumerate(train_loader):
                # warm-up learning rate
                lr = warmup_learning_rate(args, epoch, idx+sub_epoch*I, I*args.sub_epochs, model.optimizer)
            
                image, label = image.to(args.device), label.to(args.device)
                
                with torch.no_grad():
                    features = model.encoder(image)
                
                for lvl in range(args.feature_levels):
                    e = features[lvl].detach()  
                    bs, dim, h, w = e.size()
                    e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                    
                    label_r = label.view(-1, 1, 1).repeat([1, h, w])
                    # Use total number of classes for onehot encoding
                    label_onehot = onehot(label_r.reshape(-1), model.n_classes, args.label_smoothing)
                    
                    pos_embed = nfs.positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
            
                    losses = model(e, (label_r, label_onehot), pos_embed, scale=lvl, epoch=epoch)

                    if epoch < 2:
                        loss = args.lambda1 * losses['L_g'] - args.lambda2 * losses['L_mi'] + losses['L_e']
                        losses['L_g_intra'] = torch.tensor([-1])
                        losses['L_z'] = torch.tensor([-1])
                    else:
                        loss = args.lambda1 * losses['L_g'] - args.lambda2 * losses['L_mi'] + losses['L_g_intra']  + losses['L_z'] + losses['L_e'] 
                    losses['loss'] = loss
                    
                    model.optimizer.zero_grad()
                    loss.backward()
                    model.optimizer.step()
                    
                    save_training_log(args.log_dir, task_id, epoch, sub_epoch, idx, losses, lr)
                    
                print(output_fmt_live.format(*([
                                            epoch, args.meta_epochs,
                                            idx, len(train_loader)]
                                            + [losses[l].item() for l in train_loss_names] + [lr])),
                    flush=True, end='\r')

            # Validate on current task
            img_aucs, pixel_aucs = validate_single_task(model, task_classes, task_id, args)
            
            save_evaluation_results(args.result_dir, task_id, epoch, sub_epoch, task_classes, img_aucs, pixel_aucs)
            
            # Save to unified evaluation results file
            save_unified_evaluation_results(args.result_dir, task_id, epoch, sub_epoch, task_classes, img_aucs, pixel_aucs)
            
            print("\n" + "="*80)
            for idx, class_name in enumerate(task_classes):
                print(f'{class_name}--Epoch[{epoch}/{args.meta_epochs}], Image AUC: {img_aucs[idx]:.3f}, Pixel AUC: {pixel_aucs[idx]:.3f}')
            print(f'Task {task_id} Average Image AUC: {np.mean(img_aucs):.3f}, Average Pixel AUC: {np.mean(pixel_aucs):.3f}')
            print("="*80)
            
            if np.mean(img_aucs) > best_mean_img_auc:
                best_img_aucs = img_aucs
                best_mean_img_auc = np.mean(img_aucs)
                save_model(os.path.join(args.log_dir, f'task_{task_id}'), model, 
                          epoch*args.sub_epochs+sub_epoch, f'{args.dataset}_task_{task_id}', flag='img')
            if np.mean(pixel_aucs) > best_mean_pixel_auc:
                best_pixel_aucs = pixel_aucs
                best_mean_pixel_auc = np.mean(pixel_aucs)
                save_model(os.path.join(args.log_dir, f'task_{task_id}'), model, 
                          epoch*args.sub_epochs+sub_epoch, f'{args.dataset}_task_{task_id}', flag='pix')
    
    return best_img_aucs, best_pixel_aucs


def validate_single_task(model: HGAD, task_classes: List, task_id: int, args: argparse.Namespace):
    """Validate model on a single task"""
    img_aucs, pixel_aucs = [], []
    
    for class_name in task_classes:
        if class_name in MVTEC_CLASS_NAMES:
            test_dataset = MVTEC(args.data_path, class_name=class_name, train=False,
                                 img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
        else:
            raise ValueError(f'Class {class_name} not supported')
        
        # Use global class ID
        global_class_id = GLOBAL_CLASS_TO_IDX[class_name]
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                                 drop_last=False, pin_memory=True, num_workers=4)
        
        gt_label_list, gt_mask_list = [], []
        logps_list = [[] for _ in range(args.feature_levels)]
        entropy_list = [[] for _ in range(args.feature_levels)]
        
        progress_bar = tqdm(total=len(test_loader))
        progress_bar.set_description(f"Evaluating {class_name}")
        
        for idx, (image, label, mask, _, _) in enumerate(test_loader):
            progress_bar.update(1)
            gt_label_list.extend(label.cpu().numpy())
            gt_mask_list.extend(mask.cpu().numpy())
            
            image = image.to(args.device)
            
            with torch.no_grad():
                features = model.encoder(image)

                for lvl in range(args.feature_levels):
                    e = features[lvl].detach() 
                    bs, dim, h, w = e.size()
                    e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                    
                    pos_embed = nfs.positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                
                    z, log_jac_det = model.nfs[lvl](e, [pos_embed, ])
                    z, log_jac_det = z.reshape(-1, z.shape[-1]), log_jac_det.reshape(-1)
                    
                    mu_per_scale = model.mus[lvl]
                    class_centers = mu_per_scale
                    mu_delta_per_scale = model.mu_deltas[lvl]
                    mu_per_scale = mu_per_scale.unsqueeze(1)
                    mu_per_scale_ = mu_per_scale + mu_delta_per_scale
                    mu_per_scale = torch.cat([mu_per_scale, mu_per_scale_], dim=1)
                    
                    phi_per_scale = model.phi_intras[lvl]
                        
                    mu_y = e.new_full((image.shape[0], ), global_class_id, dtype=torch.long)
                    mu_intra = mu_per_scale[mu_y, :, :]
                    mu_intra = mu_intra.expand([e.shape[0], mu_intra.shape[1], dim])
                    log_py_intra = torch.log_softmax(phi_per_scale, dim=1).unsqueeze(0)
                    
                    log_py = log_py_intra[:, mu_y, :].squeeze(0)
                    log_py = log_py.expand([e.shape[0], log_py.shape[1]])
                    logps = model.get_logps(z, mu_intra, log_py, log_jac_det, model.feat_dims[lvl]) 
                    logps = logps / model.feat_dims[lvl]
                    
                    zz = model.calculate_distances_to_inter_class_centers(z, mu=class_centers)
                    
                    logits = -0.5 * zz
                    entropy = -torch.sum(-torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1)
                   
                    logps_list[lvl].append(logps.reshape(bs, h, w).cpu())
                    entropy_list[lvl].append(entropy.reshape(bs, h, w).cpu())
        
        progress_bar.close()

        scores1 = convert_to_anomaly_scores(args, logps_list)
        scores2 = convert_to_anomaly_scores(args, entropy_list)
        
        scores = scores1 * scores2
        img_scores = np.max(scores, axis=(1, 2))
        gt_label = np.asarray(gt_label_list, dtype=bool)
        img_auc = roc_auc_score(gt_label, img_scores)
        gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
        pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())

        img_aucs.append(img_auc)
        pixel_aucs.append(pix_auc)
    
    return img_aucs, pixel_aucs


def validate_all_tasks(model: HGAD, task_id: int, args: argparse.Namespace):
    """Validate model on all tasks seen so far"""
    all_classes = []
    all_img_aucs = []
    all_pixel_aucs = []
    
    for t_id in range(task_id + 1):
        task_classes = CONTINUAL_TASKS[t_id]
        img_aucs, pixel_aucs = validate_single_task(model, task_classes, t_id, args)
        
        all_classes.extend(task_classes)
        all_img_aucs.extend(img_aucs)
        all_pixel_aucs.extend(pixel_aucs)
    
    return all_classes, all_img_aucs, all_pixel_aucs


def continual_train(args):
    """Main continual learning training loop"""
    print("Starting Continual Learning Training")
    print(f"Tasks: {CONTINUAL_TASKS}")
    
    # Initialize model with all classes (for simplicity)
    # In practice, this could be done more efficiently
    total_classes = len(ALL_CLASSES)
    args.n_classes = args.classes_per_task  # Start with first task classes
    args.class_to_idx = {cls: GLOBAL_CLASS_TO_IDX[cls] for cls in CONTINUAL_TASKS[0]}
    
    model = HGAD(args)
    model.to(args.device)
    
    # Store results for all tasks
    all_results = {}
    
    for task_id in range(args.num_tasks):
        task_classes = CONTINUAL_TASKS[task_id]
        
        # Train on current task
        task_img_aucs, task_pixel_aucs = train_single_task(model, task_id, task_classes, args)
        
        # Store current task results
        all_results[f'task_{task_id}'] = {
            'classes': task_classes,
            'img_aucs': task_img_aucs,
            'pixel_aucs': task_pixel_aucs
        }
        
        # Evaluate on all tasks seen so far
        print(f"\n=== Evaluating all tasks after Task {task_id} ===")
        all_classes, all_img_aucs, all_pixel_aucs = validate_all_tasks(model, task_id, args)
        
        # Save continual learning results
        save_continual_results(args.result_dir, task_id, task_classes, all_classes, all_img_aucs, all_pixel_aucs)
        
        # Save unified evaluation results (using final epoch values)
        final_epoch = args.meta_epochs - 1
        final_sub_epoch = args.sub_epochs - 1
        save_unified_evaluation_results(args.result_dir, task_id, final_epoch, final_sub_epoch, all_classes, all_img_aucs, all_pixel_aucs)
        
        # Print summary
        print(f"\nResults after Task {task_id}:")
        for i, class_name in enumerate(all_classes):
            print(f'{class_name}: Image AUC: {all_img_aucs[i]:.3f}, Pixel AUC: {all_pixel_aucs[i]:.3f}')
        print(f'Overall Average Image AUC: {np.mean(all_img_aucs):.3f}')
        print(f'Overall Average Pixel AUC: {np.mean(all_pixel_aucs):.3f}')
        
        # Calculate and print task-wise averages
        print("\nTask-wise averages:")
        for t_id in range(task_id + 1):
            task_cls = CONTINUAL_TASKS[t_id]
            task_indices = [i for i, cls in enumerate(all_classes) if cls in task_cls]
            if task_indices:
                task_img_avg = np.mean([all_img_aucs[i] for i in task_indices])
                task_pixel_avg = np.mean([all_pixel_aucs[i] for i in task_indices])
                print(f'Task {t_id} ({task_cls}): Image AUC: {task_img_avg:.3f}, Pixel AUC: {task_pixel_avg:.3f}')
    
    # Save final summary
    final_summary_file = os.path.join(args.result_dir, "final_continual_summary.csv")
    with open(final_summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['final_average_img_auc', np.mean(all_img_aucs)])
        writer.writerow(['final_average_pixel_auc', np.mean(all_pixel_aucs)])
        writer.writerow(['total_classes', len(all_classes)])
        writer.writerow(['total_tasks', args.num_tasks])
    
    print("\nContinual Learning Training Completed!")
    print(f"Final Average Image AUC: {np.mean(all_img_aucs):.3f}")
    print(f"Final Average Pixel AUC: {np.mean(all_pixel_aucs):.3f}")


def convert_to_anomaly_scores(args, logps_list):
    """Convert log probabilities to anomaly scores"""
    normal_map = [list() for _ in range(args.feature_levels)]
    for l in range(args.feature_levels):
        logps = torch.cat(logps_list[l], dim=0)  
        logps-= torch.max(logps)
        probs = torch.exp(logps)
        normal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=args.msk_size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    scores = np.zeros_like(normal_map[0])
    for l in range(args.feature_levels):
        scores += normal_map[l]

    scores = scores.max() - scores 
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores 