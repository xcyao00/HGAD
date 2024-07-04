import argparse
from typing import List
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

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


def train(args):
    if args.dataset == 'mvtec':
        CLASS_NAMES = MVTEC_CLASS_NAMES 
        train_dataset = MVTEC(args.data_path, class_name=None, train=True,
                        img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
    elif args.dataset == 'btad':
        CLASS_NAMES = BTAD_CLASS_NAMES 
        train_dataset = BTAD(args.data_path, class_name=None, train=True,
                             img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
    elif args.dataset == 'mvtec3d':
        CLASS_NAMES = MVTEC3D_CLASS_NAMES
        train_dataset = MVTEC3D(args.data_path, class_name=None, train=True,
                                img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
    elif args.dataset == 'visa':
        CLASS_NAMES = VISA_CLASS_NAMES
        train_dataset = VISA(args.data_path, class_name=None, train=True,
                             img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
    elif args.dataset == 'union':
        CLASS_NAMES = MVTEC_CLASS_NAMES + BTAD_CLASS_NAMES + MVTEC3D_CLASS_NAMES + VISA_CLASS_NAMES
        train_dataset = UnionDataset(img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
    else:
        raise ValueError('Unrecognized or unsupported dataset!')
    args.class_to_idx = train_dataset.class_to_idx
    args.n_classes = len(CLASS_NAMES)
   
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=6, pin_memory=True, drop_last=True)
    
    model = HGAD(args)
    model.to(args.device)
    
    plot_columns = ['epoch', 'iteration', 'L_g', 'L_mi', 'L_e', 'L_g_intra', 'L_z', 'lr']
    train_loss_names = [column for column in plot_columns if column[0] == 'L']
    header_fmt = '{:<15}{:<15}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}' 
    
    output_fmt_live = '{:04d}/{:04d}      {:04d}/{:04d}    '
    fmts = ['{:10.5f}', '{:10.5f}', '{:10.5f}', '{:15.5f}', '{:10.5f}', '{:10.5f}']
    for i, name in enumerate(plot_columns[2:]):
        output_fmt_live += fmts[i]

    best_img_aucs, best_pixel_aucs = [0]*len(CLASS_NAMES), [0]*len(CLASS_NAMES)
    best_mean_img_auc, best_mean_pixel_auc = 0, 0
    
    for epoch in range(args.meta_epochs):
        adjust_learning_rate(args, model.optimizer, epoch)
        I = len(train_loader)
        for sub_epoch in range(args.sub_epochs):
            print(header_fmt.format(*plot_columns))  # print the header
            
            for idx, (image, label, _, _, _) in enumerate(train_loader):
                # warm-up learning rate
                lr = warmup_learning_rate(args, epoch, idx+sub_epoch*I, I*args.sub_epochs, model.optimizer)
            
                # x: (N, 3, 256, 256) y: (N, )
                image, label = image.to(args.device), label.to(args.device)  # (N, num_classes)
                
                with torch.no_grad():
                    features = model.encoder(image)
                
                for lvl in range(args.feature_levels):
                    e = features[lvl].detach()  
                    bs, dim, h, w = e.size()
                    e = e.permute(0, 2, 3, 1).reshape(-1, dim)  # (bs*h*w, dim)
                    
                    label_r = label.view(-1, 1, 1).repeat([1, h, w])
                    label_onehot = onehot(label_r.reshape(-1), len(CLASS_NAMES), args.label_smoothing)
                    
                    # (bs, 128, h, w)
                    pos_embed = nfs.positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
            
                    # losses: all loss items, L_x_tr, logits_tr, L_cNLL_tr, L_y_tr, acc_tr
                    losses = model(e, (label_r, label_onehot), pos_embed, scale=lvl, epoch=epoch)

                    if epoch < 2:  # only training with inter-class loss
                        loss = args.lambda1 * losses['L_g'] - args.lambda2 * losses['L_mi'] + losses['L_e']
                        losses['L_g_intra'] = torch.tensor([-1])
                        losses['L_z'] = torch.tensor([-1])
                    else:
                        loss = args.lambda1 * losses['L_g'] - args.lambda2 * losses['L_mi'] + losses['L_g_intra']  + losses['L_z'] + losses['L_e'] 
                    losses['loss'] = loss
                    
                    model.optimizer.zero_grad()
                    loss.backward()
                    model.optimizer.step()
                    
                print(output_fmt_live.format(*([
                                            epoch, args.meta_epochs,
                                            idx, len(train_loader)]
                                            + [losses[l].item() for l in train_loss_names] + [lr])),
                    flush=True, end='\r')

            # Validating every epoch
            img_aucs, pixel_aucs = validate(model, CLASS_NAMES, args)
            print("===============================================================================")
            for idx, class_name in enumerate(CLASS_NAMES):
                print('{}--Epoch[{}/{}], Image AUC: {:.3f}, Pixel AUC: {:.3f}'.
                    format(class_name, epoch, args.meta_epochs, img_aucs[idx], pixel_aucs[idx]))
            print('Average Image AUC: {:.3f}, Average Pixel AUC: {:.3f}'.format(np.mean(img_aucs), np.mean(pixel_aucs)))
            print("===============================================================================")
            if np.mean(img_aucs) > best_mean_img_auc:
                best_img_aucs = img_aucs
                best_mean_img_auc = np.mean(img_aucs)
                save_model(args.output_dir, model, epoch*args.sub_epochs+sub_epoch, args.dataset, flag='img')
            if np.mean(pixel_aucs) > best_mean_pixel_auc:
                best_pixel_aucs = pixel_aucs
                best_mean_pixel_auc = np.mean(pixel_aucs) 
                save_model(args.output_dir, model, epoch*args.sub_epochs+sub_epoch, args.dataset, flag='pix')
    
    for i, class_name in enumerate(CLASS_NAMES):
        print('{}: Image AUC: {:.3f}, Pixel AUC: {:.3f}'.format(class_name, best_img_aucs[i], best_pixel_aucs[i]))
    print('Average Image AUC: {:.3f}, Average Pixel AUC: {:.3f}'.format(np.mean(best_img_aucs), np.mean(best_pixel_aucs)))
    print('Best Mean Image AUC: {:.3f}, Best Mean Pixel AUC: {:.3f}'.format(best_mean_img_auc, best_mean_pixel_auc))


def validate(model: HGAD, class_names: List, args: argparse.Namespace):
    img_aucs, pixel_aucs = [], []
    for class_id, class_name in enumerate(class_names):
        if class_name in MVTEC_CLASS_NAMES:
            test_dataset = MVTEC(args.data_path, class_name=class_name, train=False,
                                 img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
        elif class_name in BTAD_CLASS_NAMES:
            test_dataset = BTAD(args.data_path, class_name=class_name, train=False,
                                img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
        elif class_name in MVTEC3D_CLASS_NAMES:
            test_dataset = MVTEC3D(args.data_path, class_name=class_name, train=False,
                                   img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
        elif class_name in VISA_CLASS_NAMES:
            test_dataset = VISA(args.data_path, class_name=class_name, train=False,
                                img_size=args.img_size, crp_size=args.img_size, msk_size=args.msk_size)
        else:
            raise ValueError('Unrecognized or unsupported class!')
        class_id = args.class_to_idx[class_name]
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
        
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
                    
                    # (bs, 128, h, w)
                    pos_embed = nfs.positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                
                    z, log_jac_det = model.nfs[lvl](e, [pos_embed, ])
                    z, log_jac_det = z.reshape(-1, z.shape[-1]), log_jac_det.reshape(-1)
                    
                    mu_per_scale = model.mus[lvl]  # (n_classes, dim)
                    class_centers = mu_per_scale
                    mu_delta_per_scale = model.mu_deltas[lvl]  # (n_classes, n_centers - 1, dim)
                    mu_per_scale = mu_per_scale.unsqueeze(1)
                    mu_per_scale_ = mu_per_scale + mu_delta_per_scale
                    mu_per_scale = torch.cat([mu_per_scale, mu_per_scale_], dim=1)  # (n_classes, n_centers, dim)
                    
                    phi_per_scale = model.phi_intras[lvl]
                        
                    mu_y = e.new_full((image.shape[0], ), class_id, dtype=torch.long)  # (1, )
                    mu_intra = mu_per_scale[mu_y, :, :] # (1, num_centers, dim)
                    mu_intra = mu_intra.expand([e.shape[0], mu_intra.shape[1], dim])  # (N, num_centers, dim)
                    log_py_intra = torch.log_softmax(phi_per_scale, dim=1).unsqueeze(0)  # (1, num_classes, n_centers)
                    
                    log_py = log_py_intra[:, mu_y, :].squeeze(0)  # (1, num_centers)
                    log_py = log_py.expand([e.shape[0], log_py.shape[1]])  # (N, num_centers)
                    logps = model.get_logps(z, mu_intra, log_py, log_jac_det, model.feat_dims[lvl]) 
                    logps = logps / model.feat_dims[lvl]
                    
                    zz = model.calculate_distances_to_inter_class_centers(z, mu=class_centers)  # (N, num_classes)
                    
                    logits = -0.5 * zz  # (N, num_classes)
                    entropy = -torch.sum(-torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1)  # (N, )
                   
                    logps_list[lvl].append(logps.reshape(bs, h, w).cpu())
                    entropy_list[lvl].append(entropy.reshape(bs, h, w).cpu())
        progress_bar.close()

        scores1 = convert_to_anomaly_scores(args, logps_list)
        scores2 = convert_to_anomaly_scores(args, entropy_list)
        
        # merging logps and entropy 
        scores = scores1 * scores2
        img_scores = np.max(scores, axis=(1, 2))
        gt_label = np.asarray(gt_label_list, dtype=bool)
        img_auc = roc_auc_score(gt_label, img_scores)
        gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
        pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())

        img_aucs.append(img_auc), pixel_aucs.append(pix_auc)
    
    return img_aucs, pixel_aucs


def convert_to_anomaly_scores(args, logps_list):
    normal_map = [list() for _ in range(args.feature_levels)]
    for l in range(args.feature_levels):
        logps = torch.cat(logps_list[l], dim=0)  
        logps-= torch.max(logps) # normalize log-likelihoods to (-Inf:0] by subtracting a constant
        probs = torch.exp(logps) # convert to probs in range [0:1]
        # upsample
        normal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=args.msk_size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(normal_map[0])
    for l in range(args.feature_levels):
        scores += normal_map[l]

    # normality score to anomaly score
    scores = scores.max() - scores 
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores
