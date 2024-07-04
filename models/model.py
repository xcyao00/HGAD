import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nf_model import load_flow_model


class HGAD(nn.Module):
    
    _GCONST_ = -0.9189385332046727 # ln(1/sqrt(2*pi))
    
    def __init__(self, args):
        super(HGAD, self).__init__()
        self.args = args  
        self.log_theta = torch.nn.LogSigmoid()

        self.encoder = Encoder(args)
        feat_dims = self.encoder.feat_dims
        self.feat_dims = feat_dims * args.feature_levels
        
        self.n_centers_each_class = args.n_intra_centers
        self.n_classes = args.n_classes
        
        self.nfs = [load_flow_model(args, feat_dim) for feat_dim in self.feat_dims]
        self.nfs = [nf.to(args.device) for nf in self.nfs]
        
        self.phi_inters = nn.ParameterList()  # inter-class weights
        self.phi_intras = nn.ParameterList()  # intra-class weights
        self.mus = nn.ParameterList()  # main class centers
        self.mu_deltas = nn.ParameterList()  # delta class centers
        for l in range(args.feature_levels):  # for each feature level
            mu = torch.zeros(self.n_classes, self.feat_dims[l])  # (n_classes, dim)
            mu_delta = torch.zeros(self.n_classes, self.n_centers_each_class - 1, self.feat_dims[l])  # (n_classes, n_centers - 1, dim)
            
            init_scale = 5.0 / np.sqrt(2 * self.feat_dims[l] // self.n_classes)
            
            # used to distinct each main center of each class
            for k in range(self.feat_dims[l] // self.n_classes):
                mu[:, self.n_classes * k : self.n_classes * (k+1)] = init_scale * torch.eye(self.n_classes)
                    
            # Initilize Gaussian centers for each class
            for class_id in range(self.n_classes):
                mu_delta[class_id, :, :] += torch.randn(self.n_centers_each_class - 1, self.feat_dims[l]) 
                
            mu = nn.Parameter(mu)   
            mu_delta = nn.Parameter(mu_delta)
            phi_intra = nn.Parameter(torch.zeros(self.n_classes, self.n_centers_each_class))  # (n_classes, n_centers)
            phi_inter = nn.Parameter(torch.zeros(self.n_classes))
            
            self.mus.append(mu)
            self.mu_deltas.append(mu_delta)
            self.phi_inters.append(phi_inter)
            self.phi_intras.append(phi_intra)

        # normalizing flows' parameters
        optimizer_params = list(self.nfs[0].parameters())
        for l in range(1, args.feature_levels):
            optimizer_params += list(self.nfs[l].parameters())
        optimizer_params = [{'params': list(filter(lambda p: p.requires_grad, optimizer_params))}, ]
        
        # parameters of class centers
        optimizer_params.append({'params': [mu for mu in self.mus] + [mu_delta for mu_delta in self.mu_deltas],
                                    'lr': args.lr * 3.0,
                                    'weight_decay': 0.})
        optimizer_params[-1]['betas'] = [0.95, 0.99]  

        # parameters of class weights
        optimizer_params.append({'params': [phi for phi in self.phi_intras] + [phi for phi in self.phi_inters],
                                    'lr': args.lr * 1.0,
                                    'weight_decay': 0.})
        optimizer_params[-1]['betas'] = [0.95, 0.99]

        self.optimizer = torch.optim.Adam(optimizer_params, args.lr,
                                            betas=[0.95, 0.99],
                                            weight_decay=1e-4)
    
    def calculate_distances_to_inter_class_centers(self, z, mu):
        """
        Compute the distances among each latent feature to inter class centers.
        Args:
            z (Tensor): (N, dim).
            mu (Tensor): (num_classes, dim).
        """
        z_i_z_i = torch.sum(z**2, dim=1, keepdim=True)   # (N, 1)
        mu_j_mu_j = torch.sum(mu**2, dim=1).unsqueeze(0)        # (1, num_classes)
        z_i_mu_j = torch.mm(z, mu.t())    # (N, num_classes)

        return -2 * z_i_mu_j + z_i_z_i + mu_j_mu_j
    
    def calculate_distances_to_intra_class_centers(self, z, mu):
        """
        Compute the distances among each latent feature to intra class centers.
        Args:
            z (Tensor): (N, dim).
            mu (Tensor): (num_classes, num_centers, dim).
        """
        num_classes, num_centers = mu.shape[0], mu.shape[1]
        z_i_z_i = torch.sum(z**2, dim=1, keepdim=True).unsqueeze(-1)   # (N, 1, 1)
        mu_j_mu_j = torch.sum(mu**2, dim=2).unsqueeze(0)         # (1, num_classes, num_centers)
        mu = mu.reshape(-1, mu.shape[2])  # (num_classes*num_centers, dim)
        z_i_mu_j = torch.mm(z, mu.t()).reshape(-1, num_classes, num_centers)    # (N, num_classes, num_centers)

        return -2 * z_i_mu_j + z_i_z_i + mu_j_mu_j

    def get_logps(self, z, mu, log_py, logdet_J, C):
        """
        Compute the log-likelihoods for each feature in its corresponding intra-class distribution.
        Args:
            z (Tensor): (N, dim).
            mu (Tensor): (N, num_centers, dim).
            log_py (Tensor): (N, num_centers).
            logdet_J (Tensor): (N, ).
            C (int): feature dimensions.
        """
        z = z.unsqueeze(1)  # (N, 1, dim)
        logps = torch.logsumexp(-0.5 * torch.sum((z - mu)**2, 2) + log_py, dim=1)  # (N, )
        logps = C * self._GCONST_  + logps + logdet_J
        
        return logps

    def forward(self, x, y=None, pos_embed=None, scale=0, epoch=0):
        y, y_onehot = y
        
        # x: (N, dim); z : (N, dim)
        z, log_jac_det = self.nfs[scale](x, [pos_embed, ])
        z, log_jac_det = z.reshape(-1, z.shape[-1]), log_jac_det.reshape(-1)
        
        losses = {}
        if epoch < 2:  # only training with inter-class loss
            class_centers = self.mus[scale]  # (n_classes, dim)
        
            py_inter = torch.softmax(self.phi_inters[scale], dim=0).view(1, -1)  # (1, n_classes)
            log_py_inter = torch.log_softmax(self.phi_inters[scale], dim=0).view(1, -1)  # (1, n_classes)
            
            zz = self.calculate_distances_to_inter_class_centers(z, mu=class_centers)  # (N, n_classes)

            losses = {}
            # L_g: loss function with the inter-class Gaussian mixture prior, E.q.(6) in the paper
            losses['L_g'] = (-torch.logsumexp(-0.5 * zz + log_py_inter, dim=1) - log_jac_det) / self.feat_dims[scale]
            # L_e: entropy loss, E.q.(9) in the paper
            entropy = torch.sum(-torch.softmax(-0.5 * zz, dim=1) * torch.log_softmax(-0.5 * zz, dim=1), dim=1)  # (N, )
            losses['L_e'] = entropy

            py_inter = py_inter.detach()
            log_py_inter = log_py_inter.detach()
            if y_onehot is not None:
                # L_mi: mutual information maximization loss, E.q.(8) in the paper
                losses['L_mi'] = torch.sum((torch.log_softmax(-0.5 * zz + log_py_inter, dim=1) - log_py_inter) * y_onehot, dim=1)  # (N, )
        else:
            main_centers = self.mus[scale]  # (n_classes, dim)
            # when optimizing mu_delta, detach the main centers
            mu_per_scale = main_centers.detach()
            mu_delta_per_scale = self.mu_deltas[scale]  # (n_classes, n_centers - 1, dim)
            mu_per_scale = mu_per_scale.unsqueeze(1)
            # adding with delta centers
            mu_per_scale_ = mu_per_scale + mu_delta_per_scale
            mu_per_scale = torch.cat([mu_per_scale, mu_per_scale_], dim=1)  # (n_classes, n_centers, dim)
            
            phi_per_scale = self.phi_intras[scale]
        
            y = y.reshape(-1)  # (bs, h, w) - > (N, )
            # (n_classes, n_centers, dim) -> (N, n_centers, dim)
            # For each input z, finding its corresponding class centers
            mu_intra = mu_per_scale[y, :, :]  # (N, n_centers, dim)

            py_inter = torch.softmax(self.phi_inters[scale], dim=0).view(1, -1)  # (1, n_classes)
            log_py_inter = torch.log_softmax(self.phi_inters[scale], dim=0).view(1, -1)  # (1, n_classes)
            log_py_intra = torch.log_softmax(phi_per_scale, dim=1).unsqueeze(0)  # (1, n_classes, n_centers)

            zz = self.calculate_distances_to_inter_class_centers(z, mu=main_centers)  # (N, n_classes)
            zz2 = self.calculate_distances_to_intra_class_centers(z, mu=mu_per_scale)  # (N, num_classes, num_centers)
            
            losses = {}
            # L_g: loss function with the inter-class Gaussian mixture prior, E.q.(6) in the paper
            losses['L_g'] = (-torch.logsumexp(-0.5 * zz + log_py_inter, dim=1) - log_jac_det) / self.feat_dims[scale]
            
            y_mask = y_onehot > 0.5  # (N, num_centers)
            log_py = log_py_intra[:, y, :].squeeze(0)  # (N, num_centers)
            zz_per_class = zz2[y_mask, :]  # (N, num_centers), each feature intra-class distances
            # L_g_intra: loss function with the intra-class mixted centers
            losses['L_g_intra'] = (-torch.logsumexp(-0.5 * zz_per_class + log_py, dim=1) - log_jac_det) / self.feat_dims[scale]  # (N, )
            
            logps = self.get_logps(z, mu_intra, log_py, log_jac_det, self.feat_dims[scale]) # (N, )
            logps = logps / self.feat_dims[scale]
            # L_z: maximum likelihood loss
            losses['L_z'] = -self.log_theta(logps)
            
            # L_e: entropy loss, E.q.(9) in the paper
            entropy = torch.sum(-torch.softmax(-0.5 * zz, dim=1) * torch.log_softmax(-0.5 * zz, dim=1), dim=1)  # (N, )
            losses['L_e'] = entropy

            log_py_inter = log_py_inter.detach()
            log_py_intra = log_py_intra.detach()
            if y_onehot is not None:
                # L_mi: mutual information maximization loss, E.q.(8) in the paper
                losses['L_mi'] = torch.sum((torch.log_softmax(-0.5 * zz + log_py_inter, dim=1) - log_py_inter) * y_onehot, dim=1)  # (N, )
            
        for k, v in losses.items():
            losses[k] = torch.mean(v)

        return losses


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        self.args = args
        self.feature_extractor = timm.create_model(args.backbone_arch, features_only=True, 
                    out_indices=[3], pretrained=True)
        self.feature_extractor = self.feature_extractor.to(args.device).eval()
        self.feat_dims = self.feature_extractor.feature_info.channels()

    def forward(self, x):
        outs = list()
        for s in range(self.args.feature_levels):
            x_ = F.interpolate(x, size=(self.args.img_size // (2 ** s), self.args.img_size // (2 ** s)), mode='bilinear', align_corners=True) if s > 0 else x
            feat = self.feature_extractor(x_)
            outs.extend(feat)
            
        return outs
