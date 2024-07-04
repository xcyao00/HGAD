import os
import torch


def save_model(path, model, current_epoch, dataset_name, flag='img'):
    os.makedirs(path, exist_ok=True)
    state_dict = {'epoch': current_epoch,
                  'nfs': [nf.state_dict() for nf in model.nfs],
                  'phi_inters': model.phi_inters.state_dict(),
                  'phi_intras': model.phi_intras.state_dict(),
                  'mus': model.mus.state_dict(),
                  'mu_deltas': model.mu_deltas.state_dict()}
    torch.save(state_dict, os.path.join(path, f'HGAD_{dataset_name}_{flag}.pt'))
    print('Saving model weights into {}'.format(os.path.join(path, f'HGAD_{dataset_name}_{flag}.pt')))


def load_model(path, model):
    state_dict = torch.load(path)
    model.nfs = [nf.load_state_dict(state, strict=False) for nf, state in zip(model.nfs, state_dict['nfs'])]
    model.phi_inters.load_state_dict(state_dict['phi_inters'])
    model.phi_intras.load_state_dict(state_dict['phi_intras'])
    model.mus.load_state_dict(state_dict['mus'])
    model.mu_deltas.load_state_dict(state_dict['mu_deltas'])
    print('Loading model weights from {}'.format(path))