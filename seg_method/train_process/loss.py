import torch
import torch.nn as nn
import torch.nn.functional as F

def KL_divergence(mu, logvar):
    '''
    Calculate the KL divergence between data distribution and N(0,1)
    :param mu:
    :param logvar:
    :return:
    '''
    B, C, D, H, W = mu.shape
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (C * D * H * W)

def self_contrastive_loss(x: torch.Tensor, avg_pool: bool = False):
    '''
    :param x: [B, C, D, H, W]
    :param x:
    :param avg_pool: default to be False, if True, x is reshaped as [B, 1, 1, H, W]
    :return:
    '''
    if avg_pool:
        B, C, D, H, W = x.shape
        avg_pool = nn.AdaptiveAvgPool3d((1, H, W))
        x = avg_pool(x)
        x = torch.mean(x, dim=1, keepdim=True)

    B, C, D, H, W = x.shape
    w_slice = x.view(B, C, D, H, 2, W // 2)
    w_slice = w_slice.permute(0, 4, 1, 2, 3, 5)
    w_slice = w_slice.contiguous()
    w_slice = torch.flatten(w_slice, start_dim=-4, end_dim=-1)
    loss = 0.
    for batch in range(B):
        # loss += torch.max(torch.tensor(0.), F.cosine_similarity(d_slice[batch][0].unsqueeze(dim=0), d_slice[batch][1].unsqueeze(dim=0)))
        # loss += torch.max(torch.tensor(0.), F.cosine_similarity(h_slice[batch][0].unsqueeze(dim=0), h_slice[batch][1].unsqueeze(dim=0)))
        loss += torch.max(torch.tensor(0.), F.cosine_similarity(w_slice[batch][0].unsqueeze(dim=0), w_slice[batch][1].unsqueeze(dim=0)))
    return loss / B
