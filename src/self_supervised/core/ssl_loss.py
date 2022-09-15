'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from self_supervised.apply import config
sys.path.append(os.path.dirname(__file__))

class SimCLR_loss(nn.Module):
    
    # Based on the implementation of SupContrast
    def __init__(self, gpu, temperature):
        super(SimCLR_loss, self).__init__()
        self.gpu = gpu
        self.temperature = temperature
   
    def forward(self, features):
        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda(self.gpu)
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]
        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(self.gpu), 0)
        mask = mask * logits_mask
        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        return loss

class DDS_Suport(torch.autograd.Function):
    @staticmethod
    def forward(_tens, inp_z):
        _tens.save_for_backward(inp_z)
        output = [torch.zeros_like(inp_z) for _ in range(
            torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, inp_z)
        return tuple(output)

    @staticmethod
    def backward(_tens, *grads):
        (inp_z,) = _tens.saved_tensors
        grad_out = torch.zeros_like(inp_z)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out


class SimCLR_loss_multi_GPU(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(SimCLR_loss_multi_GPU, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.entropy_loss = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_metric = nn.CosineSimilarity(dim=2)
        self.mask = self.masks(int(batch_size/world_size), world_size)

    def masks(self, batch_size, world_size):
        # population size o samples - N
        population = 2 * batch_size * world_size
        mask = torch.ones((population, population), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):  #
            mask[i, batch_size*world_size + i] = 0  #
            mask[batch_size*world_size + i, i] = 0  #
        return mask

    def forward(self, z_i, z_j):
        current_batch_size = len(z_i)
        # population size o samples - N
        population = 2 * current_batch_size * self.world_size
        if self.world_size > 1:
            z_i = torch.cat(DDS_Suport.apply(z_i), dim=0)
            z_j = torch.cat(DDS_Suport.apply(z_j), dim=0)
            z = torch.cat((z_i, z_j), dim=0)
        else:
            z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_metric(z.unsqueeze(
            1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, current_batch_size*self.world_size)
        sim_j_i = torch.diag(sim, -current_batch_size*self.world_size)
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat(
            (sim_i_j, sim_j_i), dim=0).reshape(population, 1)
        if self.batch_size == current_batch_size*self.world_size:
            negative_samples = sim[self.mask].reshape(population, -1)
        else:
            _mask = self.masks(
                current_batch_size, self.world_size)
            negative_samples = sim[_mask].reshape(population, -1)
        labels = torch.zeros(population).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        _contrast_loss = self.entropy_loss(logits, labels)
        _contrast_loss /= population
        return _contrast_loss