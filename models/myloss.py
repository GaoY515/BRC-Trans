# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from lib.utility import uniform
from config import config
from torch_geometric.nn import GATConv
device = config.CUDA_ID


class CancerSimilarityLearner(nn.Module):
    def __init__(self, num_cancer_types):
        super().__init__()

        self.similarity_matrix = nn.Parameter(torch.randn(num_cancer_types, num_cancer_types))
        with torch.no_grad():

            self.similarity_matrix.fill_diagonal_(1.0)
        

        self.gat_layer = GATConv(num_cancer_types, num_cancer_types, heads=1, concat=False)
        
    def forward(self):

        num_nodes = self.similarity_matrix.size(0)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()

        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        edge_index = edge_index.to(self.similarity_matrix.device)
        

        updated_matrix = self.gat_layer(self.similarity_matrix, edge_index)
        

        updated_matrix = (updated_matrix + updated_matrix.t()) / 2

        updated_matrix = torch.sigmoid(updated_matrix)

        updated_matrix = updated_matrix.fill_diagonal_(1.0)
        
        return updated_matrix

def Adversarial_loss(domain_s, domain_t, BCEloss, s_type=None, t_type=None, tissue_weights=None):
    # Initialize the pseudo-labels of the domain confrontation
    domain_labels = torch.cat((domain_s, domain_t), dim=0) 
    bs_s = domain_s.size(0)
    bs_t = domain_t.size(0)
    target_labels = torch.from_numpy(np.array([[0]]*bs_s + [[1]]*bs_t).astype('float32')).to(device)
    

    if tissue_weights is not None and s_type is not None and t_type is not None:

        s_loss = BCEloss(domain_s, torch.zeros_like(domain_s))
        t_loss = BCEloss(domain_t, torch.ones_like(domain_t))
        

        s_weights = torch.tensor(
            [tissue_weights.get(s_id.item(), 1.0) for s_id in s_type],
            device=domain_s.device,
            dtype=torch.float32
        )
        t_weights = torch.tensor(
            [tissue_weights.get(t_id.item(), 1.0) for t_id in t_type],
            device=domain_t.device,
            dtype=torch.float32
        )

        weighted_s_loss = (s_loss.squeeze() * s_weights).mean()
        weighted_t_loss = (t_loss.squeeze() * t_weights).mean()
        
        domain_loss = weighted_s_loss + weighted_t_loss
    else:

        domain_loss = BCEloss(domain_labels, target_labels)
    
    return domain_loss


class InfoMax_loss(nn.Module):
    def __init__(self, hidden_dim, num_cancer_types, tissue_weights=None, EPS=1e-8):
        super(InfoMax_loss,self).__init__()
        self.EPS = EPS
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.tissue_weights = tissue_weights
        self.cancer_similarity = CancerSimilarityLearner(num_cancer_types)
        self.reset_parameters()
        
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        uniform(self.hidden_dim, self.weight)
    
    def discriminate(self, z, summary, sigmoid: bool = True):
        """computes the probability scores assigned to this patch(z)-summary pair.
        Args:
            z (torch.Tensor): The latent space os samples.
            summary (torch.Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
        """
        z = torch.unsqueeze(z, dim = 1)
        summary = torch.unsqueeze(summary, dim = -1)
        value = torch.matmul(z, summary)
        value = value.squeeze()
        return torch.sigmoid(value) if sigmoid else value
    
    def process(self, data_type, p_key, p_value):
        summary = []
        for i in data_type:
            idx = torch.where(p_key == i)
            if (len(idx[0]) != 0):
                summary.append(p_value[idx[0]])
            else:
                summary.append(torch.zeros(1, self.hidden_dim).to(device))
                
        return torch.cat(summary, dim = 0)
    
    def forward(self, s_feat, t_feat, s_type, t_type, prototype, lamda=0.3):

        batch_s = next(iter(prototype[0]))
        batch_t = next(iter(prototype[1]))
        

        similarity_matrix = self.cancer_similarity()
        

        pos_summary_t = self.process(t_type, batch_s[0].to(device), batch_s[1].to(device))
        neg_summary_t = self.process(t_type, batch_t[0].to(device), batch_t[1].to(device))
        

        pos_scores = self.discriminate(t_feat, pos_summary_t, sigmoid=True)
        

        similar_pos_loss = 0.0
        batch_size = t_feat.size(0)
        

        if batch_size > 0:

            sample_indices = torch.randperm(batch_size)[:min(batch_size, 10)]
            
            for idx in sample_indices:
                i = idx.item()

                t_type_i = t_type[i].item()

                if t_type_i >= similarity_matrix.size(0):
                    continue
                    

                similar_types = torch.where(similarity_matrix[t_type_i] > 0.7)[0]
                

                similar_types = similar_types[:3]
                
                for st in similar_types:
                    if st != t_type_i:

                        mask = (t_type == st)
                        if mask.any():

                            similar_feat = t_feat[mask].mean(dim=0, keepdim=True)

                            similar_score = self.discriminate(t_feat[i:i+1], similar_feat, sigmoid=True)
                            similar_pos_loss += -torch.log(similar_score + self.EPS)
        

        if batch_size > 0:
            similar_pos_loss /= batch_size
        

        if self.tissue_weights is not None:
            t_weights = torch.tensor(
                [self.tissue_weights.get(t_id.item(), 1.0) for t_id in t_type],
                device=t_feat.device,
                dtype=torch.float32
            )
            

            pos_losses = -torch.log(pos_scores + self.EPS)
            pos_loss_t = (pos_losses * t_weights).mean()
            

            neg_scores = self.discriminate(t_feat, neg_summary_t, sigmoid=True)
            neg_losses = -torch.log(1 - neg_scores + self.EPS)
            neg_loss_t = (neg_losses * t_weights).mean()
        else:
            pos_loss_t = -torch.log(pos_scores + self.EPS).mean()
            neg_loss_t = -torch.log(1 - self.discriminate(t_feat, neg_summary_t, sigmoid=True) + self.EPS).mean()
        

        total_loss = (pos_loss_t + neg_loss_t) / 2 + lamda * similar_pos_loss
        
        return total_loss