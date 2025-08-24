# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from collections import defaultdict
from models.models import AdversarialNetwork
import torch.nn as nn
from models.myloss import Adversarial_loss, InfoMax_loss
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, auc, precision_recall_curve
from itertools import cycle
from config import config
from lib.utility import classification_metric, edge_extract

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)

def multi_eval_epoch(model, loader, node_x, edge_index, drug, device):
    model.eval()
    total_loss = 0
    alpha = 0
    y_true, y_pred, y_mask = [], [], []
    auc_list, aupr_list, acc_list, f1_list = [], [], [], []
    for x, y, mask, _ in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        mask = (mask > 0)
        with torch.no_grad():
            _, yp, _ = model(x, alpha, node_x, edge_index)
            loss_mat = nn.BCEWithLogitsLoss()(yp, y.double())
            loss_mat = torch.where(
                mask, loss_mat,
                torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(mask)
            total_loss += loss
            y_true += y.cpu().detach().numpy().tolist()
            y_pred += yp.cpu().detach().numpy().tolist()
            y_mask += mask.cpu().detach().numpy().tolist()
              
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mask = np.array(y_mask)

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1.0) > 0 and np.sum(y_true[:, i] == 0.0) > 0:
            is_valid = (y_mask[:, i] > 0)
            auc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
            aupr_list.append(auprc(y_true[is_valid, i], y_pred[is_valid, i]))
            f1_list.append(f1_score(y_true[is_valid, i], (y_pred[is_valid, i]>=0.5).astype('int')))
            acc_list.append(accuracy_score(y_true[is_valid, i], (y_pred[is_valid, i]>=0.5).astype('int')))
        else:
            print('{} is invalid'.format(i))
    
    all_results=[auc_list, aupr_list, acc_list, f1_list]
    
    return total_loss/len(loader), np.array(all_results), y_true, y_pred, y_mask


def training(encoder, classifier, s_dataloader, t_dataloader, drug, prototype, params_str, tissue_weights=None, **kwargs):
    da_network = AdversarialNetwork(encoder, classifier, len(drug)).to(kwargs['device'])
    optimizer = torch.optim.AdamW(da_network.parameters(), lr=kwargs['lr'])
    LossFunc_domain = nn.BCEWithLogitsLoss(reduction='mean')
    LossFunc_class = nn.BCEWithLogitsLoss(reduction='none')
    best_loss_sum = np.inf

    if tissue_weights is not None:
        print("Received tissue_weights in finetuning.py:", tissue_weights)

    node_x = torch.from_numpy(config.drug_feat.astype('float32'))
    node_x = node_x.to(kwargs['device'])
    edge_index = edge_extract(config.label_graph)
    edge_index = torch.from_numpy(edge_index.astype('int'))
    edge_index = edge_index.to(kwargs['device'])


    all_types = set()
    for batch in s_dataloader:
        all_types.update(batch[3].cpu().numpy())
    for batch in t_dataloader:
        all_types.update(batch[3].cpu().numpy())
    num_cancer_types = len(all_types)
    print(f"Detected {num_cancer_types} cancer types, initializing similarity learning module")


    contrastive_loss_fn = InfoMax_loss(
        hidden_dim=da_network.encoder.output_layer[0].in_features,
        num_cancer_types=num_cancer_types,
        tissue_weights=tissue_weights
    ).to(kwargs['device'])

    print("\n============Domain adaptation for TCGA data================")
    for epoch in range(int(kwargs['uda_num_epochs'])):
        return_loss_sum = 0
        len_loader = min(len(s_dataloader), len(t_dataloader))
        da_network.train(True)
        optimizer.zero_grad()
        # Beginning domain training
        for i, batch in enumerate(zip(t_dataloader, cycle(s_dataloader))):
            # Domain adapation loss
            p = float(i + epoch * len_loader) / int(kwargs['uda_num_epochs']) / len_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            s_x = batch[1][0].to(kwargs['device'])
            s_y = batch[1][1].to(kwargs['device'])
            s_mask = batch[1][2].to(kwargs['device'])
            s_type = batch[1][3].to(kwargs['device'])
            t_x = batch[0][0].to(kwargs['device'])
            t_type = batch[0][3].to(kwargs['device'])
            
            domain_s, s_yp, s_feat = da_network(s_x, alpha, node_x, edge_index)
            domain_t, t_yp, t_feat = da_network(t_x, alpha, node_x, edge_index)
            
            # Global alignment loss: pure domain-level distribution alignment
            transfer_loss = Adversarial_loss(
                domain_s=domain_s, 
                domain_t=domain_t, 
                BCEloss=LossFunc_domain
            )
            
            # Multi-label (drug) classification loss
            loss_mat = LossFunc_class(s_yp, s_y.double())
            loss_mat = torch.where(s_mask>0, loss_mat,
                                   torch.zeros(loss_mat.shape).to(kwargs['device']))
            classifier_loss = torch.sum(loss_mat) / torch.sum(s_mask)
            
            # Weighted contrastive loss function
            contrastive_loss = contrastive_loss_fn(
                s_feat=s_feat, 
                t_feat=t_feat, 
                s_type=s_type, 
                t_type=t_type, 
                prototype=prototype
            )
            

            total_loss = kwargs['alph']*transfer_loss + kwargs['beta']*contrastive_loss + \
                         (1-kwargs['alph']-kwargs['beta'])*classifier_loss
            total_loss.backward()
            optimizer.step()
            return_loss_sum += total_loss.cpu().detach().item()
        
        if (best_loss_sum > return_loss_sum):
            best_loss_sum = return_loss_sum
            torch.save(da_network.state_dict(), os.path.join(kwargs['model_save_folder'], 'AdversarialNetwork.pt')) 
            
        if (epoch+1) % 50 == 0:
            print('Domain adapation training epoch = {}'.format(epoch+1))
            print('Domain adapation loss : {:.4f}'.format(return_loss_sum/len_loader))
            
            # Print current learned cancer similarity matrix
            print('\n================Current Cancer Similarity Matrix================')  
            similarity_matrix = contrastive_loss_fn.cancer_similarity()
            print('Similarity matrix shape:', similarity_matrix.shape)
            print('Similarity matrix content:')
            print(similarity_matrix.detach().cpu().numpy())


    
    da_network.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'AdversarialNetwork.pt')))   
    
    # Save final similarity matrix after training
    final_similarity_matrix = contrastive_loss_fn.cancer_similarity()
    similarity_save_path = os.path.join(kwargs['model_save_folder'], f'cancer_similarity_matrix_{params_str}.npy')
    np.save(similarity_save_path, final_similarity_matrix.detach().cpu().numpy())
    print(f'\nFinal similarity matrix saved to: {similarity_save_path}')
    print('Final similarity matrix shape:', final_similarity_matrix.shape)
    print('Final similarity matrix content:')
    print(final_similarity_matrix.detach().cpu().numpy())
    
    return da_network, final_similarity_matrix


def testing(model, t_dataloader, drug, device):
    model.train(False)  
    node_x = torch.from_numpy(config.drug_feat.astype('float32'))
    node_x = node_x.to(device)  
    edge_index = edge_extract(config.label_graph)
    edge_index = torch.from_numpy(edge_index.astype('int'))
    edge_index = edge_index.to(device)
    test_loss, results, y_true, y_pred, y_mask = multi_eval_epoch(model=model,
                                          loader=t_dataloader, 
                                          node_x=node_x,
                                          edge_index=edge_index,
                                          drug=drug,
                                          device=device) 

    return test_loss, results, y_true, y_pred, y_mask