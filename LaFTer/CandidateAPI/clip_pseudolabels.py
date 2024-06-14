import logging
import os
import pickle

import clip
import torch
from PIL import Image
from tqdm import tqdm
import copy
from CandidateAPI.utils import find_elem_idx_BinA, PoolsAggregation
import torch.nn.functional as F
import numpy as np
from collections import Counter



def detect_candidate_bycum(data, thr):
    """
    Generates a candidate mask by cumulatively summing the data to a threshold and getting the summed items of each row.

    Args:
        data (Tensor or any): The data to be summed. If not a Tensor, it will be converted to a Tensor.
        thr (float): The threshold for the cumulative sum.

    Returns:
        Tensor: The candidate mask.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    sorted_data, indices = torch.sort(data, dim=-1, descending=True)
    cum_data = torch.cumsum(sorted_data, dim=-1)
    assert isinstance(thr, float), "thr should be a float number"

    # Create a mask for the first element of each row
    first_elem_mask = torch.zeros_like(cum_data, dtype=torch.bool)
    first_elem_mask[:, 0] = True

    # Create last_elem_mask
    exceeds_thr_mask = cum_data > thr
    shifted_exceeds_thr_mask = torch.cat([torch.zeros_like(exceeds_thr_mask[:, :1]), 
                                          exceeds_thr_mask[:, :-1]], 
                                        dim=1)
    last_elem_mask = (~shifted_exceeds_thr_mask) & exceeds_thr_mask

    # Combine masks
    candidate_mask = cum_data <= thr
    candidate_mask |= last_elem_mask
    candidate_mask |= (exceeds_thr_mask & first_elem_mask)

    # Create a new tensor and use indices and row indices to get elements from candidate_mask
    rows = torch.arange(candidate_mask.size(0)).unsqueeze(-1)
    candidate_mask = candidate_mask[rows, indices.sort(dim=-1).indices]   # sort idxs in ascending order

    return candidate_mask



def cdf_at_value(data, value, mode='count', batch_size=512):
    """
    Calculate the cumulative distribution function (CDF) at a given value for a histogram.

    Parameters:
    data (torch.Tensor or np.ndarray): The input data defining the custom PDF.
    value (float or np.ndarray): The value(s) at which to calculate the CDF.

    Returns:
    float or np.ndarray: The CDF at the given value(s).
    """
    # Expand dimensions for broadcasting
    if mode == 'count':
        scores = torch.zeros_like(value)
        for i in range(0, len(value), batch_size):
            data_ = data.unsqueeze(0)  # shape becomes (1, n)
            value_ = value[i:i+batch_size].unsqueeze(1)  # shape becomes (m, 1)
            # Count the number of elements in 'data' that are smaller than each element in 'value'
            # The result is a tensor of shape (m, n), where each row corresponds to an element in 'value'
            counts = (data_ <= value_).sum(dim=1)
            scores[i:i+batch_size] = counts / data.shape[0]

    return scores


def detect_candidate_bycls_thr(data, thr):
    """
    Generates a candidate mask by class-wise inter-instance percentage.

    Args:
        data (Tensor or any): The data to be processed. If not a Tensor, it will be converted to a Tensor.
        thr (float): The threshold for the CDF scores.

    Returns:
        Tensor: The candidate mask.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    candidate_mask = torch.zeros_like(data, dtype=torch.bool)
    range_list = torch.arange(data.shape[0])

    for cls in range(data.shape[1]):
        cls_output = data[range_list, cls]

        thr_val = torch.quantile(cls_output, thr)
        cls_mask = (cls_output >= thr_val)

        # assert (cls_mask_ == cls_mask).all(), "The two masks should be the same"
        candidate_mask[range_list, cls] = cls_mask

    return candidate_mask


def get_partialY_byThr(threshold, candidate_method, probs_list, logits):
    """
    Generates partial labels by a given threshold and candidate method.

    Args:
        threshold (float): The threshold to detect candidates.
        candidate_method (str): The method to detect candidates. Should be either 'intra_inst' or 'inter_inst'.
        probs_list (Tensor): The probabilities.
        logits (Tensor): The output logits.

    Returns:
        Tensor: The partial labels.
    """
    if candidate_method == 'intra_inst':
        # method 1: use cumulated prob to detect candidate:
        data = probs_list
        candidate_mask = detect_candidate_bycum(data=data, thr=threshold)     #prob is shape of (batch_size, num_classes)
    elif candidate_method == 'inter_inst':
        # method 2: use cdf as class-wise inter_inst percentage to detect candidate:
        data = probs_list
        candidate_mask = detect_candidate_bycls_thr(data=probs_list, thr=threshold)
    else:
        raise ValueError("candidate_method should be 'intra_inst' or 'intra_inst'")
    # pred_id = labels_range[candidate_mask]
    PL_label = candidate_mask.float() * data

    # normalize the PL_label according to the PLL requirements
    base_value = PL_label.sum(dim=1).unsqueeze(1).repeat(1, PL_label.shape[1])
    PL_label_ = PL_label / base_value
    return PL_label_


def compute_logits(
    dataloader,
    clip_model,
    forward_method,
    T,
):
    print(f"Compute pseudo-labeles on all unlabeled data")
    probs_list = []
    new_imgs = []
    logits = []
    clip_model.eval()

    for img, aug_1, idxs, label, img_path in tqdm(dataloader):
        with torch.no_grad():
            logits_per_image = forward_method(img)
            probs = F.softmax(logits_per_image / T, dim=-1)
        
        probs_list.append(probs)
        new_imgs.extend(list(img_path))
        logits.append(logits_per_image)

    probs_list = torch.cat(probs_list, dim=0).float()
    logits = torch.cat(logits, dim=0).float()

    return new_imgs, probs_list, logits

def compute_unlabled_logits(    
    dataset,
    transform,
    clip_model,
    tag="PL",
    forward_method=None,
):
    #create a new dataloader:
    dataset.transform = transform
    dataset.label_id = True
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True if torch.cuda.is_available() else False,
    )
    filepaths, probs, output_logits = compute_logits(        
        dataloader=train_loader,
        clip_model=clip_model,
        forward_method=forward_method,
        T=1.0,
    )
    return filepaths, probs, output_logits


def gererate_partialY(config, probs, output_logits, info=None):
    """
    This function generates partial labels for a given configuration, probabilities, and output logits.

    Args:
        config: The configuration parameters.
        probs (Tensor): The probabilities.
        output_logits (Tensor): The output logits.
        info (optional): Additional information.

    Returns:
        Tensor, Tensor: The partial labels and a mask indicating non-zero rows.
    """
    if config.CANDIDATE_METHOD == 'CPL':
        assert config.CONF_THRESHOLD == 'quantile'
        conf_thr_final_1, PL_labels_1 = get_partialY_byAutoThr(
            output_logits, probs, 
            config.CONF_THRESHOLD, 
            'intra_inst',
            target_quantile=config.CONF_QUANTILE,
        )
        PL_labels_2 = get_partialY_byThr(
            logits=output_logits, 
            probs_list=probs, 
            threshold=config.REGULAR_THRESHOLD, 
            candidate_method='inter_inst'
        )
        # Combine the two sets of partial labels through intersection
        candidate_mask = ((PL_labels_1 > 1e-7) & (PL_labels_2 > 1e-7)).float()
        
        # If the sum of candidates in a row is 0, tag this row as a zero row
        zero_row_mask = (candidate_mask.sum(dim=1) == 0)

        # Normalize the PL_label according to the PLL requirements
        PL_label = candidate_mask * probs
        base_value = PL_label.sum(dim=1).unsqueeze(1).repeat(1, PL_label.shape[1])
        PL_labels = PL_label / base_value
        conf_thr_final = (conf_thr_final_1, config.REGULAR_THRESHOLD) 

    else:
        raise ValueError("config.CANDIDATE_METHOD should be 'CPL'")
            
    print(f"So, selected CONF_THRESHOLD for method {config.CANDIDATE_METHOD}, is <{conf_thr_final}>")
    return PL_labels, ~zero_row_mask


def get_partialY_byAutoThr(output_logits, probs, conf_thr_method, select_method, 
                           target_quantile=None):
    """
    Adjusts the confidence threshold according to the given method and gets the partial labels.

    Args:
        output_logits (Tensor): The output logits.
        probs (Tensor): The probabilities.
        conf_thr_method (str): The method to adjust the confidence threshold.
        select_method (str): The method to select the partial labels.
        target_quantile (float, optional): The target quantile to adjust the confidence threshold. Default is None.

    Returns:
        float, Tensor: The adjusted confidence threshold and the partial labels.
    """
    if conf_thr_method == 'quantile':
        if target_quantile == 0:
            conf_thr = probs.max(dim=1).values.mean().cpu().item()
        else:
            # set the confidence threshold to the quantile of the maximum probabilities
            conf_thr = torch.quantile(probs.max(dim=1).values, target_quantile/100).cpu().item()
        PL_labels_ = get_partialY_byThr(
            threshold=conf_thr,
            candidate_method=select_method,
            probs_list=probs,
            logits=output_logits)

    else:
        raise ValueError("conf_thr_method should be 'quantile'")

    return conf_thr, PL_labels_



class InstanceSelector(object):
    # opt params (assigned during code running)

    def __init__(self, 
                 label_to_idx=None, 
                 train_labels_true=None,
                 eps=1e-7, cfg=None,
                 convert_pred_method=None,):
        super(InstanceSelector, self).__init__()
        self.cfg = cfg
        self.label_to_idx = label_to_idx
        self.eps=eps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class_ids = list(range(0, len(label_to_idx)))
        self.Pools = PoolsAggregation(class_ids=class_ids, K=self.cfg.N_PSEUDOSHOTS)
        if convert_pred_method is not None:
            self.convert_pred_idxs_to_real = convert_pred_method
            for cls_idx, pool in self.Pools.cls_pools_dict.items():
                pool.convert_pred_idxs_to_real = convert_pred_method

        if train_labels_true is not None:
            for cls_idx, pool in self.Pools.cls_pools_dict.items():
                pool.labels_true = train_labels_true


    def cal_pred_conf_uncs(self, logits, PL_labels, method):
        """
        Calculates the prediction confidence and uncertainties for each data point.
        
        And the uncs will be used in the following steps to select the top-k samples (in ascending order).
        This munipulation is equivalent to just use conf to select top-k samples (in descending order).
        """
        # method 1. cal conf by prob:
        conf = F.softmax(logits, dim=1)
        uncs = 1 - conf
    
        # assign attrs:
        conf = conf * PL_labels
        uncs = uncs * PL_labels

        return conf, uncs


    def _prepare_items_attrs(self, PL_labels, outputs, method, max_num="all"):   
        """
        Prepares labels and uncertainties for all items based on model outputs and a specified method.
        
        Args:
            PL_labels (Tensor): The pseudolabels for the data points.
            outputs (Tensor): The output from the model for all data points.
            method (str): The method used to calculate uncertainties.
            max_num (str or int, optional): The maximum number of labels to process.
        
        Returns:
            tuple: A tuple containing tensors of labels and uncertainties for the data points.
        """
        conf_all, uncs_all = self.cal_pred_conf_uncs(outputs, PL_labels, method=method)

        if max_num == "all":
            # Find the max count of non-zero labels across all items
            max_num = (PL_labels > self.eps).sum(dim=1).max().item()

        labels_ = []; uncs_ = []
        for i in range(max_num):
            max_val, max_idx = torch.max(conf_all, dim=1)   # get max val in each row
            mask = (max_val < self.eps)                     
            uncs = uncs_all[torch.arange(0, uncs_all.shape[0]), max_idx] #uncs is shape of (batch_size, max_num)
            uncs[mask] = torch.inf
            conf_all[torch.arange(0, conf_all.shape[0]), max_idx] = -torch.inf

            labels_.append(max_idx.unsqueeze(1))
            uncs_.append(uncs.unsqueeze(1))
        labels_ = torch.cat(labels_, dim=1)
        uncs_ = torch.cat(uncs_, dim=1)

        return labels_, uncs_

    def convert_pred_idxs_to_real(self, pred_idxs):
        """convert pred_idxs to real idxs"""
        # default do nothing and would be redefined (in __init__) only in TRZSL setting
        return pred_idxs

    def select_topk_for_eachcls(self, PL_labels, indexs_all, output_all, 
                                K_max, candidate_method, 
                                N_iter=1, increase_percentage=0):
        """
        Selects the top K instances for each class based on the provided criteria.
        
        Args:
            PL_labels (Tensor): The pseudolabels for the data points.
            indexs_all (Tensor): Indices of all data points.
            output_all (Tensor): The output from the model for all data points.
            K_max (int): The maximum number of instances to select for each class.
            candidate_method (str): The method used to select candidates.
            N_iter (int, optional): The current iteration number. Defaults to 1.
            increase_percentage (float, optional): The percentage to increase the pool size by.
        
        Returns:
            tuple: A tuple containing the indices of the selected instances and info.
        """
        #1. prepare necessary attrs for all items:
        labels, uncs = self._prepare_items_attrs(PL_labels, output_all, candidate_method)
        max_iters = PL_labels.sum(dim=1).long().cpu()
        #check Top-1 pred_label ACC:
        labels_top1 = self.convert_pred_idxs_to_real(labels[:, 0])
        acc = labels_top1.cpu() == self.Pools.cls_pools_dict[0].labels_true[indexs_all]
        ACC = acc.sum()/labels.shape[0]
        print(f"Top-1 pred_label ACC: {ACC}")
        
        #2. revise the pool caps:
        if increase_percentage == 0:
            past_caps = self.Pools.get_pool_caps()
            increment_nums = K_max - max(past_caps)
            assert increment_nums >= 0
            pool_caps = [min(past_caps[i] + increment_nums, K_max) 
                            for i in range(len(self.Pools))]
        else:
            raise ValueError("increase_percentage should be == 0")
        
        self.Pools.reset_all()
        self.Pools.scale_all_pools(scale_nums=pool_caps)  

        #3. fill pools with samples according to the uncs for each class:
        not_inpool_feat_idxs, notinpool_uncs = self._fill_pools(
            labels, uncs, indexs_all, 
            max_iter_num=max_iters, 
            record_popped=True,)
        
        selected_idxs = self.Pools.get_all_feat_idxs()
        # print info:
        self.Pools.print()
        self.Pools.cal_pool_ACC()

        return selected_idxs, {"top1_acc": ACC.item()}


    def _fill_pools(self, labels, uncs, feat_idxs, max_iter_num, 
                    record_popped=True, pool_idxs=None):
        """
        Fills pools with top samples for each class based on uncertainties.
        
        Args:
            labels (Tensor): The labels for the data points.
            uncs (Tensor): The uncertainties associated with each data point.
            feat_idxs (Tensor): The feature indices of the data points.
            max_iter_num (Tensor): The maximum number of iterations for each data point.
            record_popped (bool, optional): Whether to record popped elements. Defaults to True.
            pool_idxs (Tensor, optional): The pool indices. If None, they are generated.
        """
        # Initialize pool indices if not provided
        if pool_idxs is None:
            pool_idxs = torch.arange(0, len(self.label_to_idx)).to(self.device)
        if labels.shape[1] == 0:
            return 
        assert max_iter_num.all() >= 1

        # Initialize tensors for tracking pool status
        not_in_pool_init = torch.ones(labels.shape[0], dtype=torch.bool)
        del_elems_init = torch.zeros(labels.shape[0], dtype=torch.bool)
        all_idxs = torch.arange(0, labels.shape[0])
        quary_num = torch.zeros(labels.shape[0], dtype=torch.long)      
        
        def recursion(top_uncs, top_labels, not_in_pool, del_elems):
            """
            Recursively fills pools with samples, updating their status.
            
            Args:
                top_uncs (Tensor): The top uncertainties for each data point.
                top_labels (Tensor): The top labels for each data point.
                not_in_pool (Tensor): A boolean tensor indicating if a data point is not in the pool.
                del_elems (Tensor): A boolean tensor indicating if a data point should be deleted.
            """
            in_itering = (not_in_pool & ~del_elems)
            if (in_itering).sum() == 0 or (not_in_pool==False).all():
                return 
            else:
                # Select indices and corresponding uncertainties and labels for this iter
                this_loop_idxs = all_idxs[in_itering]  
                this_loop_uncs = top_uncs[this_loop_idxs, quary_num[this_loop_idxs]]
                this_loop_labels = top_labels[this_loop_idxs, quary_num[this_loop_idxs]]
                not_in_pool[:] = True
                quary_num[this_loop_idxs] += 1
                assert labels.shape[0] == (this_loop_idxs.shape[0] + 
                                           self.Pools.cal_pool_sum_num() + 
                                           del_elems.sum()), \
                        "All_samples = not_in + in_pool + not_in_but_enough_iter"
                # Fill the pool with the selected samples
                self.Pools.batch_fill_assigned_pool(
                    feat_idxs[this_loop_idxs], 
                    this_loop_uncs, 
                    this_loop_labels
                )
                # Update pool and deletion status
                inpool_idxs = self.Pools.get_all_feat_idxs()  
                # self.Pools.cal_pool_ACC()
                elem_idxs = find_elem_idx_BinA(A=feat_idxs, B=inpool_idxs)  

                not_in_pool[elem_idxs] = False
                del_elems = (quary_num[:] == max_iter_num[:]) & (not_in_pool)
                recursion(top_uncs, top_labels, not_in_pool, del_elems)
        
        # call recursion:
        recursion(uncs, labels, not_in_pool_init, del_elems_init)
        return feat_idxs[not_in_pool_init], uncs[not_in_pool_init, 0]    #get top 1 uncertainty for each sample
        


