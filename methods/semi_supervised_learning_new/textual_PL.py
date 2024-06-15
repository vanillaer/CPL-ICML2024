import logging

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn
from tqdm import tqdm

accelerator = Accelerator()

from methods.unsupervised_learning_new import TextualPrompt
#new:
from utils import (
    dataset_object,
    make_scheduler, 
    gererate_partialY, compute_unlabled_logits, TrainSampler,
    InstanceSelector,
)
import copy
from collections import defaultdict, Counter
import torch.nn.functional as F

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class TextualFPL_PL(TextualPrompt):
    def __init__(
        self,
        config,
        label_to_idx,
        data_folder,
        unlabeled_files,
        classes,
        seen_classes,
        unseen_classes,
        device,
    ):
        """This class define Coop baseline.

        :param config: dictionaries of prameters in models_config/coop_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """
        super().__init__(
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )

        self.data_folder = data_folder

        # new added：
        self.check_unlabeled = {item: i for i, item in enumerate(unlabeled_files)}
        self.Selector = InstanceSelector(label_to_idx=self.label_to_idx, cfg=self.config)
        self.clip_model.encoder_name = self.config.VIS_ENCODER


    def create_training_dataset(self, train_data, unlabeled_data, iter_num):
        """
        Create the dataset for training including pseudolabels for unseen classes.

        Args:
            train_data (Dataset): The dataset of the training seen classes.
            unlabeled_data (Dataset): The dataset of unlabeled data for unseen classes.
            iter_num (int): The iteration number.

        Raises:
            NotImplementedError: If the learning paradigm is not 'ul'.

        Returns:
            Dataset, Tensor: The updated training dataset and the selected pseudolabels.
        """
        if self.config.LEARNING_PARADIGM != 'ssl':
            raise NotImplementedError

        forward_method = self.get_clip_forward(target_class=self.classes, iter_num=iter_num)
        filepaths, probs, output_logits = compute_unlabled_logits(
            dataset=copy.deepcopy(unlabeled_data),
            transform=self.transform,
            clip_model=self.clip_model,
            forward_method=forward_method,
        )
        unlabeled_data_, PL_labels_selected, info = self._create_training_dataset(
            unlabeled_data, iter_num,
            filepaths, probs, output_logits
        )
        # 2. Merge unlabeled data and labeled data
        train_data_ = self._merge_unlabeled_dataset(train_data, unlabeled_data_, iter_num)
        PL_labels_selected_new = train_data_.labels.to(PL_labels_selected.device)

        return train_data_, PL_labels_selected_new


    def _merge_unlabeled_dataset(self, train_data, unlabeled_data, iter_num):
        """
        Merges unlabeled data with labeled data for semi-supervised learning (SSL),
        updating the training dataset with pseudo-labeled samples.

        Args:
            train_data: The original training dataset containing labeled samples.
            unlabeled_data: The dataset containing unlabeled samples that have been
                            pseudo-labeled.
            iter_num: The current iteration number of the SSL process.

        Returns:
            The updated training dataset containing both originally labeled and
            pseudo-labeled samples.
        """
        # 1. get unlabeled data and labeled data for SSL
        all_imgs = train_data.filepaths

        labeled_imgs = list(set(all_imgs) - set(self.check_unlabeled.keys()))
        labeled_lbs_true = self._get_gt_label(labeled_imgs, dtype=torch.long)
        labeled_lbs_true = torch.nn.functional.one_hot(labeled_lbs_true).float().cpu()

        unlabeded_imgs = unlabeled_data.filepaths
        unlabeled_pl_labels = unlabeled_data.labels

        self.balance_param = len(labeled_imgs) / len(unlabeded_imgs)
        log.info(f"UNLABELD(selected num): {len(unlabeded_imgs)} and LABELED(all num) {len(labeled_imgs)}, balance_param: {self.balance_param:.2f}")

        # 2. create a custom dataloader for labeled data:
        train_labeled_feats, train_labeled_lbs = train_data.update_xy(labels=labeled_lbs_true, 
                                                                    filepaths=labeled_imgs)
        bs_labeled = self.balance_param * 64    #assert bs of unlabeled_data is 64:
        if bs_labeled < 8:
            bs_labeled = 8
        if bs_labeled > 256:
            bs_labeled = 256
        log.info(f"iter <{iter_num}> bs_labeled is: {bs_labeled}")
        self.train_labeled_sampler = TrainSampler(x=train_labeled_feats, 
                                                  y=labeled_lbs_true, 
                                                  bs=int(bs_labeled),   #(len(train_labeled_feats)/10)
                                                  device=self.device,
                                                  mode="feats",)
        # 3. get unlabel dataset for SSL:
        filepaths_new = list(unlabeded_imgs)
        labels_new = unlabeled_pl_labels
        train_data.update_xy(labels=labels_new, filepaths=filepaths_new)

        return train_data


    def _create_training_dataset(self, train_data, iter_num,
                                       filepaths, probs, output_logits):
        """
        Create the dataset for training by merging pseudo-labels and labeled data.

        Args:
            train_data (Dataset): The dataset of the training seen classes.
            iter_num (int): The iteration number.
            filepaths (list): List of file paths for the data.
            probs (Tensor): Probabilities from the model.
            output_logits (Tensor): Logits from the model.

        Returns:
            Dataset, Tensor, info dict: The updated training dataset, the selected pseudolabels
        """
        selector_cfg = self.config.Selector_CFG
        partialY_cfg = self.config.PartialY_CFG

        PL_labels, mask_idxs = gererate_partialY(
            config=partialY_cfg, 
            probs=probs, 
            output_logits=output_logits,
        )

        log.info(f"Num of passed/unlabeled_data: {mask_idxs.sum()}/{len(filepaths)}")

        info_1 = self.check_partialY_acc(
            PL_labels[mask_idxs], 
            [filepaths[i] for i in range(len(filepaths)) if mask_idxs[i]==True], 
            partialY_cfg.TARGET_PARTIAL_RATIO, 
            partialY_cfg.INIT_PARTIAL_RATIO,)

        selected_idxs, info_2 = self.Selector.select_topk_for_eachcls(
            PL_labels=(PL_labels > 1e-7).float()[mask_idxs],
            output_all=output_logits[mask_idxs],
            indexs_all=torch.arange(len(filepaths))[mask_idxs],
            K_max=self.config.N_PSEUDOSHOTS,
            candidate_method=partialY_cfg.CANDIDATE_METHOD,
            N_iter=iter_num,
        )

        # Determine the final format of candidate pseudolabels
        if partialY_cfg.USE_SOFT_PARTIAL:
            PL_labels_selected = PL_labels[selected_idxs, :]    # soft candidate
        else:
            if 'grip' in self.config.MODEL and iter_num == 1:
                PL_labels_selected = PL_labels[selected_idxs, :]    # soft candidate
            elif 'grip' in self.config.MODEL and iter_num > 1:
                PL_labels_selected = (PL_labels[selected_idxs, :] > 1e-7).float()  # hard candidate
            elif 'fpl' in self.config.MODEL:
                PL_labels_selected = (PL_labels[selected_idxs, :] > 1e-7).float()   

        # Update the training dataset
        filepaths_new = [filepaths[i] for i in selected_idxs.tolist()]
        train_data.update_xy(labels=PL_labels_selected.cpu(), filepaths=filepaths_new)
        
        info_3 = self.check_partialY_acc(
            PL_labels_selected, filepaths_new, 
            partialY_cfg.TARGET_PARTIAL_RATIO, 
            partialY_cfg.INIT_PARTIAL_RATIO) 

        return train_data, PL_labels_selected, {"info_1": info_1, "info_2": info_2, "info_3": info_3}


    def check_partialY_acc(self, PL_labels, filepaths, target_partialR, init_partialR):
        # check the accuracy of pseudolabels
        gt_labels = self._get_gt_label(impath=filepaths, dtype=torch.long)

        # initialize a list to store the results
        results = []
        distribution = []
        # iterate over each row of PL_labels and the corresponding gt_labels
        for i in range(PL_labels.shape[0]):
            # get the indices where the values are 1.0 in the current row
            indices = torch.nonzero(PL_labels[i], as_tuple=True)

            # test if the corresponding gt_label is in these indices
            is_in = gt_labels[i] in indices[0]
            distribution.extend(indices[0].tolist())

            # append the result to the list
            results.append(is_in)
        
        results = torch.tensor(results)
        coverage_acc = results.sum() / results.shape[0]
        ct = Counter(distribution)
        ct = sorted(ct.items(), key=lambda x: x[0])
        partial_avgnum = (PL_labels > 1e-7).sum(dim=1).float()

        log.info(f"\t label_estimate_acc: {coverage_acc}")
        # log.info(f"coverage distribution: {ct}")
        partialR = partial_avgnum.mean().item()/PL_labels.shape[1]

        return {"label_estimate_acc": coverage_acc.item(), 
                "partial_ratio": partialR, 
                }


    def parser_batch(self, img, aug_1, idxs, label, img_path):
        #--------------for SSL----------------
        x, y = self.train_labeled_sampler.get_onebatch()
        img = torch.cat([img, x], dim=0)
        label = torch.cat([label, y], dim=0)

        return img, label
    

    def define_loss_function(self, logits, labs, idxs, paths):
        """Return the loss value for the given batch."""
        unlabeled_len = len(paths)

        loss_unseen = self.loss_func(logits[:unlabeled_len], labs[:unlabeled_len], idxs)
        loss_seen = F.cross_entropy(logits[unlabeled_len:], labs[unlabeled_len:])

        return loss_seen + self.config.LAMBDA * loss_unseen


