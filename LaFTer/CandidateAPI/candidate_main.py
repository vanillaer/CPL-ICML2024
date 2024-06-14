import os.path as osp
from CandidateAPI.PLL_loss import PLL_loss
import torch
import copy
from dassl.utils import (
    MetricMeter, AverageMeter, 
)
from dassl.metrics import compute_accuracy

from CandidateAPI.clip_pseudolabels import (
    gererate_partialY, 
    compute_unlabled_logits,
)

from collections import Counter

class CandidateTrainer(object):

    def __init__(self, 
                cfg, 
                forward_method,
                transform_weak,
                eps=1e-7,
                idx_to_label=None,
                labels_true=None,):
        super(CandidateTrainer, self).__init__()
        self.cfg = cfg
        self.eps = eps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.forward_method = forward_method
        self.transform_test = transform_weak
        self.gt_label_dict = {}
        self.label_to_idx = {v:k for k,v in idx_to_label.items()}
        self.count = 0

    def _build_loss(self, cfg, partialY):
        if cfg.LOSS_TYPE == 'CE':
            criterion = torch.nn.CrossEntropyLoss()     
            criterion.cfg = cfg
        else:
            if cfg.HAS_CONF:
                criterion = PLL_loss(type=cfg.LOSS_TYPE, cfg=cfg,
                                     PartialY=copy.deepcopy(partialY))
            else:
                criterion = PLL_loss(type=cfg.LOSS_TYPE, cfg=cfg,
                                     PartialY=None)
        self.loss_func = criterion
        return self.loss_func

    def get_dataloader(self, train_data, transform, shuffle):
        # Declare the data pre processing for train and validation data
        train_data.transform = transform
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=64,
            shuffle=shuffle,
            num_workers=8,
            drop_last=False,
            pin_memory=(torch.cuda.is_available()),
        )
        return train_loader

    def after_epoch(self, train_data_all, train_data_current, epoch):
        """
        Performs operations after each training epoch. This includes updating the training
        dataset with new candidate pseudolabels and adjusting the loss function based on
        the epoch. 

        Args:
            train_data_all: The complete training dataset.
            train_data_current: The subset of the training dataset used in the current epoch.
            epoch: The current epoch number.

        Returns:
            A tuple containing the updated loss function and the updated training dataset
            if the epoch aligns with the update frequency. Otherwise, returns None, None.
        """
        if epoch % self.cfg.PartialY_CFG.UPDATE_FREQ == 0:
            # Prepare a new set of candidate pseudolabels and selected indices
            PL_labels, train_idxs, train_data_ = self.prepare_partialY_dataset(
                train_data_all, 
                epoch=epoch
            )
            # Update the loss function with the new candidate pseudolabels
            loss_func = self._build_loss(self.cfg.LOSS_CFG, partialY=PL_labels[train_idxs])
            return loss_func, train_data_
        else:
            if hasattr(self.loss_func, 'losstype') and '_' in self.loss_func.losstype:
                # Update the confidence of loss function
                self.prepare_candidates(train_data_current, epoch)
                self.loss_func.clean_conf()
            return None, None


    @torch.no_grad()
    def prepare_candidates(self, train_data, epoch):
        """
        Prepares candidate labels for the training data by processing it through the model
        and collecting the logits. This method supports both initial preparation and
        periodic updates based on the epoch. It could also be called when needed to update
        the confidence of some certain loss functions.

        Args:
            train_data: The dataset to prepare candidates from.
            epoch: The current epoch number, used to determine whether to update the
                   candidate labels.

        Returns:
            A tuple containing the probabilities of the model predictions and the raw logits.
        """
        shuffle = False
        train_loader = self.get_dataloader(copy.deepcopy(train_data), self.transform_test, shuffle)
        acc_cum = AverageMeter()
        logits_list = []

        for i, batch in enumerate(train_loader):
            if epoch == -1:
                # Record the ground truth labels for the initial preparation
                gt_label = self._get_gt_label(batch['impath'], batch['label'])
            else:
                # Retrieve the ground truth labels for the current batch
                gt_label = self._get_gt_label(batch['impath'])
            logits = self.forward_method(batch["img"].to(self.device))
            logits_list.append(logits)

            # Check whether to update the confidence of loss func
            if epoch != -1 and epoch % self.cfg.PartialY_CFG.UPDATE_FREQ != 0:
                self.loss_func.check_conf_update(
                    batch["img"], 
                    batch["label"].to(self.device), 
                    batch["index"], 
                    output=logits
                )
            acc_cum.update(compute_accuracy(logits, gt_label)[0].item())
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(
                    f"EVAL on epoch [{epoch}/50][{(i + 1)}/{len(train_loader)}]\t" 
                    f"acc {acc_cum.val:.3f} ({acc_cum.avg:.3f})\t"
                )

        logits_list = torch.cat(logits_list, dim=0)
        # Compute probabilities from logits using softmax with temperature scaling
        probs = torch.softmax(logits_list / self.cfg.TEMPERATURE, dim=1)
        return probs, logits_list
            

    def update_dataset(self, train_data, PL_labels, idxs):
        """
        Updates the training dataset with candidate pseudolabels for selected indices.

        Args:
            train_data: The original training dataset.
            PL_labels: Tensor containing candidate pseudolabels for the entire dataset.
            idxs: Indices of the samples selected for training based on candidate
                pseudolabels.

        Returns:
            The updated training dataset with samples having updated pseudolabels.
        """
        filepaths = []
        data_source_new = []
        idxs = idxs.tolist()
        for i, item in enumerate(train_data.data_source):
            filepaths.append(item.impath)
            if i in idxs:
                item._label = PL_labels[i].cpu()
                data_source_new.append(item)
        train_data.data_source = data_source_new
        return train_data
        

    def before_train(self, train_data):
        """
        Prepares the training data before the training process begins by adjusting the
        threshold based on the configuration and preparing the dataset with candidate labels.

        Args:
            train_data: The initial training dataset.

        Returns:
            A tuple containing the loss function configured for the training with candidate
            labels and the updated training dataset.
        """
        if hasattr(self.cfg, 'PartialY_CFG') and isinstance(self.cfg.PartialY_CFG.REGULAR_THRESHOLD, str):
            # Deal with the case where REGULAR_THRESHOLD is formatted like "auto*1.5" 
            mul = eval(self.cfg.PartialY_CFG.REGULAR_THRESHOLD.split('*')[1])
            self.cfg.PartialY_CFG.REGULAR_THRESHOLD = 1 - (1 / len(self.label_to_idx))*mul

        PL_labels, train_idxs, train_data_ = self.prepare_partialY_dataset(train_data, epoch=-1)
        loss_func = self._build_loss(self.cfg.LOSS_CFG, partialY=PL_labels[train_idxs])

        return loss_func, train_data_

    def prepare_partialY_dataset(self, train_data, epoch):
        """
        Prepares the dataset with candidate labels by selecting candidates based on the
        model's predictions and the configuration for handling candidate labels. 

        Args:
            train_data: The training dataset to be prepared.
            epoch: The current epoch number, used for logging and adjustments.

        Returns:
            A tuple containing the candidate labels, the indices of the training samples
            selected for training, and the updated training dataset.
        """
        partialY_cfg = self.cfg.PartialY_CFG
        # Prepare necessary attr based on the model's predictions
        probs, output_logits = self.prepare_candidates(train_data, epoch=epoch)

        # Generate candidate labels and selected indices
        PL_labels, train_idxs = gererate_partialY(
            config=partialY_cfg, 
            probs=probs, 
            output_logits=output_logits,
        )
        # Determine the format of candidate pseudolabels
        if not partialY_cfg.USE_SOFT_PARTIAL:
            PL_labels = (PL_labels > 1e-7).float()

        print(f"Num of passed/unlabeled_data: {train_idxs.sum()}/{PL_labels.shape[0]}")

        filepaths = [item.impath for item in train_data.data_source]
        info_1 = self._check_partialY_acc(
            PL_labels[train_idxs], 
            [filepaths[i] for i in range(len(filepaths)) if train_idxs[i]==True], 
            partialY_cfg.TARGET_PARTIAL_RATIO, 
            partialY_cfg.INIT_PARTIAL_RATIO) # check for all data (unlabeled)

        # Update the training dataset
        idxs_selected = torch.nonzero((train_idxs).float()).squeeze()
        train_data_ = self.update_dataset(train_data, PL_labels, idxs_selected)
        return PL_labels, train_idxs, train_data_


    def _check_partialY_acc(self, PL_labels, filepaths, target_partialR, init_partialR):
        # check the accuracy of pseudolabels
        gt_labels = self._get_gt_label(img_path=filepaths)

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

        print(f"\t label_estimate_acc: {coverage_acc}")
        # log.info(f"coverage distribution: {ct}")
        partialR = partial_avgnum.mean().item()/PL_labels.shape[1]

        return {"label_estimate_acc": coverage_acc.item(), 
                "partial_ratio": partialR,}


    def _get_gt_label(self, img_path=None, labels=None):
        """
        Retrieves or updates the ground truth labels for the given image paths. If labels
        are provided, they are associated with the respective image paths in a dictionary
        for future retrieval. If only image paths are provided, the method returns the
        corresponding ground truth labels from the dictionary.

        Args:
            img_path: List of image paths for which ground truth labels are retrieved or updated.
            labels: Tensor of labels to be associated with the given image paths.

        Returns:
            A tensor of ground truth labels corresponding to the provided image paths.
        """
        if (labels is not None) and (img_path is not None):
            for i, item in enumerate(img_path):
                self.gt_label_dict[item] = labels[i].item()

            return labels.to(self.device)
        elif (img_path is not None) and (labels is None):
            gt_lb_list = []
            for item in img_path:
                gt_lb_list.append(self.gt_label_dict[item])

            return torch.tensor(gt_lb_list).to(self.device)


