import logging

import clip
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator
from PIL import Image
from torch.nn import functional as F

accelerator = Accelerator()
from dassl.metrics import compute_accuracy
from dassl.utils import (
    MetricMeter, AverageMeter, 
    load_checkpoint,
)

from methods.unsupervised_learning_new import TrainingStrategy
from utils import make_scheduler, seed_worker, calculate_class_accuracy_as_dict

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class TextualPrompt(TrainingStrategy):
    def __init__(
        self,
        config,
        label_to_idx,
        classes,
        seen_classes,
        unseen_classes,
        device,
    ):
        """This class define Coop's training and evaluation.

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

        # Build dictionaries to correctly label model's predictions
        seen_to_idx = {c: idx for idx, c in enumerate(self.seen_classes)}
        self.idx_to_real = {
            seen_to_idx[c]: self.label_to_idx[c] for c in self.seen_classes #将seen_to_idx中的索引映射到实际的标签索引（self.label_to_idx)
        }
        self.real_to_idx = {
            self.label_to_idx[c]: seen_to_idx[c] for c in self.seen_classes #反向映射
        }
        
        # Load custom encoder
        self.declare_custom_encoder()
        # log.info(f"Custom Encoder: {self.image_encoder}.")
        # Initialize prompt parameters
        self.initialize_prompts_parameters()

    def _get_gt_label(self, impath, dtype):
        """
        Retrieves the ground truth labels for a given list of image paths.

        :param impath: A list of image paths for which the ground truth labels are to be retrieved.
        :param dtype: The data type to be used for the returned tensor of labels.
        :return: A tensor containing the ground truth labels for the provided image paths, 
                converted to the specified data type and moved to the model's device.
        """
        gt_label_list = []
        for ip in impath:
            gt_label = self.all_gt_label_dict[ip]
            gt_label_list.append(gt_label)
        gt_label = torch.tensor(gt_label_list, dtype=dtype).to(self.device)
        return gt_label

    def _train_epoch(
        self, 
        loss, 
        total_loss, 
        train_loader, 
        accum_iter, 
        epoch, 
        only_unlabelled=False,
        only_seen=False,
    ):
        """This function defines the training epoch of self.model.

        :param loss: float loss (average across batches)
        :param total_loss: float total loss
        :param train_loader: Dataloader object - training data defined in self.train
        :param accum_iter: number of accumulation steps minimum 1
        :param epoch: current epoch
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        :param only_seen: boolean.  It is True if the training only involves seen data
        """
        acc_cum = AverageMeter()
        loss_cum = AverageMeter()
        forward_method = self.get_clip_forward(target_class=self.classes)

        for i, (img, aug_1, idxs, label, img_path) in enumerate(train_loader):
            gt_label = self._get_gt_label(img_path, dtype=label.dtype)
            # loss, logits = self.loss_func(self.forward, img, label, idxs, reduce=True)
            img, label = self.parser_batch(img, aug_1, idxs, label, img_path)

            logits = forward_method(img)

            loss = self.define_loss_function(logits, label, idxs, img_path)

            total_loss += loss.item()
            accelerator.wait_for_everyone()

            loss = loss / accum_iter
            self.detect_anomaly(loss)
            accelerator.backward(loss)

            # Accumulate grandient
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.8)
                self.backpropagate()

            # compute accuracy:
            acc_cum.update(compute_accuracy(logits[:len(img_path)], gt_label)[0].item())
            loss_cum.update(loss.item())
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                log.info(
                    f"epoch [{epoch}/{self.config.EPOCHS}][{(i + 1)}/{len(train_loader)}]  \t" 
                    f"loss {loss_cum.val:.3f} ({loss_cum.avg:.3f})\t"
                    f"acc {acc_cum.val:.3f} ({acc_cum.avg:.3f})\t"
                )

        accelerator.wait_for_everyone() 

        # predictions_outputs = accelerator.gather(predictions)
        # labels_outputs = accelerator.gather(labels)

        self.update_scheduler()

        unwrapped_model = self.unwrap_model()
        epoch_parameters = [
            unwrapped_model.prefix.detach().cpu().numpy()
        ]
        return loss, total_loss, epoch_parameters


    def get_fixed_text_feature(self, target_class):
        # Define text queries
        prompts = self.define_textual_prompts(target_class)

        # Encode text
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features


    def get_clip_forward(self, target_class, iter_num=2, dtype=torch.float32):
        """
        This function returns the forward method for CLIP under the correct settings.
        """
        # 1. Define forward method:
        def clip_forward_image(img):
            if not hasattr(self, 'fixed_text_feats') or self.fixed_text_feats_cls != target_class:
                self.fixed_text_feats = self.get_fixed_text_feature(target_class).type(dtype) 
                self.fixed_text_feats_cls = target_class

            image_features = self.model(img.to(self.device)).type(dtype)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits:
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ self.fixed_text_feats.t()
            return logits
        
        def clip_forward_text(img):
            text_features = self.model(target_class).type(dtype)  
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if img.dim() == 4:
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(img.to(self.device))
                    image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True).type(dtype)
            elif img.dim() == 2:
                image_features = img.to(self.device).type(dtype)
            else:
                raise ValueError(f"Image dimension {img.dim()} not supported.")

            # cosine similarity as logits:
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits

        def clip_zsl_forward(img):
            prompts = [self.config.PROMPT_TEMPLATE.format(c.replace("_", " ")) for c in target_class]
            # log.info(f"clip_zsl Prompts: {prompts[0:10]}")
            text = clip.tokenize(prompts).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.encode_text(text).type(dtype)
                text_features = (text_features / text_features.norm(dim=-1, keepdim=True))

                if img.dim() == 4:
                    image_features = self.clip_model.encode_image(img.to(self.device))
                    image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True).type(dtype)
                elif img.dim() == 2:
                    image_features = img.to(self.device).type(dtype)
                else:
                    raise ValueError(f"Image dimension {img.dim()} not supported.")

            # cosine similarity as logits:
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits

        # 2. return the correct forward method:
        if iter_num == 1:
            forward_method = clip_zsl_forward
            log.info(f"Use zero-shot prompt template: {self.config.PROMPT_TEMPLATE}")
        else:
            if self.config.MODALITY == 'image':
                forward_method = clip_forward_image
            elif self.config.MODALITY == 'text':
                forward_method = clip_forward_text

        return forward_method


    @torch.no_grad()
    def test_predictions(self, data, standard_zsl=False):
        """
        Computes predictions on the test dataset and evaluates the model's performance.

        Args:
            data: A dataset object representing the test dataset.
            standard_zsl (bool): temp var to be removed

        Returns:
            The harmonic mean of seen and unseen classes' accuracies in TRZSL setting, 
            or overall accuracy in other settings.
        """
        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE,
            num_workers=8,
            drop_last=False,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.model, test_loader = accelerator.prepare(self.model, test_loader)

        # Get corresponding text features
        if self.config.MODALITY == 'image':
            text_features = self.get_fixed_text_feature(self.classes)
        elif self.config.MODALITY == 'text':
            text_features = self.model(self.model.classes)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        log.info(f"TEXT FEATURES SHAPE: {text_features.size()}")

        log.info(f"Start inference for test data")
        # This is required for distributed training
        test_files = [f for f in test_loader.dataset.filepaths]

        predictions, labels_true, logits_all = [], [], []
        forward_method = self.get_clip_forward(target_class=self.classes)

        for img, aug_1, idxs, label, img_path in test_loader:
            logits = forward_method(img)
            pred = torch.argmax(logits, dim=1)

            predictions.append(pred)
            labels_true.append(label)
            logits_all.append(logits)

        accelerator.wait_for_everyone()

        predictions = torch.cat(predictions, dim=0)
        labels_true = torch.cat(labels_true, dim=0)
        logits_all = torch.cat(logits_all, dim=0)   

        if self.config.LEARNING_PARADIGM == 'trzsl':
            # In TRZSL, calculate accuracies for seen and unseen classes separately
            unseen_idxs = [self.label_to_idx[item] for item in self.unseen_classes]
            seen_idxs = [self.label_to_idx[item] for item in self.seen_classes]
            unseen_accuracy = self.get_classes_acc(predictions, labels_true, unseen_idxs, classes_type='Unseen')
            seen_accuracy = self.get_classes_acc(predictions, labels_true, seen_idxs, classes_type='Seen')
            # Calculate the harmonic mean of seen and unseen classes' accuracies
            harmonic_mean = st.hmean([unseen_accuracy.item(), seen_accuracy.item()])
            log.info(f"Harmonic Mean: {harmonic_mean:.4f}")
            return harmonic_mean
        else:
            # In other settings, calculate overall accuracy
            # log.info(f"acc_dict: {calculate_class_accuracy_as_dict(gt_lbs=labels_true, output_logits=logits_all)}")
            overall_acc = (predictions == labels_true).sum() / predictions.shape[0]
            return overall_acc.item()


    def get_classes_acc(self, predictions, labels_true, class_idxs, classes_type='Unseen'):
        """
        Calculates the accuracy for a specified set of classes (either seen or unseen) by comparing
        the model's predictions against the true labels.

        Args:
            predictions (Tensor): The predicted labels for the dataset.
            labels_true (Tensor): The true labels for the dataset.
            class_idxs (list): A list of class indices for which to calculate accuracy.
            classes_type (str): A string indicating the type of classes ('Seen' or 'Unseen').

        Returns:
            float: The accuracy for the specified set of classes.
        """
        correct_num_sum = 0
        cls_num_sum = 0
        for i in class_idxs:
            true_mask = (labels_true == i)
            pred_cls = predictions[true_mask]
            correct_num = (pred_cls == i).sum()
            cls_num = len(pred_cls)
            correct_num_sum += correct_num; cls_num_sum += cls_num
            acc_cls = correct_num / cls_num
            # log.info(f"Unseen Class {i} - Accuracy: {acc_cls:.3f}, pred/true samples: {(pred_cls == i).sum()}/{true_mask.sum()}")
        log.info(f"{classes_type} Classes - Accuracy: {correct_num_sum / cls_num_sum:.3f}, "
                 f"pred/true samples: {correct_num_sum}/{cls_num_sum}")
        return correct_num_sum / cls_num_sum


    def load_model_eval(self):
        self.define_model(self.classes)
