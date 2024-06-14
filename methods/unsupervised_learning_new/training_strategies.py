import copy
from functools import reduce
import logging
import math
from operator import mul

import clip
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision import transforms
import re
import copy
from utils.loss import PLL_loss

from dassl.metrics import compute_accuracy
from dassl.utils import (
    MetricMeter, AverageMeter, 
    load_checkpoint,
)

accelerator = Accelerator()

from models import (
    CustomImageEncoder, 
    CustomTextEncoder, 
    ImagePrefixModel,
    TextPrefixModel,
    UPTModel,
)
from utils import (
    make_scheduler, 
    seed_worker, 
    save_parameters,
    save_pseudo_labels,
    save_pseudo_labels_torch,
)


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class TrainingStrategy(object):
    def __init__(
        self, 
        config, 
        label_to_idx, 
        classes, 
        seen_classes, 
        unseen_classes, 
        device
    ):
        """ This class defines functions for the training strategies.

        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """

        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx

        self.device = device
        self.clip_model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )
        self.transform_train = self.modify_transform(self.transform)
        self.template = self.config.PROMPT_TEMPLATE

    def modify_transform(self, transform):
        """
        Modify an existing transform.
        
        Parameters:
        transform (torchvision.transforms.Compose): The existing transform
    
        Returns:
        torchvision.transforms.Compose: The modified transform
        """
        # Get the normalization transform from the existing transform
        normalize = [t for t in transform.transforms if isinstance(t, transforms.Normalize)][0]
        # Get the Resize transform from the existing transform
        resize_transform = [t for t in transform.transforms if isinstance(t, transforms.CenterCrop)][0]
        # Parse the size from the Resize transform's print information
        size_info = re.search(r'size=\((\d+), (\d+)\)', str(resize_transform))
        H, W = map(int, size_info.groups())

        # Build the new transform
        transform_new = transforms.Compose([
            transforms.RandomResizedCrop(size=(H, W), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize  # Use the same normalization as the existing transform
        ])
        
        return transform_new


    def declare_custom_encoder(self):
        """ This function declares the custom encoder
        needed depending on the prompt modality.

        :param modality: either text or image
        """

        if self.config.MODALITY == 'image':
            self.visual_transformer = self.clip_model.visual
            self.image_encoder = CustomImageEncoder(self.visual_transformer
            ).to(self.device)
            log.info(f"Freeze visual encoder.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        elif self.config.MODALITY == 'text':
            if torch.cuda.is_available():
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.float16
                ).to(self.device)  
            else:
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.half
                ).to(self.device)

            log.info(f"Freeze text encoder.")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        elif self.config.MODALITY == 'multi':
            self.visual_transformer = self.clip_model.visual
            self.image_encoder = CustomImageEncoder(self.visual_transformer).to(self.device)
            if torch.cuda.is_available():
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.float16
                ).to(self.device)
            else:
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.half
                ).to(self.device)

            log.info(f"Freeze visual encoder.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            log.info(f"Freeze text encoder.")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def initialize_prompts_parameters(self):
        """ This function initialized the prompt parameters
        depending on the prompt modality.

        :param modality: either text or image
        """

        if self.config.MODALITY == 'image':
            width = self.visual_transformer.class_embedding.size()[0]
            scale = width**-0.5
            if self.config.VIS_PREFIX_INIT == "normal":
                vis_initial_prefix = scale * torch.randn(self.config.PREFIX_SIZE, width)

            elif self.config.VIS_PREFIX_INIT == "uniform":
                val = math.sqrt(6.0 / float(3 * reduce(mul, (16, 16), 1) + width))  # noqa
                vis_initial_prefix = torch.zeros(self.config.PREFIX_SIZE, width)
                vis_initial_prefix = scale * nn.init.uniform_(vis_initial_prefix, -val, val)

            self.vis_initial_prefix = vis_initial_prefix

        elif self.config.MODALITY == 'text':
            # Prefix initialization
            prefix_dim = (
                1,
                self.config.PREFIX_SIZE,
                self.clip_model.token_embedding.embedding_dim,
            )
            self.initial_prefix = torch.normal(         #TODO set prompt init
                self.config.MEAN_INIT, self.config.VAR_INIT, size=prefix_dim
            ).to(self.device)

        elif self.config.MODALITY == 'multi':
            # Get relevant dimensions
            vpt_dim = self.clip_model.visual.conv1.weight.shape[0]
            coop_dim = self.clip_model.ln_final.weight.shape[0]

            # Initialize the coop prompt
            self.coop_embeddings = torch.empty(
                1, 
                self.config.TEXT_PREFIX_SIZE, 
                coop_dim,
                dtype=self.dtype).to(self.device)
            nn.init.normal_(self.coop_embeddings, std=0.02)

            # Initialize the vpt prompt
            clip_patchsize = self.clip_model.visual.conv1.weight.shape[-1]
            clip_patchsize = _pair(clip_patchsize)
            val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))  # noqa

            self.vpt_embeddings = torch.zeros(
                1, 
                self.config.VISION_PREFIX_SIZE, 
                vpt_dim, 
                dtype=self.dtype).to(self.device)
            # xavier_uniform initialization
            nn.init.uniform_(self.vpt_embeddings.data, -val, val)

            if self.config.VPT_DEEP:
                self.vision_layers = len([k for k in self.clip_model.state_dict().keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

                self.vpt_embeddings_deep = torch.zeros(
                        self.vision_layers-1, 
                        self.config.VISION_PREFIX_SIZE, 
                        vpt_dim, 
                        dtype=self.dtype).to(self.device)
                # xavier_uniform initialization
                nn.init.uniform_(self.vpt_embeddings_deep.data, -val, val)
            else:
                self.vpt_embeddings_deep = None


    def define_model(self, classes=None):
        """ This function initialized the model
        depending on the prompt modality.

        :param modality: either text or image
        :param classes: the list of classes for textual model
        """

        if self.config.MODALITY == 'image':
            # Define model
            self.model = ImagePrefixModel(
                copy.deepcopy(self.vis_initial_prefix),
                self.image_encoder,
                device=self.device,
            ).to(self.device)

        elif self.config.MODALITY == 'text':
            # Define model
            self.model = TextPrefixModel(
                copy.deepcopy(self.initial_prefix),    #torch.Size([1, 16, 512])
                self.text_encoder,
                [" ".join(c.split("_")) for c in classes],
                device=self.device, 
            ).to(self.device)

        elif self.config.MODALITY == 'multi':

            # Define model
            self.model = UPTModel(
                self.coop_embeddings,
                self.vpt_embeddings,
                self.vpt_embeddings_deep,
                self.image_encoder,
                self.text_encoder,
                self.classes,
                self.config.TRANSFORMER_DIM, 
                device=self.device,
                dtype=self.dtype
            ).to(self.device)

        for i, parameter in enumerate(self.model.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

        if self.config.OPTIM == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.LR,
                weight_decay=self.config.DECAY,
                momentum=0.9,
            )

        self.scheduler = make_scheduler(self.optimizer, self.config)
    
    def build_loss(self, cfg, partialY):
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


    def create_training_dataset(self, train_data, unlabeled_data=None):
        """This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        """
        self.val_unseen_files = None
        return train_data


    def _before_train(self, train_data, val_data=None, train_transform=None, val_transform=None):
        # Declare the data pre processing for train and validation data
        train_data.transform = train_transform
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=8,
            drop_last=True,
            pin_memory=(torch.cuda.is_available()),
        )
        if val_data is not None:
            val_data.transform = val_transform
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=8,
                drop_last=False,
            )
        else:
            val_loader = None
        
        accelerator.wait_for_everyone()

        self.model, self.optimizer, train_loader, val_loader = accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader)
        if val_loader is not None:
            log.info(f"Size of validation dataset: {len(val_data.filepaths)}")
        
        return train_loader, val_loader

    def train(
        self,
        train_data,
        unlabeled_data,
        val_data=None,
        only_unlabelled=False,
        only_seen=False,
        iter_num=None,
    ):
        """This function defines the current training iteration of self.model.

        Args:
            train_data (CustomDataset): The labeled training dataset.
            unlabeled_data (CustomDataset): The unlabeled dataset.
            val_data (CustomDataset, optional): The validation dataset. Default is None.
            only_unlabelled (bool, optional): If True, train only with unlabeled data. Default is False.
            only_seen (bool, optional): If True, train only with seen classes. Default is False.
            iter_num (int, optional): The current iteration number. Default is None.
        """
        # 1. Define training dataset, model and loss
        log.info(f"Current train data (all num can be used) is: {len(train_data.filepaths)}.")
        log.info(f"And only with unlabeled is: {len(unlabeled_data.filepaths)}.")
        log.info(f"We select max {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")

        if self.config.LEARNING_PARADIGM == 'trzsl':
            current_pl_classes = self.unseen_classes
            log.info(f"[TRZSL] The number of unseen classes/all classes is: "
                     f"{len(self.unseen_classes)}/{len(self.classes)}.")
        else:
            current_pl_classes = self.classes
        log.info(f"Thus we expect an max number of pseudolabeles equal to "
                 f"{len(current_pl_classes) * self.config.N_PSEUDOSHOTS}.")

        log.info(f"Start generating train_dataset..")

        if (hasattr(self.config, 'PartialY_CFG') and iter_num == 1 and 
            isinstance(self.config.PartialY_CFG.REGULAR_THRESHOLD, str)
        ):
            # Deal with the case where REGULAR_THRESHOLD is formatted like "auto*1.5" 
            mul = eval(self.config.PartialY_CFG.REGULAR_THRESHOLD.split('*')[1])
            self.config.PartialY_CFG.REGULAR_THRESHOLD = 1 - (1 / len(current_pl_classes))*mul

        # Create the current training dataset from unlabeled_data
        train_data, partialY = self.create_training_dataset(
            train_data, unlabeled_data, 
            iter_num=iter_num
        )
        log.info(f"After replaced by pseudolabels, The train_data has size: {len(train_data.filepaths)}.")
        log.info(f"The unlabeled_data has size: {len(unlabeled_data.filepaths)}.")

        # Initialize the model
        log.info(f"Model Initialization..")
        if self.config.MODALITY == 'text':
            self.define_model(self.classes)
        else:
            self.define_model()
        # Initialize the loss function
        log.info(f"Building loss function..")
        self.loss_func = self.build_loss(self.config.LOSS_CFG, partialY)

        #2. prepare train loader
        train_loader, val_loader = self._before_train(
            train_data, val_data, 
            train_transform=self.transform, 
            val_transform=self.transform
        )
        best_val_accuracy = 0; loss = None

        # 3. start training:
        for epoch in range(self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER

            loss, total_loss, epoch_parameters = self._train_epoch(
                loss,
                total_loss,
                train_loader,
                accum_iter,
                epoch,
                only_unlabelled=only_unlabelled,
                only_seen=only_seen,
            )
            accelerator.wait_for_everyone()
            self._after_epoch(                
                train_data,
                epoch,)

            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")

            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, only_unlabelled)
                log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters

            if self.config.MODALITY == 'text':
                # After validation on seen classes redefine the set of training classes
                self.model.classes = self.classes

        return best_val_accuracy, epoch_parameters


    @torch.no_grad()
    def _after_epoch(self, train_data, epoch):
        if not hasattr(self.loss_func, 'losstype') or '_' not in self.loss_func.losstype:
            """the loss_func do not need post-epoch processing (update conf)"""
            return

        elif epoch >= 0:
            train_loader, val_loader = self._before_train(train_data, val_data=None, 
                                                          train_transform=self.transform)

            acc_cum = AverageMeter()
            forward_method = self.get_clip_forward(target_class=self.classes)
            for i, (img, aug_1, idxs, label, img_path) in enumerate(train_loader):
                gt_label = self._get_gt_label(img_path, dtype=label.dtype)

                logits = forward_method(img)
                self.loss_func.check_conf_update(img, label, idxs, output=logits)   

                acc_cum.update(compute_accuracy(logits, gt_label)[0].item())
                if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                    log.info(
                        f"EVAL on epoch [{epoch}/{self.config.EPOCHS}] [{(i + 1)}/{len(train_loader)}]\t" 
                        f"acc {acc_cum.val:.3f} ({acc_cum.avg:.3f})\t"
                    )

            self.loss_func.clean_conf()
                

    def fixed_iterative_train(self,
        train_data,
        val_data,
        unlabeled_data,
        only_seen=False,
    ):
        # Number of total iterations to cover all unlabeled data
        num_iter = int(100/self.config.STEP_QUANTILE)
        num_samples = int(len(unlabeled_data) / num_iter)
        # Initialize the number of pseudo-labels per class
        n_per_class = int(num_samples / len(self.classes))
        n_unseen = len(self.classes)

        log.info(f"We select {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")
        log.info(f"The number of unseen classes is: {len(self.classes)}.")
        log.info(f"Thus we expect an initial number of pseudo labeles equal to {len(self.classes) * self.config.N_PSEUDOSHOTS}.")

        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        # log.info(f"Training data labels: {original_train_data.labels}")
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        # Original val
        original_val_data = copy.deepcopy(val_data)

        # 1. Get training data (also get pseudo-labeles from CLIP)
        self.create_training_dataset(train_data, unlabeled_data)
        log.info(f"The original train data has size: {len(original_train_data.filepaths)}.")
        log.info(f"Only with unlabeled is: {len(unlabeled_data.filepaths)}.")
        log.info(f"Current train data is: {len(train_data.filepaths)}.")

        # Save pseudolabels
        log.info(f"Saving pseudo-labels for init")
        # save_pseudo_labels(
        #     unlabeled_data.filepaths, 
        #     unlabeled_data.labels, 
        #     self.config, 
        #     iteration=0,
        # )
        log.info(f"Unlabeled is: {len(unlabeled_data.filepaths)}.")

        for niter in range(1, num_iter + 1):
            log.info(f"Start {niter} round of training..")

            train_data.filepaths = [
                f for i, f in enumerate(original_train_data.filepaths)
            ]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels)]
           
            log.info(f"Unlabeled is {len(unlabeled_data.filepaths)} at iter: {niter}.")
            self.update_training_set(train_data, unlabeled_data)
            log.info(f"Train data is {len(train_data.filepaths)} at iter: {niter}.")

            # 2. Define model
            if self.config.MODALITY == 'text':
                self.define_model(self.classes)
            else:
                self.define_model()

            log.info(f"[MODEL] Initialization iter {niter}")

            # 3. Train model
            log.info(f"[MODEL] Start model training iter {niter}..")
            t_best_val_accuracy, t_best_prompt = self.train(
                train_data, val_data, only_seen=only_seen, iterative=True,
            )
            log.info(f"[MODEL] Training completed iter {niter}.")

            log.info(f"[MODEL] Collecting model pseudo-labels on unlabeled data..")
            unlabeled_data = self.get_pseudo_labels(
                original_unlabeled_data
            )

             # Save pseudolabels
            log.info(f"Saving pseudo-labels for iteration {niter}")
            # save_pseudo_labels(
            #     unlabeled_data.filepaths, 
            #     unlabeled_data.labels, 
            #     self.config, 
            #     iteration=niter,
            # )

            save_parameters(
                t_best_prompt,
                self.config, 
                iteration=niter
            )

            val_data = original_val_data
            original_val_data = copy.deepcopy(val_data)
        
        return t_best_val_accuracy, t_best_prompt

    def grip_train(
        self,
        train_data,
        val_data=None,
        unlabeled_data=None,
        test_data=None,
        only_seen=False,
    ):
        """
        scenario: fine-tuning with full unlabeled data (using iterations)
        """
        assert train_data is not None
        # Number of total iterations to cover all unlabeled data
        num_iter = int(100/self.config.STEP_QUANTILE)
        num_samples = int(len(unlabeled_data) / num_iter)
        # Initialize the number of pseudo-labels per class
        if self.config.LEARNING_PARADIGM == 'trzsl':
            current_pl_classes = self.unseen_classes
        else:
            current_pl_classes = self.classes

        # update initial N_PSEUDOSHOTS based STEP_QUANTILE
        n_per_class = int(num_samples / len(current_pl_classes))
        n_unseen = len(current_pl_classes)
        if n_per_class * n_unseen <= len(unlabeled_data.filepaths):
            self.config.N_PSEUDOSHOTS = n_per_class
        else:
            self.config.N_PSEUDOSHOTS = math.floor(
                len(unlabeled_data.filepaths) / n_unseen)

        log.info(f"We select max {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")
        log.info(f"The number of unseen classes is: {len(current_pl_classes)}.")
        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data.filepaths)
        original_unlabeled_data = copy.deepcopy(unlabeled_data.filepaths)

        # Start iterations of pseudolabels' updates
        for niter in range(1, num_iter + 1):
            log.info(f"Start {niter} round of training..")

            log.info(f"[TEACHER] Start model training..")
            t_best_val_accuracy, t_best_prompt = self.train(
                train_data=train_data, 
                unlabeled_data=unlabeled_data,
                val_data=None, 
                only_seen=only_seen,
                iter_num=niter,
            )
            log.info(f"[TEACHER] Training completed.")

            # Increase the number of max pseudolabels (N_PSEUDOSHOTS) after an iter
            n_per_class = int((niter + 1) * num_samples / n_unseen)
            if n_per_class * n_unseen <= len(original_unlabeled_data):
                self.config.N_PSEUDOSHOTS = n_per_class
            else:
                self.config.N_PSEUDOSHOTS = math.floor(
                    len(original_unlabeled_data) / n_unseen
                )

            acc_test = self.test_predictions(test_data)
            log.info(f"=======> Test accuracy after {niter} iteration: {acc_test}")
            save_parameters(t_best_prompt, self.config, iteration=niter)

            # Reset the training data to the original one
            train_data.update_xy(filepaths=original_train_data, labels=None)
            unlabeled_data.update_xy(filepaths=original_unlabeled_data, labels=None)

        return t_best_val_accuracy, t_best_prompt


    def define_loss_function(self, logits, labs):
        """return the loss function value."""
        return self.loss_func(logits, labs)

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            log.info(f"Loss is infinite or NaN!")
            raise FloatingPointError("Loss is infinite or NaN!")

    def backpropagate(self):
        self.optimizer.step()
        self.model.zero_grad()

    def update_scheduler(self):
        current_lr = self.scheduler.get_last_lr()
        self.scheduler.step()

    def unwrap_model(self):
        return accelerator.unwrap_model(self.model)



