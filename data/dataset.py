import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import clip
from data.extract_features import (
    prepare_image_features,
)

log = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        # Adjust filepaths
        self.train = train
        if self.train:
            filepaths_ = [f"{root}/train/{f}" for f in filepaths]
        else:
            filepaths_ = [f"{root}/test/{f}" for f in filepaths]

        self.transform = transform
        if augmentations:
            self.aug1_transform = augmentations[0]
        else:
            self.aug1_transform = None
        self.use_features_mode = False
        self.update_xy(filepaths=filepaths_, labels=labels)
        self.label_map = label_map
    
    def update_xy(self, filepaths, labels):
        """
        Update the filepaths and labels in the dataset

        Args:
            filepaths (list): List of filepaths
            labels (list): List of labels
        """
        self.labels = labels
        self.filepaths = filepaths
        if self.use_features_mode:
            idxs_updated = []
            for i in range(len(filepaths)):
                idx = self.filepaths_all.index(filepaths[i])
                idxs_updated.append(idx)
            self.features = self.features_all[idxs_updated]

        if self.labels is None:
            self.labels_placeholder = torch.zeros(len(self.filepaths)) - 1
        else:
            if not isinstance(self.labels[0], str):
                self.label_id = True
            else:
                self.label_id = False
                
        if self.use_features_mode:
            return self.features, self.labels


    @torch.no_grad()
    def prepare_features(self, model):
        """
        Prepare features for the dataset in advance to speed up training

        Args:
            model (torch.nn.Module): The model to use for feature extraction
        """
        assert self.use_features_mode == False
        model.eval()
        # prepare_text_features(clip_model, dataset=self)
        log.info(f"Preparing image features for {self.dataset_name}, {'train' if self.train else 'test'} split")
        features = prepare_image_features(model, dataset=self)

        self.features_all = features['img_features']
        self.filepaths_all = features['img_paths']
        self.features = self.features_all
        self.filepaths = self.filepaths_all 
        self.use_features_mode = True


    def __len__(self):
        # dataset size
        return len(self.filepaths)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, aug1, index, target) where target is index of the target class.
        """
        if self.use_features_mode:
            img = self.features[index]
            aug_1 = self.features[index]
        else:
            img = Image.open(self.filepaths[index]).convert("RGB")
            if self.aug1_transform is not None:
                aug_1 = self.aug1_transform(img)
            else:
                img1 = self.transform(img)
                aug_1 = img1

            if self.transform is not None:
                img = self.transform(img)

        # Get image label
        if self.labels is None:
            label = self.labels_placeholder[index]
        else:
            if self.label_id:
                label = self.labels[index]
            else:
                label = self.label_map[self.labels[index]]
        return img, aug_1, index, label, self.filepaths[index]



class EuroSAT(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        self.filepaths = [f"{root}/{f.split('_')[0]}/{f}" for f in filepaths]


class DTD(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        if class_folder:
            paths = []
            for f in filepaths:
                cl = f.split("_")[0]
                tr_files = os.listdir(f"{root}/train/{cl}")
                val_files = os.listdir(f"{root}/val/{cl}")
                if f in tr_files:
                    paths.append(f"{root}/train/{cl}/{f}")
                elif f in val_files:
                    paths.append(f"{root}/val/{cl}/{f}")

            self.filepaths = paths

        else:
            self.filepaths = [f"{root}/{f}" for f in filepaths]


class CUB(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        self.filepaths = [f"{root}/{f}" for f in filepaths]


class RESICS45(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        self.filepaths = []
        for f in filepaths:
            folder = "_".join(f.split("_")[:-1])
            self.filepaths.append(f"{root}/{folder}/{f}")


class FGVCAircraft(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        if class_folder:
            filepaths = list(filepaths)
            new_paths = []
            for f in original_filepaths:
                img = f.split("/")[-1]
                if img in filepaths:
                    new_paths.append(f"{f}")

            self.filepaths = new_paths
        else:
            # Adjust filepaths
            self.filepaths = [f"{root}/{f}" for f in filepaths]


class MNIST(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        if class_folder:
            filepaths = list(filepaths)
            new_paths = []
            for f in original_filepaths:
                img = f.split("/")[-1]
                if img in filepaths:
                    new_paths.append(f"{f}")

            self.filepaths = new_paths
        else:
            # Adjust filepaths
            self.filepaths = [f"{root}/{f}" for f in filepaths]


class Flowers102(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        if class_folder:
            filepaths = list(filepaths)
            new_paths = []
            for f in original_filepaths:
                img = f.split("/")[-1]
                if img in filepaths:
                    new_paths.append(f"{f}")

            self.filepaths = new_paths

        else:
            self.filepaths = [f"{root}/{f}" for f in filepaths]
