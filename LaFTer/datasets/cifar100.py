import math
import random
import os.path as osp

from dassl.utils import listdir_nohidden
import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
# from ..build import DATASET_REGISTRY
# from ..base_dataset import Datum, DatasetBase
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
import datasets.cifar as cifar_classes

@DATASET_REGISTRY.register()
class CIFAR10_local(DatasetBase):
    """CIFAR10 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """
    dataset_dir_ = 'cifar10'

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        # print(self.dataset_dir)
        # quit()
        self.dataset_dir = osp.join(root, self.dataset_dir_)
        train_dir = osp.join(self.dataset_dir, 'train')
        test_dir = osp.join(self.dataset_dir, 'test')
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # cfg.DATASET.NUM_LABELED = 50000
        # assert cfg.DATASET.NUM_LABELED > 0

        train_all = self._read_data_train(
            train_dir, 0
        )

        # train_stripped = self._read_data_train_stripped(train_dir, 0)

        test = self._read_data_test(test_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train_all, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)




        # super().__init__(train_x=train, test=test, train_u=train_all)
        super().__init__(train_x=train_all, test=test)

        # super().__init__(train_x=train_stripped, test=test, train_u=train_all)

    def _read_data_train(self, data_dir, val_percent):
        if self.dataset_dir_ == 'cifar10':
            class_names_ = cifar_classes.cifar10_classes
        else:
            class_names_ = cifar_classes.cifar100_classes
        items_x = []
        for label, class_name in enumerate(class_names_):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)
            num_val = math.floor(len(imnames) * val_percent)
            imnames_train = imnames[num_val:]
            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=class_name)
                items_x.append(item)

        return items_x

    def _read_data_train_stripped(self, data_dir, val_percent):
        if self.dataset_dir_ == 'cifar10':
            class_names_ = cifar_classes.cifar10_classes
        else:
            class_names_ = cifar_classes.cifar100_classes[:50]
        items_x = []
        for label, class_name in enumerate(class_names_):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)
            num_val = math.floor(len(imnames) * val_percent)
            imnames_train = imnames[num_val:]
            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=class_name)
                items_x.append(item)

        return items_x

    def _read_data_test(self, data_dir):
        class_names = listdir_nohidden(data_dir)
        # print(class_names)
        # quit()
        class_names.sort()
        items = []
        for label, class_name in enumerate(class_names):
            # print(label, class_name)
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)
            for imname in imnames:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=class_name)
                items.append(item)
        return items


@DATASET_REGISTRY.register()
class CIFAR100_local(CIFAR10_local):
    """CIFAR100 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """
    dataset_dir_ = 'cifar100'

    def __init__(self, cfg):
        super().__init__(cfg)
