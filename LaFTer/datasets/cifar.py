import math
import os.path as osp
import os
import pickle
from dassl.utils import listdir_nohidden, mkdir_if_missing
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
import numpy as np
import random
import math
import os.path as osp
import os
import pickle

cifar10_classes = ["airplane",
                        "automobile",
                        "bird",
                        "cat",
                        "deer",
                        "dog",
                        "frog",
                        "horse",
                        "ship",
                        "truck"]
cifar100_classes = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]
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
        self.dataset_dir = osp.join(root, self.dataset_dir_)
        train_dir = osp.join(self.dataset_dir, 'train')
        test_dir = osp.join(self.dataset_dir, 'test')
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train_all = self._read_data_train(
            train_dir, 0
        )

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
        
        #new added for imbalance dataset
        if hasattr(cfg.DATASET, 'IMBALANCE_RATIO') and cfg.DATASET.IMBALANCE_RATIO > 0:
            img_num_list = self.get_img_num_per_cls(self.num_cls, 
                                                    dataset_len=len(train_all),
                                                    imb_type='exp', 
                                                    imb_factor=cfg.DATASET.IMBALANCE_RATIO)
            train_all = self.gen_imbalanced_data(img_num_list, train_all)
            print(f"Imbalance dataset with ratio {cfg.DATASET.IMBALANCE_RATIO}")
        super().__init__(train_x=train_all, test=test)


    def get_img_num_per_cls(self, cls_num, dataset_len, imb_type, imb_factor):
        img_max = dataset_len / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, train_all):
        new_data = []
        # new_targets = []
        targets_all = []
        for i in range(len(train_all)):
            targets_all.append(train_all[i].label)
        targets_np = np.array(targets_all, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # new_data.append(self.x_paths[selec_idx, ...])
            new_data.extend([train_all[i] for i in selec_idx])
            # new_targets.extend([the_class, ] * the_img_num)

        return new_data


    def _read_data_train(self, data_dir, val_percent):
        if self.dataset_dir_ == 'cifar10':
            class_names_ = cifar10_classes
        else:
            class_names_ = cifar100_classes
        items_x = []
        self.num_cls = len(class_names_)
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
        class_names.sort()
        items = []
        for label, class_name in enumerate(class_names):
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
