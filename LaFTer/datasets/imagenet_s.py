import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
import random
from .imagenet import ImageNet
import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class ImageNetSketch(DatasetBase):
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-sketch"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        data = self.read_data(classnames)
        random.shuffle(data)
        train = data[:int(len(data) * 0.80)]
        test = data[int(len(data) * 0.80):]

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
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


        # test = data[1:]
        super().__init__(train_x=train, test=test)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
