import argparse
import logging
import os
import sys
from logging import StreamHandler
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator


#new:
from utils import (
    Config,
    set_obj_conf,
    dataset_object,
    evaluate_predictions,
    get_class_names,
    get_labeled_and_unlabeled_data, 
    save_parameters,
    save_predictions,
    store_results,
    monitor_and_accelerate,
    become_deterministic,
)
import os.path as osp
from methods.semi_supervised_learning_new import (
    # MultimodalFPL_PL,
    TextualFPL_PL,
    VisualFPL_PL,
)
import time

accelerator = Accelerator()
torch.set_num_threads(2) #NOTE To maximize efficiency, please tune the number of threads for your machine

logger_ = logging.getLogger()
logger_.level = logging.INFO
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")


class AccelerateHandler(StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)

    def emit(self, record):
        if accelerator.is_local_main_process:
            super().emit(record)


stream_handler = AccelerateHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger_.addHandler(stream_handler)

log = logging.getLogger(__name__)


def prepare_dataset_SSL(obj_conf, classes, labeled_data, test_data):
    all_files, all_labels = zip(*labeled_data)
    SSL_test_files, SSL_test_labeles = zip(*test_data)
    label_to_idx = {c: idx for idx, c in enumerate(classes)}

    # Select few-samples
    few_shots_files = []
    few_shots_labs = []
    
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    for c in classes:
        np.random.seed(obj_conf.validation_seed)
        indices = np.random.choice(
            np.where(all_labels == c)[0], 
            size=obj_conf.N_LABEL, 
            replace=False, 
        )
        few_shots_files += list(all_files[indices])
        few_shots_labs += list(all_labels[indices])

    log.info(f"NUMBER OF SHOTS = {len(classes)} (NUM_CLASSES) X {obj_conf.N_LABEL} (SHOTS PER CLASS): {obj_conf.N_LABEL*len(classes)}")
    log.info(f"NUMBER OF SHOTS {len(few_shots_labs)}")
    
    # Define the set of unlabeled data which excludes the few samples labeled data
    SSL_unlabeled_files = []
    SSL_unlabeled_lbs = []
    for idx, f in enumerate(all_files):
        if f not in few_shots_files:
            SSL_unlabeled_files += [f]
            SSL_unlabeled_lbs += [all_labels[idx]]

    log.info(f"Size of unnlabeled data: {len(SSL_unlabeled_files)}")
    
    # Define the few shots as the labeled data
    SSL_labeled_files = few_shots_files
    SSL_labeled_labeles = few_shots_labs

    SSL_train_files = SSL_unlabeled_files + SSL_labeled_files
    SSL_train_lbs_true = SSL_unlabeled_lbs + SSL_labeled_labeles

    return (SSL_test_files, SSL_test_labeles, 
            label_to_idx, 
            SSL_train_files, SSL_train_lbs_true,
            SSL_unlabeled_files, SSL_unlabeled_lbs)




def workflow_new(dataset_dir, obj_conf):
    # Get dataset name
    # We get the dataset name from the dev_config.py
    dataset = obj_conf.DATASET_NAME
    # Get class names of target task
    # define function for each dataset
    classes, seen_classes, unseen_classes = get_class_names(dataset, dataset_dir, obj_conf.SPLIT_SEED)
    # diff1: We set seen and unseen to classes, since are not in the trzsl setting
    seen_classes = classes
    unseen_classes = classes
    # Create dict classes
    dict_classes = {
        "classes": classes,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    # Log number of classes
    log.info(f"\n----------------------DATA INFO-----------------------\n")
    log.info(f"Number of classes <{obj_conf.SPLIT_SEED} split>: {len(classes)}")
    # Path for images
    data_folder = f"{dataset_dir}/{dataset}"
    log.info(f"Data folder: {data_folder}")
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"cuda Device {i}: {torch.cuda.get_device_name(i)}")
    log.info(f"\n-------------------------------------------------------------\n")
    
    # Get data 
    labeled_data, unlabeled_data, test_data = get_labeled_and_unlabeled_data(
        dataset, data_folder, seen_classes, unseen_classes, classes
    )
    # diff1: From labeled data of all the target classes we sample few-examples
    (SSL_test_files, SSL_test_labeles, 
    label_to_idx, 
    SSL_train_files, SSL_train_lbs_true,
    SSL_unlabeled_files, SSL_unlabeled_lbs) = prepare_dataset_SSL(obj_conf, classes, labeled_data, test_data)

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
    # Labeled training set
    train_dataset = DatasetObject(
        SSL_train_files,
        data_folder,
        transform=None, # Set later 
        augmentations=None,
        train=True,
        labels=None,
        label_map=label_to_idx,
    )
    # Unlabeled training set 
    train_unlabeled_dataset = DatasetObject(
        SSL_unlabeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=None,
        label_map=label_to_idx,
    )
    # Adjust the name file to correctly load data
    truncated_unseen_labeled_files = train_unlabeled_dataset.filepaths
    # Test set 
    test_dataset = DatasetObject(
        SSL_test_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=SSL_test_labeles,
        label_map=label_to_idx,
    )
    # Log info data
    log.info(f"\n----------------------TRAINING DATA INFO-----------------------\n")
    log.info(f"Size labeled data: {len(train_dataset.filepaths) - len(train_unlabeled_dataset.filepaths)}")
    log.info(f"Size unlabeled data: {len(train_unlabeled_dataset.filepaths)}")
    log.info(f"Size test data: {len(test_dataset.filepaths)}")
    log.info(f"\n-------------------------------------------------------------\n")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"\n----------------------MODEL INFO-----------------------\n")

    if obj_conf.MODEL == "textual_fpl":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TextualFPL_PL(
            obj_conf, 
            label_to_idx, 
            data_folder,
            unlabeled_files=truncated_unseen_labeled_files,
            device=device, 
            **dict_classes
        )
        # only for monitoring training process acc (optional):
        monitor_and_accelerate(SSL_train_lbs_true, train_dataset, 
                               test_dataset, model, unlabeled_train_lbs_true=SSL_unlabeled_lbs)
        val_accuracy, optimal_prompt = model.train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=train_unlabeled_dataset,
            only_seen=False,
            iter_num=1,
        )

    elif obj_conf.MODEL == "grip_textual":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TextualFPL_PL(
            obj_conf, 
            label_to_idx, 
            data_folder,
            unlabeled_files=truncated_unseen_labeled_files,
            device=device, 
            **dict_classes
        )
        # only for monitoring training process acc (optional):
        monitor_and_accelerate(SSL_train_lbs_true, train_dataset, 
                               test_dataset, model, unlabeled_train_lbs_true=SSL_unlabeled_lbs)
        val_accuracy, optimal_prompt = model.grip_train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=train_unlabeled_dataset,
            test_data=test_dataset,
            only_seen=False,
        )

    elif obj_conf.MODEL == "grip_visual":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VisualFPL_PL(
            obj_conf, 
            label_to_idx, 
            data_folder,
            unlabeled_files=truncated_unseen_labeled_files,
            device=device, 
            **dict_classes
        )
        # only for monitoring training process acc (optional):
        monitor_and_accelerate(SSL_train_lbs_true, train_dataset, 
                               test_dataset, model, unlabeled_train_lbs_true=SSL_unlabeled_lbs,
                               model_type=model.config.MODALITY)
        val_accuracy, optimal_prompt = model.grip_train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=train_unlabeled_dataset,
            test_data=test_dataset,
            only_seen=False,
        )

    elif obj_conf.MODEL == "visual_fpl":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VisualFPL_PL(
            obj_conf, 
            label_to_idx, 
            data_folder,
            unlabeled_files=truncated_unseen_labeled_files,
            device=device, 
            **dict_classes
        )
        # only for monitoring training process acc (optional):
        monitor_and_accelerate(SSL_train_lbs_true, train_dataset, 
                               test_dataset, model, unlabeled_train_lbs_true=SSL_unlabeled_lbs,
                               model_type=model.config.MODALITY)
        val_accuracy, optimal_prompt = model.train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=train_unlabeled_dataset,
            only_seen=False,
            iter_num=1,
        )

    if obj_conf.MODEL != 'clip_baseline':
        # Save prompt
        save_parameters(optimal_prompt, obj_conf)
        
    # Validate on test set (standard)
    std_predictions = model.test_predictions(test_dataset, standard_zsl=True)

    log.info(f"Testset accuracy: {std_predictions}")
    # Store model results
    # store_results(obj_conf, [std_predictions, None, None]) 


#============= Set logger and config =============
def log_args_and_env(cfg):
    log.info('************')
    log.info('** Config **')
    log.info('************')
    log.info(cfg)
    log.info('Collecting env info ...')
    # log.info('** System info **\n{}\n'.format(collect_env_info()))
 
def setup_log_path(dir=None):
    if dir is None:
        return

    if dir.endswith(".txt") or dir.endswith(".log"):
        fpath = dir
    else:
        fpath = osp.join(dir, "log.txt")

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")
    return fpath

def set_logger(obj_conf):
    if obj_conf.OUTPUT_DIR != "":
        obj_conf.OUTPUT_DIR = obj_conf.OUTPUT_DIR
    else:
        obj_conf.OUTPUT_DIR = f"logs/for_DEBUG/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/', '-')}"

    if not os.path.exists(obj_conf.OUTPUT_DIR):
        os.makedirs(obj_conf.OUTPUT_DIR)
    fpath = setup_log_path(dir=obj_conf.OUTPUT_DIR)
    file_handler = logging.FileHandler(fpath)

    file_handler.setFormatter(formatter)
    # Add the FileHandler to the logger
    logger_.addHandler(file_handler)


#============= Main function =============
def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config.yml",
        help="Name of model config file",
    )
    parser.add_argument(
        "--learning_paradigm",
        type=str,
        default="trzsl",
        help="Choose among trzsl, ssl, and ul",
    )

    args = parser.parse_args()

    with open(f"methods_config/{args.model_config}", "r") as file:
        config = yaml.safe_load(file)

    # ===========Cast configs to object===========
    obj_conf, dataset_dir = set_obj_conf(args, config)
    
    # Set the file path for the log file
    set_logger(obj_conf)

    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Dataset dir: {dataset_dir}")

    log_args_and_env(obj_conf)

    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        print(dataset_dir)
        raise Exception("`dataset_dir` does not exist..")

    # Set random seeds:
    become_deterministic(obj_conf.OPTIM_SEED)
    log.info('Setting fixed seed: {}'.format(obj_conf.OPTIM_SEED))

    accelerator.wait_for_everyone()

    # ===========run workflow===========
    # workflow(dataset_dir, obj_conf)
    workflow_new(dataset_dir, obj_conf)


if __name__ == "__main__":
    main()
