import argparse
import logging
import os
import sys
from logging import StreamHandler
from pathlib import Path

import copy
import time
import os.path as osp
import torch
import yaml
from accelerate import Accelerator

from methods.unsupervised_learning_new import (
    TextualFPL_PL,
    VisualFPL_PL,
)
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
    become_deterministic,
    monitor_and_accelerate,
)


torch.set_num_threads(2) #NOTE To maximize efficiency, please tune the number of threads for your machine
accelerator = Accelerator()

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


#============= CPL Workflow =============
def prepare_dataset_UL(classes, labeled_data, unlabeled_data, test_data):
    """
    Prepare datasets for Unsupervised Learning (UL).

    Parameters:
    classes (List[str]): List of class names.
    labeled_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.
    unlabeled_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.
    test_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.

    Returns:
    Tuple[List[str], List[int], List[str], List[int], Dict[str, int]]: Returns a tuple containing lists of test files, 
    test labels, train files, train labels and a dictionary mapping class names to indices.
    """
    labeled_files, labeles = zip(*labeled_data)
    unseen_labeled_files, unseen_labeles = zip(*unlabeled_data)     #unlabeled == unseen
    # test_labeled_files, test_labeles = zip(*test_data)
    
    # define datasets for UL:
    UL_test_files, UL_test_lbs = zip(*test_data)
    UL_train_files = unseen_labeled_files + labeled_files # for UL we use all the trian data
    UL_train_lbs_true = unseen_labeles + labeles
    label_to_idx = {c: idx for idx, c in enumerate(classes)}
    
    return (UL_test_files, UL_test_lbs, 
            UL_train_files, UL_train_lbs_true, 
            label_to_idx)


def workflow_new(dataset_dir, obj_conf):
    # Get dataset name
    # We get the dataset name from the dev_config.py
    dataset = obj_conf.DATASET_NAME
    # Get class names of target task
    # define function for each dataset
    classes, seen_classes, unseen_classes = get_class_names(dataset, dataset_dir, obj_conf.SPLIT_SEED)
    # Create dict classes to pass as variable
    dict_classes = {
        "classes": classes,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    # Log number of classes
    log.info(f"\n----------------------DATA INFO-----------------------\n")
    log.info(f"Number of classes split {obj_conf.SPLIT_SEED}: {len(classes)}")
    log.info(f"Number of seen classes split {obj_conf.SPLIT_SEED}: {len(seen_classes)}")
    log.info(f"Number of unseen classes split {obj_conf.SPLIT_SEED}: {len(unseen_classes)}")
    # Path for images
    data_folder = f"{dataset_dir}/{dataset}"
    log.info(f"Data folder: {data_folder}")
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"cuda Device {i}: {torch.cuda.get_device_name(i)}")
    log.info(f"\n-------------------------------------------------------------\n")
    
    # Get labeled data (seen classes)
    # Get unlabeled data (unseen classes)
    # Get test data (both seen and unseen classes)
    labeled_data, unlabeled_data, test_data = get_labeled_and_unlabeled_data(
        dataset, data_folder, seen_classes, unseen_classes, classes
    )
    # 1. Create datasets
    (UL_test_files, UL_test_lbs, 
    UL_train_files, UL_train_lbs_true, 
    label_to_idx) = prepare_dataset_UL(classes, labeled_data, unlabeled_data, test_data)
 
    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
    # Training set (unlabeled unseen)
    train_dataset = DatasetObject(
        UL_train_files,       #NOTE here we use all the train data in UL
        data_folder,
        transform=None,                 
        augmentations=None,
        train=True,
        labels=None,
        label_map=label_to_idx,
    )
    # Test set (test seen and unseen)
    test_dataset = DatasetObject(
        UL_test_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=UL_test_lbs,
        label_map=label_to_idx,
    )
    
    # Log info data
    log.info(f"\n----------------------TRAINING DATA INFO-----------------------\n")
    log.info(f"Sice unlabeled data: {len(train_dataset.filepaths)}")
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
            device=device, 
            **dict_classes
        )
        # only for monitoring training process acc (optional):
        monitor_and_accelerate(UL_train_lbs_true, train_dataset, 
                               test_dataset, model)
        val_accuracy, optimal_prompt = model.train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=copy.deepcopy(train_dataset),   #all the train data
            only_seen=False,
            iter_num=1,
        )

    elif obj_conf.MODEL == "grip_textual":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TextualFPL_PL(
            obj_conf, 
            label_to_idx, 
            data_folder,
            device=device, 
            **dict_classes
        )
        # only for monitoring training process acc and accelerating training (optional):
        monitor_and_accelerate(UL_train_lbs_true, train_dataset, 
                               test_dataset, model)
        val_accuracy, optimal_prompt = model.grip_train(
            train_data=train_dataset, 
            val_data=None,
            test_data=test_dataset,
            unlabeled_data=copy.deepcopy(train_dataset),
            only_seen=False,
        )

    elif obj_conf.MODEL == "grip_visual":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VisualFPL_PL(
            obj_conf, 
            label_to_idx, 
            data_folder,
            device=device, 
            **dict_classes
        )
        monitor_and_accelerate(UL_train_lbs_true, train_dataset, 
                               test_dataset, model, model_type=obj_conf.MODALITY)
        val_accuracy, optimal_prompt = model.grip_train(
            train_data=train_dataset, 
            val_data=None,
            test_data=test_dataset,
            unlabeled_data=copy.deepcopy(train_dataset),
            only_seen=False,
        )

    elif obj_conf.MODEL == "visual_fpl":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VisualFPL_PL(
            obj_conf, 
            label_to_idx, 
            data_folder,
            device=device, 
            **dict_classes
        )
        monitor_and_accelerate(UL_train_lbs_true, train_dataset, 
                               test_dataset, model, model_type=obj_conf.MODALITY)
        val_accuracy, optimal_prompt = model.train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=copy.deepcopy(train_dataset),
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
        obj_conf.OUTPUT_DIR = f"logs/for_DEBUG1/{obj_conf.LEARNING_PARADIGM}/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/', '-')}_seed{obj_conf.OPTIM_SEED}"

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
    parser.add_argument(
        "--use_original",
        action="store_true",
        help="Choose if use original settings",
    )

    args = parser.parse_args()
    try:
        with open(f"methods_config/{args.model_config}", "r") as file:
            config = yaml.safe_load(file)
    except:
        with open(f"enhanceCLIPwithCLIP/methods_config/{args.model_config}", "r") as file:
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
    workflow_new(dataset_dir, obj_conf)


if __name__ == "__main__":
    main()

