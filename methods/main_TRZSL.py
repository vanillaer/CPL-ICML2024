import argparse
import logging
import os
import sys
from logging import StreamHandler
from pathlib import Path

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
from methods.transductive_zsl_new import (
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


#============= CPL Workflow =============
def prepare_dataset_TRZSL(classes, labeled_data, unlabeled_data, test_data):
    labeled_files, labeled_labels = zip(*labeled_data)
    TRZSL_unlabeled_files, TRZSL_unlabeled_lbs = zip(*unlabeled_data)
    test_labeled_files, test_labeles = zip(*test_data)
    label_to_idx = {c: idx for idx, c in enumerate(classes)}

    TRZSL_train_files = list(TRZSL_unlabeled_files) + list(labeled_files)
    TRZSL_train_lbs_true = list(TRZSL_unlabeled_lbs) + list(labeled_labels)

    return (test_labeled_files, test_labeles, 
            label_to_idx,
            TRZSL_train_files, TRZSL_train_lbs_true,
            TRZSL_unlabeled_files, TRZSL_unlabeled_lbs)
 

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

    # Create datasets
    (test_labeled_files, test_labeles, 
    label_to_idx,
    TRZSL_train_files, TRZSL_train_lbs_true,
    TRZSL_unlabeled_files, TRZSL_unlabeled_lbs) = prepare_dataset_TRZSL(classes, labeled_data, unlabeled_data, test_data)

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
    # Training set (all data)
    train_dataset = DatasetObject(
        TRZSL_train_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=None,
        label_map=label_to_idx,
    )
    # Training set (unlabeled unseen)
    train_unseen_dataset = DatasetObject(
        TRZSL_unlabeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=None,
        label_map=label_to_idx,
    )
    # Adjust the name file to correctly load data
    truncated_unseen_labeled_files = train_unseen_dataset.filepaths
    # Test set (test seen and unseen)
    test_dataset = DatasetObject(
        test_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=test_labeles,
        label_map=label_to_idx,
    )

    # Log info data
    log.info(f"\n----------------------TRAINING DATA INFO-----------------------\n")
    log.info(f"Len training seen data: {len(train_dataset.filepaths)}")
    log.info(f"Average number of labeled images per seen class:{(len(train_dataset) - len(train_unseen_dataset))/len(seen_classes)} ")
    log.info(f"Len training unseen data: {len(train_unseen_dataset.filepaths)}")
    log.info(f"Len test data: {len(test_dataset.filepaths)}")
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
        monitor_and_accelerate(TRZSL_train_lbs_true, train_dataset, 
                               test_dataset, model, 
                               unlabeled_train_lbs_true=TRZSL_unlabeled_lbs)
        val_accuracy, optimal_prompt = model.train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=train_unseen_dataset,
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
        monitor_and_accelerate(TRZSL_train_lbs_true, train_dataset, 
                               test_dataset, model, 
                               unlabeled_train_lbs_true=TRZSL_unlabeled_lbs)
        val_accuracy, optimal_prompt = model.grip_train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=train_unseen_dataset,
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
        monitor_and_accelerate(TRZSL_train_lbs_true, train_dataset, 
                               test_dataset, model, unlabeled_train_lbs_true=TRZSL_unlabeled_lbs,
                               model_type=model.config.MODALITY)
        val_accuracy, optimal_prompt = model.grip_train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=train_unseen_dataset,
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
        monitor_and_accelerate(TRZSL_train_lbs_true, train_dataset, 
                               test_dataset, model, unlabeled_train_lbs_true=TRZSL_unlabeled_lbs,
                               model_type=model.config.MODALITY)
        val_accuracy, optimal_prompt = model.train(
            train_data=train_dataset, 
            val_data=None,
            unlabeled_data=train_unseen_dataset,
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
