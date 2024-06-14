import logging
import random

import numpy as np
import torch
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from data import dataset_custom_prompts

log = logging.getLogger(__name__)


def set_obj_conf(args, config):
    obj_conf = Config(config)

    # Set seed
    optim_seed = int(os.environ["OPTIM_SEED"])
    obj_conf.OPTIM_SEED = optim_seed
    # Set backbone
    obj_conf.VIS_ENCODER = os.environ["VIS_ENCODER"]
    # Set dataset name
    obj_conf.DATASET_NAME = os.environ["DATASET_NAME"]
    # Set dataset dir
    obj_conf.DATASET_DIR = os.environ["DATASET_DIR"]
    # Set model name
    obj_conf.MODEL = os.environ["MODEL"]
    # Set split seed
    obj_conf.SPLIT_SEED = int(os.environ["SPLIT_SEED"])
    # Set dataset's template for textual prompts
    obj_conf.PROMPT_TEMPLATE = dataset_custom_prompts[obj_conf.DATASET_NAME]
    # Set data dir
    dataset_dir = obj_conf.DATASET_DIR
    # Set learning paradigm
    obj_conf.LEARNING_PARADIGM = args.learning_paradigm
    if os.environ.get("STEP_QUANTILE") is not None:
        obj_conf.STEP_QUANTILE = eval(os.environ.get("STEP_QUANTILE"))

    #set loss config:
    if os.environ.get("LOSS_TYPE") is not None:
        obj_conf.LOSS_CFG = getattr(obj_conf.LOSS_CFG, os.environ.get("LOSS_TYPE"))
    #set TARGET_PARTIAL_RATIO and INIT_PARTIAL_RATIO config:
    if os.environ.get("INIT_PARTIAL_RATIO") is not None:
        obj_conf.PartialY_CFG.INIT_PARTIAL_RATIO = eval(os.environ.get("INIT_PARTIAL_RATIO"))
        obj_conf.PartialY_CFG.TARGET_PARTIAL_RATIO = obj_conf.PartialY_CFG.INIT_PARTIAL_RATIO
    #set MAX_CONF_THRESHOLD:
    if os.environ.get("MAX_CONF_THRESHOLD") is not None:
        obj_conf.PartialY_CFG.MAX_CONF_THRESHOLD = eval(os.environ.get("MAX_CONF_THRESHOLD"))
    #set LAMBDA:
    if os.environ.get("LAMBDA") is not None:
        obj_conf.LAMBDA = eval(os.environ.get("LAMBDA"))

    #set conf CANDIDATE_METHOD config:
    if os.environ.get("CANDIDATE_METHOD") is not None:
        candidate_method = os.environ.get("CANDIDATE_METHOD")
        obj_conf.PartialY_CFG.CANDIDATE_METHOD = candidate_method
        if candidate_method == 'CPL' or candidate_method == 'inter_inst':
            obj_conf.PartialY_CFG.CONF_THRESHOLD = 'quantile'
        elif candidate_method == 'intra_inst' or candidate_method == 'mix':
            obj_conf.PartialY_CFG.CONF_THRESHOLD = 'auto'
        else:
            raise NotImplementedError
        
    #set conf thr config:
    if os.environ.get("CONF_THRESHOLD") is not None:
        obj_conf.PartialY_CFG.CONF_THRESHOLD = os.environ.get("CONF_THRESHOLD")
    #set conf OUTPUT_DIR config:
    if os.environ.get("OUTPUT_DIR") is not None:
        obj_conf.OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
    #set PSEUDOSHOTS_PERCENT config:
    if os.environ.get("PSEUDOSHOTS_PERCENT") is not None:
        obj_conf.Selector_CFG.PSEUDOSHOTS_PERCENT = eval(os.environ.get("PSEUDOSHOTS_PERCENT"))
    #set USE_SOFT_PARTIAL config:
    if os.environ.get("USE_SOFT_PARTIAL") is not None:
        obj_conf.PartialY_CFG.USE_SOFT_PARTIAL = eval(os.environ.get("USE_SOFT_PARTIAL"))
    #set TEMPERATURE config:
    if os.environ.get("TEMPERATURE") is not None:
        obj_conf.TEMPERATURE = eval(os.environ.get("TEMPERATURE"))

    if os.environ.get("EPOCHS") is not None:
        obj_conf.EPOCHS = eval(os.environ.get("EPOCHS"))
    if os.environ.get("LR") is not None:
        obj_conf.LR = eval(os.environ.get("LR"))
    if os.environ.get("DECAY") is not None:
        obj_conf.DECAY = eval(os.environ.get("DECAY"))
    if os.environ.get("BATCH_SIZE") is not None:
        obj_conf.BATCH_SIZE = eval(os.environ.get("BATCH_SIZE"))
    
    # NOTE: CONF_QUANTILE is used to represent the hyperparameter (alpha*100) in the paper
    # NOTE: REGULAR_THRESHOLD is used to represent the hyperparameter beta in the paper
    #set CONF_QUANTILE
    if os.environ.get("CONF_QUANTILE") is not None:
        obj_conf.PartialY_CFG.CONF_QUANTILE = eval(os.environ.get("CONF_QUANTILE"))
    #set REGULAR_THRESHOLD thr config:
    if os.environ.get("REGULAR_THRESHOLD") is not None:
        r_thr = os.environ.get("REGULAR_THRESHOLD")
        try:
            obj_conf.PartialY_CFG.REGULAR_THRESHOLD = eval(r_thr)
        except:
            assert 'auto' in r_thr
            obj_conf.PartialY_CFG.REGULAR_THRESHOLD = r_thr
    
    return obj_conf, dataset_dir


def dataset_object(dataset_name):
    if dataset_name == "aPY":
        from data import aPY as DataObject
    elif dataset_name == "Animals_with_Attributes2":
        from data import AwA2 as DataObject
    elif dataset_name == "EuroSAT":
        from data import EuroSAT as DataObject
    elif dataset_name == "DTD":
        from data import DTD as DataObject
    elif dataset_name == "sun397":
        from data import SUN397 as DataObject
    elif dataset_name == "CUB":
        from data import CUB as DataObject
    elif dataset_name == "RESICS45":
        from data import RESICS45 as DataObject
    elif dataset_name == "FGVCAircraft":
        from data import FGVCAircraft as DataObject
    elif dataset_name == "MNIST":
        from data import MNIST as DataObject
    elif dataset_name == "Flowers102":
        from data import Flowers102 as DataObject
    
    DataObject.dataset_name = dataset_name
    return DataObject

def makedirs(path, verbose=False):
    '''Make directories if not exist.'''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if verbose:
            print(path + " already exists.")



def calculate_class_accuracy_as_dict(gt_lbs, output_logits):
    """
    Calculates the accuracy for each class as a dictionary where keys are class indices
    and values are accuracy rates.

    Args:
        gt_lbs (Tensor): Ground truth labels for the dataset.
        output_logits (Tensor): The logits output by the model for the dataset.

    Returns:
        dict: A dictionary mapping class indices to their corresponding accuracy rates.
    """
    # Convert logits to predicted labels
    _, predicted_labels = torch.max(output_logits, 1)

    num_classes = output_logits.shape[1]
    class_accuracies = {}

    for cls in range(num_classes):
        indices = (gt_lbs == cls)
        correct = predicted_labels[indices].eq(gt_lbs[indices]).sum().item()
        total = indices.sum().item()

        if total > 0:
            class_accuracies[cls] = correct / total
        else:
            # Set accuracy to 0 for classes with no samples
            class_accuracies[cls] = 0

    return class_accuracies


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def become_deterministic(seed=0):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Seed for cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # can set to false
    # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # torch.autograd.set_detect_anomaly(True)

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


class Config(object):
    def __init__(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def __str__(self, level=0) -> str:
        ret = ""
        indent = "  " * level
        dict_list = []
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                dict_list.append("\n".join([indent + f"{k}:", v.__str__(level + 1)]))
            else:
                dict_list.append(indent + f"{k}: {v}")
        ret += "\n".join(dict_list)
        return ret


class CustomImageDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TrainSampler(object):
    """This class is used to sample labeled examples for each batch for SSL task."""

    def __init__(self, x, y, bs, device, mode='feats', 
                                        transform=None,
                                        img_path=None):
        self.mode = mode
        assert len(x) == len(y)
        if mode == 'feats':
            self.x = x.to(device)
            self.y = y.to(device)
            self.bs = bs
            self.n = len(x)
            self.reset_indices()
            self.device = device
        elif mode == 'filepaths':
            # Create the dataset
            assert img_path is not None and transform is not None
            dataset = CustomImageDataset(img_path, y, transform=transform)
            # Create the dataloader
            dataloader = DataLoader(dataset, 
                                    batch_size=bs, 
                                    num_workers=6,
                                    drop_last=False,
                                    pin_memory=(torch.cuda.is_available()),
                                    shuffle=True)
            self.dataloader = dataloader
            self.bs = bs
            self.n = len(img_path)
            self.device = device

    def reset_indices(self):
        self.indices = torch.randperm(self.n)
        self.current_idx = 0

    def get_onebatch(self):
        if self.mode == 'feats':
            # not drop the last batch if it is not full
            if self.current_idx >= self.n:
                self.reset_indices()
            batch_x = self.x[self.indices[self.current_idx: self.current_idx+self.bs]]
            batch_y = self.y[self.indices[self.current_idx: self.current_idx+self.bs]]
            self.current_idx += self.bs
            return batch_x, batch_y
        
        elif self.mode == 'filepaths':
            try:
                batch_x, batch_y = next(self.dataloader.__iter__())
            except StopIteration:
                self.dataloader = iter(self.dataloader)
                batch_x, batch_y = next(self.dataloader.__iter__())
            return batch_x.to(self.device), batch_y.to(self.device)


    def __len__(self):
        # Number of batches
        return (self.n + self.bs - 1) // self.bs
        


def monitor_and_accelerate(train_lbs_true, train_dataset, test_dataset, model, 
                           unlabeled_train_lbs_true=None, 
                           model_type='text'):
    """
    Monitor and accelerate the training process.

    Parameters:
    train_lbs_true (List[int]): List of true labels for the training data.
    train_dataset (Dataset): The training dataset.
    test_dataset (Dataset): The test dataset.
    model (Model): The model to be trained.
    unlabeled_train_lbs_true (List[int], optional): List of true labels for the unlabeled training data. Defaults to None.
    model_type (str, optional): The type of the model 'text' or 'image'.
    """
    train_labels_true = [train_dataset.label_map[item] for item in train_lbs_true]
    if unlabeled_train_lbs_true is not None: # For UL:
        unlabeled_train_lbs_true = [train_dataset.label_map[item] for item in unlabeled_train_lbs_true]
    else: # For SSL, TRZSL:
        unlabeled_train_lbs_true = train_labels_true

    for cls_idx, pool in model.Selector.Pools.cls_pools_dict.items():
        pool.labels_true = torch.tensor(unlabeled_train_lbs_true)

    model.all_gt_label_dict = {img_path: label_true for (img_path, label_true) in 
                                    zip(train_dataset.filepaths, train_labels_true)}
        
    if model_type == 'text':
    # when using text PT, we can cache the features to accelerate the training process
        train_dataset.prepare_features(model=model.clip_model)
        test_dataset.prepare_features(model=model.clip_model)


def find_elem_idx_BinA(A, B):
    """
    This function finds the indices of the elements of tensor b in tensor a.
    
    Parameters:
    a (torch.Tensor): The tensor in which to find the indices.
    b (torch.Tensor): The tensor whose elements' indices are to be found.
    
    Returns:
    torch.Tensor: A tensor containing the indices of the elements of b in a.
    """
    # Create a dictionary with elements of a as keys and their indices as values
    a_dict = {item.item(): i for i, item in enumerate(A)}
    
    # Map the elements of b to their corresponding indices in a using the dictionary
    indices = torch.tensor([a_dict[item.item()] for item in B], dtype=torch.long)
    
    return indices



class PoolsAggregation:
    """
    Administer the pool of each class.
    """

    def __init__(self, class_ids, K, max_capacity_per_class=None):
        """
        Initialize the PoolsAggregation.
        Args:
            cfg (Config): The configuration object.
            class_ids (list): a list of class ids.
            K (int): Number of top samples to select per class.
            max_capacity_per_class (dict): Maximum capacity per class. 
        """
        self.min_cap = K
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize cls_pools_dict
        self.cls_pools_dict = {}
        if max_capacity_per_class is None:
            max_capacity_per_class = {cls: K for cls in class_ids}

        # Convert max_capacity_per_class to a tensor for efficient computation
        max_capacity_per_class = [max_capacity_per_class[cls] for cls in class_ids]

        # Loop through each unique class id
        for i, cls in enumerate(class_ids):                                   
            self.cls_pools_dict[cls] = ClassPool(max_capacity=max_capacity_per_class[i], 
                                                 cls_id=cls)

    def __len__(self):
        return len(self.cls_pools_dict)

    def scale_all_pools(self, scale_nums):
        """Manipulate the scale of each pool in its government"""
        for cls_idx, pool in self.cls_pools_dict.items():
            next_capacity = scale_nums[cls_idx]
            next_capacity = next_capacity
            pool.scale_pool(next_capacity=next_capacity)


    def reset_all(self):
        """Reset all pools in its government"""
        for pool in self.cls_pools_dict.values():
            pool.reset()


    def batch_fill_assigned_pool(self, feat_idxs: torch.LongTensor, feat_uncs: torch.Tensor, pool_ids):
        """
        Fill the assigned pool with new values in batch.
        Args:
            feat_idxs (torch.Tensor): A tensor of feature indices, better to be ascending order.
            feat_uncs (torch.Tensor): A tensor of feature uncertainties.
            pool_ids: the assigned_pool to fill all the items.
        """
        # in_pool = torch.zeros_like(feat_idxs, dtype=torch.bool)
        for pool_id in pool_ids.unique():
            mask = pool_ids == pool_id
            cur_pool = self.cls_pools_dict[pool_id.item()]
            cur_pool.batch_update(feat_idxs[mask], feat_uncs[mask]) # in_pool[mask] = 


    def get_all_feat_idxs(self):
        """
        Get all feature indices for all pool in pool_dict.
        Returns:
            torch.Tensor: A tensor of feature indices.
        """
        feat_idxs = torch.LongTensor([])
        for pool in self.cls_pools_dict.values():
            feat_idxs = torch.cat((feat_idxs, pool.pool_idx), dim=0)
        return feat_idxs
    

    def cal_pool_sum_num(self):
        sum_num = 0
        for i, pool in enumerate(self.cls_pools_dict.values()):
            sum_num += pool.pool_capacity
            # print(f'pool_id: {i}, pool_capacity: {pool.pool_capacity}')
        return sum_num
    
    def get_pool_caps(self):
        cap_list = []
        for i, pool in enumerate(self.cls_pools_dict.values()):
            cap_list.append(pool.pool_capacity)
            # print(f'pool_id: {i}, pool_capacity: {pool.pool_capacity}')
        return cap_list

    def cal_pool_ACC(self):
        correct_num = 0
        all_num = 0
        for pool in self.cls_pools_dict.values():
            pred_labels = pool.convert_pred_idxs_to_real(torch.LongTensor([pool.cls_id]))
            correct = (pool.labels_true[pool.pool_idx] == pred_labels).sum()
            correct_num += correct
            all_num += pool.pool_capacity
        log.info(f'====> overall pools ACC: {correct_num}/{all_num} = {correct_num/all_num}')


    def print(self):
        for pool_id, cur_pool in self.cls_pools_dict.items():
            log.info(cur_pool)



class ClassPool:
    """
    Store the average and current values for uncertainty of each class samples and the max capacity of the pool.
    """

    def __init__(self, max_capacity: int, cls_id):
        """
        Initialize the ClassPool.
        Args:
            max_capacity (int): The maximum capacity of the pool.
            items_idx (torch.LongTensor): A tensor of item indices.
            items_unc (torch.Tensor): A tensor of item uncertainties.
        """
        self.pool_max_capacity = max_capacity
        self.is_freeze = False
        self.cls_id = cls_id
        self.device = 'cuda'
        self.unc_dtype = torch.float32
        self.baseline_capacity = max_capacity
        self.reset()
        
    def _update_pool_attr(self):
        """
        Update the pool attributes.
        """
        # self.unc_avg = torch.mean(self.pool_unc)
        self.unc_max, self.unc_max_idx = torch.max(self.pool_unc, dim=0)
        assert self.pool_unc.shape == self.pool_unc.shape
        assert self.pool_unc.shape[0] <= self.pool_max_capacity
    
    def reset(self):
        """
        Reset the pool.
        """
        self.pool_idx = torch.LongTensor([])
        self.pool_unc = torch.Tensor([]).type(self.unc_dtype).to(self.device)
        self.popped_idx = torch.LongTensor([])
        self.popped_unc = torch.Tensor([]).type(self.unc_dtype).to(self.device)
        #attribute:
        self.pool_capacity = 0
        self.unc_max = 1e-10

        assert self.is_freeze == False
        self.pool_unc_past = None
        self.pool_idx_past = None
        self.replace_num = 0
        self.not_in_num = 0

    def scale_pool(self, next_capacity: int):
        """
        Scale the pool to the next iter capacity.
        """
        self.pool_max_capacity = next_capacity
        return


    def batch_update(self, feat_idxs: torch.LongTensor, feat_uncs: torch.Tensor, record_popped=False):
        """
        Update the pool with new values in batch.
        Args:
            feat_idxs (torch.Tensor): A tensor of feature indices, better to be ascending order.
            feat_uncs (torch.Tensor): A tensor of feature uncertainties.
        """
        in_pool = torch.zeros_like(feat_idxs, dtype=torch.bool)
        for i, (feat_idx, feat_unc) in enumerate(zip(feat_idxs, feat_uncs)):
            in_pool[i] = self.update(feat_idx, feat_unc, record_popped)
        return in_pool


    def update(self, feat_idx: torch.LongTensor, feat_unc: torch.Tensor, record_popped=False):
        """
        Update the pool with new values.
        Args:
            feat_idxs (torch.Tensor): A tensor of feature indices, better to be ascending order.
            feat_unc (torch.Tensor): A tensor of feature uncertainties.
        """
        if self.pool_capacity < self.pool_max_capacity:
            if feat_unc < 1e4:
                self.pool_idx = torch.cat((self.pool_idx, feat_idx.unsqueeze(0)))  
                self.pool_unc = torch.cat((self.pool_unc, feat_unc.unsqueeze(0)))  
                # self.saved_logits = torch.cat((self.saved_logits, feat_logit.unsqueeze(0)))  
                self.pool_capacity += 1
                in_pool = True
            else:
                in_pool = False
        else:
            assert self.pool_max_capacity >= self.pool_capacity, \
                f"pool_max_capacity: {self.pool_max_capacity}, pool_capacity: {self.pool_capacity}"
            if self.unc_max <= feat_unc:
                if record_popped:
                    self.popped_idx = torch.cat((self.popped_idx, feat_idx.unsqueeze(0)))  
                    self.popped_unc = torch.cat((self.popped_unc, feat_unc.unsqueeze(0)))  
                in_pool = False
            else:
                if record_popped:
                    self.popped_idx = torch.cat((self.popped_idx, 
                                                 self.pool_idx[self.unc_max_idx].unsqueeze(0)))
                    self.popped_unc = torch.cat((self.popped_unc, 
                                                 self.pool_unc[self.unc_max_idx].unsqueeze(0)))
                    # self.popped_img_feats.append(info_dict['image_feat'])      
                    # self.poped_logits.append(info_dict['logit'])
                
                self.pool_idx[self.unc_max_idx] = feat_idx
                self.pool_unc[self.unc_max_idx] = feat_unc
                # self.saved_logits[self.unc_max_idx] = feat_logit
                in_pool = True
                
        if in_pool:
            self._update_pool_attr()

        return in_pool


    def __str__(self):
        str_ = f'pool_id: {self.cls_id}, '
        if hasattr(self, 'unc_avg'):
            str_ += f"unc_avg: {self.unc_avg:.4f}, "
        if self.unc_max != None:
            str_ += f"unc_max: {self.unc_max:.4f}, "
        else:
            str_ += f"unc_max: None, "
        if hasattr(self, 'labels_true'):
            pred_labels = self.convert_pred_idxs_to_real(torch.LongTensor([self.cls_id]))
            corrcet_num = (self.labels_true[self.pool_idx] == pred_labels).sum()
            str_ += f"pool ACC: {corrcet_num}/{self.pool_capacity}, "
        return str_ + f"pool_capacity: {self.pool_capacity}/{self.pool_max_capacity}"
    
    
    def convert_pred_idxs_to_real(self, pred_idxs):
        """convert pred_idxs to real idxs"""
        #default do nothing and would be redefined in TRZSL setting
        return pred_idxs
    
