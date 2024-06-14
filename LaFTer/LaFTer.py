import argparse
import torch
import datetime
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
# custom
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.food101
import datasets.sun397
import datasets.ucf101
import datasets.imagenet_r
import datasets.imagenet
import datasets.imagenet_s
import datasets.imagenet_a
import datasets.caltech101
import datasets.cifar
import trainers.LaFTer as lafter_uft
from utils.utils import *
import os
import yaml
from CandidateAPI.utils import Config, makedirs, detect_anomaly
from CandidateAPI.candidate_main import CandidateTrainer
import torch.nn.functional as F
import copy


torch.set_num_threads(3) #NOTE To maximize efficiency, please tune the number of threads for your machine

def print_args(args=None, cfg=None):
    if args is not None:
        print("***************")
        print("** Arguments **")
        print("***************")
        optkeys = list(args.__dict__.keys())
        optkeys.sort()
        for key in optkeys:
            print("{}: {}".format(key, args.__dict__[key]))
    if cfg is not None:
        print("************")
        print("** Config **")
        print("************")
        print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    #new settings:
    cfg.USE_CANDIDATE = args.use_candidate


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.txt_cls = args.txt_cls
    cfg.gpt_prompts = args.gpt_prompts

    cfg.DATASET.IMBALANCE_RATIO = 0.0


def set_obj_conf(args, config):
    obj_conf = Config(config)

    #set loss config:
    if os.environ.get("LOSS_TYPE") is not None:
        obj_conf.LOSS_CFG = getattr(obj_conf.LOSS_CFG, os.environ.get("LOSS_TYPE"))
    #set TARGET_PARTIAL_RATIO and INIT_PARTIAL_RATIO config:
    if os.environ.get("INIT_PARTIAL_RATIO") is not None:
        obj_conf.PartialY_CFG.INIT_PARTIAL_RATIO = eval(os.environ.get("INIT_PARTIAL_RATIO"))
        obj_conf.PartialY_CFG.TARGET_PARTIAL_RATIO = obj_conf.PartialY_CFG.INIT_PARTIAL_RATIO
    #set conf CANDIDATE_METHOD config:
    if os.environ.get("CANDIDATE_METHOD") is not None:
        candidate_method = os.environ.get("CANDIDATE_METHOD")
        obj_conf.PartialY_CFG.CANDIDATE_METHOD = candidate_method
        if candidate_method == 'CPL' or candidate_method == 'regular':
            obj_conf.PartialY_CFG.CONF_THRESHOLD = 'quantile'
        elif candidate_method == 'cum_prob' or candidate_method == 'mix':
            obj_conf.PartialY_CFG.CONF_THRESHOLD = 'auto'
        else:
            raise NotImplementedError

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

    #set PSEUDOSHOTS_PERCENT config:
    if os.environ.get("PSEUDOSHOTS_PERCENT") is not None:
        obj_conf.Selector_CFG.PSEUDOSHOTS_PERCENT = eval(os.environ.get("PSEUDOSHOTS_PERCENT"))
    #set USE_SOFT_PARTIAL config:
    if os.environ.get("USE_SOFT_PARTIAL") is not None:
        obj_conf.PartialY_CFG.USE_SOFT_PARTIAL = eval(os.environ.get("USE_SOFT_PARTIAL"))
    #set TEMPERATURE config:
    if os.environ.get("TEMPERATURE") is not None:
        obj_conf.TEMPERATURE = eval(os.environ.get("TEMPERATURE"))
    if os.environ.get("UPDATE_FREQ") is not None:
        obj_conf.PartialY_CFG.UPDATE_FREQ = eval(os.environ.get("UPDATE_FREQ"))
        
    #set conf OUTPUT_DIR config:
    if os.environ.get("OUTPUT_DIR") is not None:
        obj_conf.OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
        makedirs(obj_conf.OUTPUT_DIR)
        
    return obj_conf


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    if args.use_candidate:
        with open(args.candidate_config_file, "r") as file:
            config = yaml.safe_load(file)
        # Cast configs to object From input arguments
        candiate_cfg = set_obj_conf(args, config)
        if candiate_cfg.OUTPUT_DIR:
            cfg.OUTPUT_DIR = candiate_cfg.OUTPUT_DIR 
    else:
        candiate_cfg = None

    return cfg, candiate_cfg


class lossmeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


def test(args, teloader, model):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_pl = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    one_hot_pl = []

    for i, (inputs) in enumerate(tqdm(teloader)):
        img = inputs["img"]
        labels = inputs["label"]

        if args.zero_shot:
            with torch.no_grad():
                output_pseudo_label = model(inputs.cuda(), zero_shot=True)
                _, predicted_pl = output_pseudo_label.max(1)
                one_hot_pl.append(predicted_pl.eq(labels.cuda()).cpu())
                acc1_pl = one_hot_pl[-1].sum().item() / len(labels)
                top1_pl.update(acc1_pl, len(labels))

        else:
            with torch.no_grad():
                inputs, labels = img.cuda(), labels.cuda()
                outputs = model(inputs, clip_eval=True)
                _, predicted = outputs.max(1)
                one_hot.append(predicted.eq(labels).cpu())
                acc1 = one_hot[-1].sum().item() / len(labels)
                top1.update(acc1, len(labels))

    if not args.zero_shot:
        return top1.avg * 100, top1_pl.avg * 100
    else:
        return top1_pl.avg * 100


def train_txt_cls(args, model):
    optimizer, _, _ = setup_text_training_utils(args, model)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for i in tqdm(range(args.txt_epochs)):
        loss = model.train_txt_clas(criteria)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.txt_cls_init()


def train_lafter_(args, model, 
                  tr_loader, val_loader, candiate_cfg, trainer):
    # Initialize the candidate trainer (CPL API) with the given configuration
    cand = CandidateTrainer(cfg=candiate_cfg, 
                            forward_method=model.forward_normal_for_pl,
                            transform_weak=val_loader.dataset.transform,
                            idx_to_label=trainer.lab2cname)

    # first train text classifier
    train_txt_cls(args, model)

    all_acc = list()
    optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    batch_time = lossmeter()
    data_time = lossmeter()

    # Prepare the training data with the candidate pseudolabels before starting the epochs
    criteria, train_data_new = cand.before_train(train_data=copy.deepcopy(tr_loader.dataset))
    tr_loader_new = trainer.create_new_loader_fromprevious(train_data_new)

    for epoch in range(1, args.epochs+1):
        print(f'Epoch: {epoch}')
        model.eval()
        model.adapter.train()
        end = time.time()

        for i, batch in enumerate((tr_loader_new)):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(model.device)

            optimizer.zero_grad()

            # pl = model.forward_normal_for_pl(input[0])
            out = model.forward_aug_with_prompts(input[1].float().cuda())

            # pseudo_label = F.softmax(pl, dim=-1)  # / 0.04
            # pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
            # pseudo_label = pseudo_label.flatten().cuda()
            # Compute the loss using the output, labels, and batch indices
            loss = criteria(out.squeeze(), batch["label"].to(model.device), batch["index"])
            detect_anomaly(loss)
            if i % args.print_freq == 0:
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {losses}\t"
                    "lr {lr:.6e}".format(
                        epoch,
                        args.epochs,
                        i + 1,
                        len(tr_loader_new),
                        losses=loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    ))

            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Evaluation: {epoch}')
        if epoch % 2 == 0:
            acc = test_prompting(val_loader, model)
            print(f'=====> TOP-1 Accuracy: {acc}')
            all_acc.append(acc)

        # After each epoch, update the training data and criteria with the candidate trainer
        criteria_, train_data_new_ = cand.after_epoch(
            train_data_all=copy.deepcopy(tr_loader.dataset), 
            train_data_current=copy.deepcopy(tr_loader_new.dataset),
            epoch=epoch
        )
        # If new criteria and training data are provided, update for the next epoch
        if criteria_ is not None:
            assert train_data_new_ is not None
            criteria = criteria_
            tr_loader_new = trainer.create_new_loader_fromprevious(train_data_new_)

    acc = test_prompting(val_loader, model)
    all_acc.append(acc)
    print(f'Testset accuracy: {acc}')
    print(f'-------------------------------- Best Accuracy: {max(all_acc)} ')



def train_lafter(args, model, tr_loader, val_loader):
    # first train text classifier
    train_txt_cls(args, model)

    all_acc = list()
    optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    batch_time = lossmeter()
    data_time = lossmeter()
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.adapter.train()
        end = time.time()

        for i, batch in enumerate((tr_loader)):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(model.device)

            optimizer.zero_grad()

            pl = model.forward_normal_for_pl(input[0])
            out = model.forward_aug_with_prompts(input[1].float().cuda())

            pseudo_label = F.softmax(pl, dim=-1)  # / 0.04
            pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
            pseudo_label = pseudo_label.flatten().cuda()

            loss = criteria(out.squeeze(), pseudo_label)
            if i % args.print_freq == 0:
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {losses}\t"
                    "lr {lr:.6e}".format(
                        epoch + 1,
                        args.epochs,
                        i + 1,
                        len(tr_loader),
                        losses=loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    ))

            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 2 == 0:
            print(f'Evaluation: {epoch}')
            acc = test_prompting(val_loader, model)
            print(f'=====> TOP-1 Accuracy: {acc}')
            all_acc.append(acc)
            
    acc = test_prompting(val_loader, model)
    all_acc.append(acc)
    print(f'Testset accuracy: {acc}')
    print(f'-------------------------------- Best Accuracy: {max(all_acc)} ')
    

def main(args):
    cfg, candiate_cfg = setup_cfg(args)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = cfg.DATASET.NAME
    setup_txt_epochs(args, dataset_name)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    print_args(args, cfg)
    print_args(cfg=candiate_cfg)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    model = trainer.model
    model.args = args
    test_loader = trainer.test_loader
    train_loader = trainer.train_loader_x

    if args.zero_shot:
        zero_shot(model, test_loader)
    elif args.use_candidate:
        train_lafter_(args, model, train_loader, test_loader, candiate_cfg=candiate_cfg, trainer=trainer)
    else:
        train_lafter(args, model, train_loader, test_loader)
        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=7777, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--print_freq", type=int, default=10, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--use_candidate",
        default=False,
        action="store_true",
        help="use candidate for tarining or not",
    )
    parser.add_argument(
        "--candidate-config-file",
        type=str,
        default="configs/trainers/candidate.yaml",
        help="path to config file for candidate setup",
    )
    parser.add_argument('--exp-name', type=str, required=False)
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--arch', type=str, default='ViT-B/32', required=False)
    parser.add_argument('--gpt_prompts', action='store_true')
    parser.add_argument('--text_prompts', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--txt_cls', type=str, default='tap', required=True, choices=['cls_only',
                                                                                      'templates_only', 'lafter', 'zero_shot'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--logfolder', default='logs', type=str)
    args = parser.parse_args()
    args.mile_stones = None
    main(args)

