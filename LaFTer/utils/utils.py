import logging
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
logger_initialized = {}
import getpass
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import random
from pathlib import Path


lafter_datasets = ['DescribableTextures',  'EuroSAT', 'OxfordFlowers', 'SUN397', 'UCF101', 'ImageNetR', 'ImageNetSketch',
                   'ImageNetA', 'CIFAR10_local', 'CIFAR100_local', 'ImageNet', 'Caltech101']


def setup_text_training_utils(args, model):
    model = model.cuda()
    model = model.float()
    params = list()
    mile_stones = args.mile_stones

    for key, value in model.named_parameters():
        if 'adapter' in key and 'adapter_pl' not in key:
            value.requires_grad = True
        else:
            value.requires_grad = False

    print('------------------ Learnable Parameters ------------------')
    for key, value in model.named_parameters():
        if value.requires_grad:
            print("\t{}, {}, {}".format(key, value.numel(), value.shape))
            params.append((key, value))
    print('----------------------------------------------------------')

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999))

    if args.scheduler == 'coslr':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=args.epochs,
                                      lr_min=1e-6,
                                      warmup_lr_init=1e-4,
                                      warmup_t=5,
                                      cycle_limit=1,
                                      t_in_epochs=True)
    elif args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, 0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs))
    else:
        raise NotImplementedError

    criteria = LabelSmoothingCrossEntropy()
    return optimizer, scheduler, criteria

def setup_lafter_training_utils(args, model):
    model = model.cuda()
    model = model.float()
    params = list()
    for key, value in model.named_parameters():
        if key == 'prompt_embeddings':
            value.requires_grad = True
        elif 'adapter' in key and 'adapter_pl' not in key:
            value.requires_grad = True
        elif 'projector' in key and not args.entropy:
            value.requires_grad = True
        elif 'ln' in key:
            value.requires_grad = True
        else:
            value.requires_grad = False

    for key, value in model.named_parameters():
        if 'visual' in key:
            if 'ln' in key or 'bn' in key:
                value.requires_grad = True
            else:
                value.requires_grad = False

    print('------------------ Learnable Parameters ------------------')
    for key, value in model.named_parameters():
        if value.requires_grad:
            print("\t{}, {}, {}".format(key, value.numel(), value.shape))
            params.append((key, value))
    print('----------------------------------------------------------')

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.mile_stones, 0.60)
    criteria = LabelSmoothingCrossEntropy()
    return optimizer, scheduler, criteria
def test_prompting(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, inputs in enumerate(tqdm(teloader)):
        labels = inputs['label']
        inputs = inputs['img']
        if isinstance(inputs, list):
            inputs = inputs[0]
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.eval_clip(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100

text_cls_epochs = {
    'DescribableTextures': 400, # 5.5k for txt_cls
    'EuroSAT': 400,
    'FGVCAircraft': 500,
    'Food101': 400,
    'CIFAR10_local': 400,
    'CIFAR100_local': 400, # 4k for txt_cls
    'ImageNet': 500,
    'OxfordFlowers': 600, # 600 for txt_cls
    'SUN397': 500, # 2k for txt_cls
    'UCF101': 400,
    'ImageNetR': 500, # 4k for txt_cls
    'ImageNetA': 500, # 4k for txt_cls
    'ImageNetSketch': 500, # 4k for txt_cls
    'Caltech101': 500, # 4k for txt_cls
}

def setup_txt_epochs(args, dataset):
    args.txt_epochs = text_cls_epochs[dataset]


def get_env_id():
    return getpass.getuser()


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DINOLoss(nn.Module):
    def __init__(self, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        # teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            loss = torch.sum(-q * F.log_softmax(student_out, dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


def setup_log_folder(args):
    Path(args.logfolder).mkdir(exist_ok=True, parents=True)
    args.logfile = args.logfolder + f'/{time.strftime("%Y%m%d_%H%M%S")}.txt'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def zero_shot(model, loader):
    print('-------------- ZERO SHOT INFERENCE --------------')
    total = 0.
    correct_base = 0.
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            target = inputs['label']
            images = inputs['img']
            if isinstance(images, list):
                images = images[0]

            images = images.cuda()
            target = target.cuda()
            out = model(images)
            logits_base = out
            pred_base = torch.argmax(logits_base, dim=1)
            for j in range(len(target)):
                total += 1.
                if pred_base[j] == target[j]:
                    correct_base += 1.
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy standard: {top1:.2f}")


def test(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, (inputs, labels) in enumerate(tqdm(teloader)):
        if isinstance(inputs, list):
            inputs = inputs[0]

        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.eval_clip(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100


def update_ema_variables(model, alpha_teacher):
    teacher_prompt_param = []
    student_prompt_param = []

    for key, value in model.named_parameters():
        if key == 'prompt_embeddings':
            student_prompt_param.append(value)

        elif key == 'prompt_embeddings_teacher':
            teacher_prompt_param.append(value)

    for ema_param, param in zip(teacher_prompt_param, student_prompt_param):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[
                                                                                         :]

    for k, v in model.named_parameters():
        if k == 'prompt_embeddings_teacher':
            v = ema_param

    # return ema_model


def update_ema_variables_sanity(ema_model, model, alpha_teacher):
    for kv_ema, kv_student in zip(ema_model.named_parameters(), model.named_parameters()):
        if 'ln' in kv_ema[0] and 'ln' in kv_student[0]:
            kv_ema[1].data[:] = alpha_teacher * kv_ema[1][:].data[:] + (1 - alpha_teacher) * kv_student[1][:].data[:]
    return ema_model


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            last_epoch=-1,
            verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            cons_lr,
            last_epoch=-1,
            verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


