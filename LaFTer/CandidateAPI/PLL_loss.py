import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy
from collections import Counter

class PLL_loss(nn.Module):
    """
    The loss functions in partial label learning (PLL).
    
    This class implements a customizable loss function for scenarios where each training instance
    might be associated with multiple candidate labels, with only one being the correct label.
    """
    softmax = nn.Softmax(dim=1)
    pred_label_dict = defaultdict(list)
    gt_label_dict: dict = {}

    def __init__(self, type=None, PartialY=None,
                 eps=1e-7, cfg=None):
        """Initializes the PLL_loss instance.
        
        Args:
            type (str, optional): Specifies the type of loss function to use.
            PartialY (Tensor, optional): The partial labels for the training instances.
            eps (float, optional): A small value to avoid division by zero or log(0).
            cfg (object, optional): Configuration object containing parameters for the loss function.
        """
        super(PLL_loss, self).__init__()
        self.eps = eps
        self.losstype = type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.T = self.cfg.TEMPERATURE

        if '_' in type:   # '_' in type means need to update conf
            self.conf = self.init_confidence(PartialY)

        if 'gce' in type:
            self.q = 0.7

        if 'lw' in type:
            self.lw_weight = 2; self.lw_weight0 = 1


    def init_confidence(self, PartialY):       
        tempY = PartialY.sum(dim=1, keepdim=True).repeat(1, PartialY.shape[1])
        confidence = (PartialY/tempY).float()
        confidence = confidence.to(self.device)
        return confidence
    
    def forward(self, *args, reduce=True):
        """
        Computes the loss based on the specified loss type.
        
        Args:
            *args: Arguments required by the specific loss computation method.
            reduce (bool): If True, returns the mean of the loss; otherwise, returns the loss for each instance.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        if self.losstype == 'cc':
            loss = self.forward_cc(*args)
        elif self.losstype == 'ce':
            loss = self.forward_ce(*args)
        elif self.losstype == 'gce':
            loss = self.forward_gce(*args)
        elif self.losstype in ['rc_rc', 'rc_cav',]:
            loss = self.forward_rc(*args)
        elif self.losstype in ['lw_lw',]:
            loss = self.forward_lw(*args)
        else:
            raise ValueError
        if reduce:
            loss = loss.mean()
        return loss
    

    def forward_gce(self, x, y, index=None):
        """y is shape of (batch_size, num_classes (0 ~ 1.)), one-hot vector"""
        p = F.softmax(x, dim=1)      #outputs are logits
        # Create a tensor filled with a very small number to represent 'masked' positions
        masked_p = p.new_full(p.size(), float('-inf'))
        # Apply the mask
        masked_p[y.bool()] = p[y.bool()] + self.eps         
        # Adjust masked positions to avoid undefined gradients by adding epsilon
        masked_p[y.bool()] = (1 - masked_p[y.bool()] ** self.q) / self.q
        masked_p[~y.bool()] = self.eps 
        loss = masked_p.sum(dim=1)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")
        return loss
    
    def forward_cc(self, x, y, index=None):
        sm_outputs = F.softmax(x, dim=1)
        final_outputs = sm_outputs * y
        loss = - torch.log(final_outputs.sum(dim=1))
        return loss
    
    def forward_ce(self, x, y, index=None):
        sm_outputs = F.log_softmax(x, dim=1)
        final_outputs = sm_outputs * y
        loss = - final_outputs.sum(dim=1)
        return loss
    
    def forward_rc(self, x, y, index):
        logsm_outputs = F.log_softmax(x, dim=1)  #x is the model ouputs
        final_outputs = logsm_outputs * self.conf[index, :]
        loss = - final_outputs.sum(dim=1)  
        return loss     

    def forward_lw(self, x, y, index):
        # (onezero, counter_onezero)
        # onezero: partial label
        # counter_onezero: non-partial label
        onezero = torch.zeros(x.shape[0], x.shape[1])
        onezero[y > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(self.device)
        counter_onezero = counter_onezero.to(self.device)
        sm_outputs = F.softmax(x, dim=1)

        sig_loss1 = - torch.log(sm_outputs + self.eps)
        l1 = self.conf[index, :] * onezero * sig_loss1
        average_loss1 = l1.sum(dim=1) 

        sig_loss2 = - torch.log(1 - sm_outputs + self.eps)
        l2 = self.conf[index, :] * counter_onezero * sig_loss2      
        average_loss2 = l2.sum(dim=1) 

        average_loss = self.lw_weight0 * average_loss1 + self.lw_weight * average_loss2
        return average_loss


    def check_conf_update(self, images, y, index, output=None):
        """
        Checks if confidence update is needed based on the loss type and updates accordingly.
        
        Args:
            images (Tensor): The input images.
            y (Tensor): The partial labels for the batch.
            index (Tensor): The indices of the batch samples.
            output (Tensor, optional): The output logits from the model. Required for some conf_types.
        """
        if '_' in self.losstype:
            conf_type = self.losstype.split('_')[-1]
            assert conf_type in ['rc', 'cav', 'lw'], 'conf_type is not supported'
            self.update_confidence(y, index, conf_type, outputs=output)
        else:
            return

    @torch.no_grad()
    def update_confidence(self, PL_labels, batch_idxs, conf_type, outputs):
        """
        Updates the confidence scores based on the specified confidence type.
        
        Args:
            PL_labels (Tensor): The partial labels for the batch.
            batch_idxs (Tensor): The indices of the batch samples.
            conf_type (str): The type of confidence update to perform.
            outputs (Tensor): The output logits from the model.
        """
        if conf_type == 'rc':
            rc_conf = self.cal_pred_conf(outputs, PL_labels, conf_type)
            base_value = rc_conf.sum(dim=1).unsqueeze(1).repeat(1, rc_conf.shape[1])
            self.conf[batch_idxs, :] = rc_conf / base_value  # use maticx for element-wise division

        elif conf_type == 'cav':
            cav_conf = self.cal_pred_conf(outputs, PL_labels, conf_type)
            cav_pred = torch.max(cav_conf, dim=1)[1]
            gt_label = F.one_hot(cav_pred, PL_labels.shape[1])
            self.conf[batch_idxs, :] = gt_label.float()

        elif conf_type == 'lw':
            lw_conf = self.cal_pred_conf(outputs, PL_labels, conf_type)
            new_weight1, new_weight2 = lw_conf
            new_weight1 = new_weight1 / (new_weight1 + self.eps).sum(dim=1).repeat(
                    self.conf.shape[1], 1).transpose(0, 1)
            new_weight2 = new_weight2 / (new_weight2 + self.eps).sum(dim=1).repeat(
                    self.conf.shape[1], 1).transpose(0, 1)
            new_weight = (new_weight1 + new_weight2) 
            self.conf[batch_idxs, :] = new_weight


    @torch.no_grad()
    def cal_pred_conf(self, logits, PL_labels, conf_type, lw_return2=True):
        """
        Calculates the predicted confidence scores based on the specified type.
        
        Args:
            logits (Tensor): The output logits from the model.
            PL_labels (Tensor): The partial labels for the batch.
            conf_type (str): The type of confidence calculation to perform.
            lw_return2 (bool, optional): If True, returns two weights for 'lw' type.
        
        Returns:
            Tensor: The calculated confidence scores.
        """
        if conf_type == 'rc':
            conf = F.softmax(logits, dim=1)
            conf = conf * PL_labels
        elif conf_type == 'cav':
            conf = (logits * torch.abs(1 - logits)) 
            conf = conf * PL_labels
        elif conf_type == 'cc':
            conf = F.softmax(logits, dim=1)
            conf = conf * PL_labels
        elif conf_type == 'lw':
            sm_outputs = F.softmax(logits, dim=1)
            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[PL_labels > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.to(self.device)
            counter_onezero = counter_onezero.to(self.device)

            new_weight1 = sm_outputs * onezero
            new_weight2 = sm_outputs * counter_onezero
            if lw_return2:
                conf = (new_weight1, new_weight2)
            else:
                conf = new_weight1

        return conf


    def clean_conf(self):
        """
        Cleans the confidence scores by setting very low values to zero and normalizing.
        """
        if hasattr(self, 'conf'):
            self.conf = torch.where(self.conf < self.eps*10, 
                                    torch.zeros_like(self.conf), 
                                    self.conf)
            base_value = self.conf.sum(dim=1).unsqueeze(1).repeat(1, self.conf.shape[1])
            self.conf = self.conf / base_value

