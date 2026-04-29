#prompt.py
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from utils.metric import accuracy, AverageMeter, Timer
from .default import NormalNN, weight_reset, accumulate_acc
from utils.schedulers import CosineSchedule

class Prompt_Learner(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']
        # Hyperparameters for additional losses
        self.orthogonal_weight = learner_config.get('orthogonal_weight', 0.01)
        self.contrastive_weight = learner_config.get('contrastive_weight', 0.1)
        self.temperature = learner_config.get('temperature', 0.1)
        print(f"[DEBUG Prompt_Learner __init__] orthogonal_weight={self.orthogonal_weight}, contrastive_weight={self.contrastive_weight}, temperature={self.temperature}")
        super(Prompt_Learner, self).__init__(learner_config)

    def orthogonal_loss(self):
        """Encourage prompts from different tasks to be orthogonal."""
        if not hasattr(self.model, 'prompt') or not hasattr(self.model.prompt, 'prompts'):
            return torch.tensor(0.0, device=self.config['gpuid'][0] if self.gpu else 'cpu')
        
        prompts = self.model.prompt.prompts  # ParameterList
        if len(prompts) <= 1:
            return torch.tensor(0.0, device=self.config['gpuid'][0] if self.gpu else 'cpu')
        
        # Compute Gram matrix of prompt similarities (normalized)
        num_tasks = len(prompts)
        prompt_vectors = []
        for p in prompts:
            prompt_vectors.append(p.data.flatten())
        
        # Stack and compute correlation
        stacked = torch.stack(prompt_vectors, dim=0)  # (num_tasks, dim)
        # Normalize before computing Gram
        stacked_norm = F.normalize(stacked, dim=1)
        gram = torch.mm(stacked_norm, stacked_norm.t())  # (num_tasks, num_tasks)
        
        # Target: diagonal should be 1 (self-similarity), off-diagonal should be 0
        target = torch.eye(num_tasks, device=gram.device)
        loss = F.mse_loss(gram, target)
        
        return loss * self.orthogonal_weight

    def contrastive_loss(self, inputs, targets):
        """Contrastive loss on CLS embeddings to improve task discrimination."""
        if not hasattr(self.model, 'prompt'):
            return torch.tensor(0.0, device=self.config['gpuid'][0] if self.gpu else 'cpu')
        
        # Get CLS embeddings
        with torch.no_grad():
            self.model.eval()
            features = self.model.feat(inputs)[:, 0, :]  # CLS token
            self.model.train()
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix with higher temperature for stability
        temp = max(self.temperature, 0.5)  # Use at least 0.5 for stability
        sim_matrix = torch.mm(features, features.t()) / temp
        
        # Create positive mask (same target) and negative mask (different target)
        labels = targets.unsqueeze(0)
        pos_mask = (labels == labels.t()).float()
        neg_mask = 1 - pos_mask
        
        # Mask out diagonal (self-contrast)
        diag_mask = torch.eye(pos_mask.size(0), device=pos_mask.device)
        pos_mask = pos_mask - diag_mask
        
        # InfoNCE loss with numerical stability
        exp_sim = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0])  # subtract max for stability
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)
        
        # Avoid division by zero
        denominator = pos_sum + neg_sum
        denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        
        loss = -torch.log(pos_sum / denominator + 1e-8)
        loss = loss.mean() * self.contrastive_weight
        
        return loss

    def update_model(self, inputs, targets):
        # logits
        logits = self.model(inputs, train=True)
        
        logits = logits[:,:self.valid_out_dim]
        logits[:,:self.last_valid_out_dim] = -float('inf')
        ce_loss = self.criterion(logits, targets.long())
        
        # Additional losses
        ortho_loss = self.orthogonal_loss()
        cont_loss = self.contrastive_loss(inputs, targets)
        
        # Debug: Print detailed loss info
        print(f"  [DEBUG Loss] CE: {ce_loss.item():.6f}, Ortho: {ortho_loss.item():.6f} (w={self.orthogonal_weight}), Cont: {cont_loss.item():.6f} (w={self.contrastive_weight})")
        
        # Total loss
        total_loss = ce_loss + ortho_loss + cont_loss
        
        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Debug: Check if prompts have gradients
        if hasattr(self.model, 'prompt') and hasattr(self.model.prompt, 'prompts'):
            prompt_grad_norm = 0
            for p in self.model.prompt.prompts:
                if p.grad is not None:
                    prompt_grad_norm += p.grad.norm().item()
            print(f"  [DEBUG] Prompt grad norm: {prompt_grad_norm:.6f}")
        
        self.optimizer.step()
        
        return total_loss.detach(), logits

    def get_attn_heatmap(self, inputs):
        return 

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            if isinstance(self.schedule, (list, tuple)):
                K = self.schedule[-1] if len(self.schedule) > 0 else 1
            else:
                K = self.schedule
            self.scheduler = CosineSchedule(self.optimizer, K=K)
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

class APT_Learner(Prompt_Learner):

    def __init__(self, learner_config):
        super(APT_Learner, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, ema_coeff=self.ema_coeff, prompt_flag = 'apt', prompt_param=self.prompt_param, tasks=self.tasks)
        return model

