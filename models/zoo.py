# models/zoo.py
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
from .moco import vit_base as moco_base
import numpy as np
import copy
from timm.models.layers import trunc_normal_, DropPath
import random
import math
from operator import mul
from functools import reduce

class SharedGate(nn.Module):
    def __init__(self, emb_d, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_d, emb_d // 4),
            nn.ReLU(),
            nn.Linear(emb_d // 4, num_layers),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class APT(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, ema_coeff):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.n_tasks = n_tasks
        self._init_smart(prompt_param)

        self.merge_flag = True
        self.ema_coeff = ema_coeff

        # Khởi tạo prompt_tokens
        self.prompt_tokens = create_prompt_with_init(12*2, emb_d) 
        global_merged_prompt = torch.zeros(12*2, emb_d).cuda()
        self.register_buffer('global_merged_prompt', global_merged_prompt.clone().detach()) 
        
        # Shared Gate
        self.gate_net = SharedGate(emb_d, 12)
        
        # Buffer lưu giá trị cổng trung bình
        self.register_buffer('avg_gate_values', torch.zeros(12))
        self.current_gate_values = None 

        trunc_normal_(self.prompt_tokens, std=0.02)

    def _init_smart(self, prompt_param):
        self.prompt_dropout_ratio = float(prompt_param[0])
        self.prompt_dropout = nn.Dropout(self.prompt_dropout_ratio)

    def process_task_count(self):
        self.task_count += 1

    def forward(self, l, x_block, train=False):
        B, N, D = x_block.shape

        # Tính toán Gate tại layer 0
        if l == 0:
            cls_token = x_block[:, 0]
            self.current_gate_values = self.gate_net(cls_token)
        
        current_g = self.current_gate_values[:, l].view(B, 1)

        # Chọn prompt
        if train or not self.merge_flag:
            prompt_k = self.prompt_tokens[l*2]
            prompt_v = self.prompt_tokens[l*2 + 1]
        else:
            prompt_k = self.global_merged_prompt[l*2]
            prompt_v = self.global_merged_prompt[l*2 + 1]

        # Áp dụng Gate
        prompt_k = prompt_k.unsqueeze(0) * current_g 
        prompt_v = prompt_v.unsqueeze(0) * current_g

        # Reshape sang Multi-head (12 heads * 64 dims)
        P_root_k = prompt_k.reshape(B, 12, 1, 64)
        P_root_v = prompt_v.reshape(B, 12, 1, 64)

        P_k = torch.cat((P_root_k, torch.zeros((B, 12, N-1, 64), device=x_block.device)), dim=-2)
        P_v = torch.cat((P_root_v, torch.zeros((B, 12, N-1, 64), device=x_block.device)), dim=-2)
        
        return [P_k, P_v]
    
    @torch.no_grad()
    def priority_fusion(self):
        print("Executing Priority Fusion based on Gate Importance...")
        for l in range(12):
            alpha = self.avg_gate_values[l]
            idx_k, idx_v = l*2, l*2 + 1
            self.global_merged_prompt[idx_k] = alpha * self.global_merged_prompt[idx_k] + \
                                               (1 - alpha) * self.prompt_tokens[idx_k]
            self.global_merged_prompt[idx_v] = alpha * self.global_merged_prompt[idx_v] + \
                                               (1 - alpha) * self.prompt_tokens[idx_v]

def create_prompt_with_init(a, b, c=None, ortho=False, mean=None, std=None, init_ref=None):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    elif init_ref is not None:
        p = torch.nn.Parameter(init_ref.squeeze(dim=0).expand(a, b),  requires_grad=True)
    elif mean and std:
        nn.init.normal_(p, mean=mean, std=std)
    else:
        nn.init.uniform_(p)
    return p

class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, ema_coeff=0.5, pt=False, prompt_flag=False, prompt_param=None, tasks=[]):
        super(ViTZoo, self).__init__()
        self.num_classes = num_classes
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.tasks = tasks

        # 1. Khởi tạo backbone
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0, drop_path_rate=0)
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict, strict=False)
        else:
            zoo_model = None
        
        self.feat = zoo_model

        # 2. Khởi tạo APT module và gán vào Backbone
        if self.prompt_flag == 'apt':
            # Chú ý: Truyền tham số đúng từ prompt_param
            self.apt = APT(768, n_tasks=len(tasks), prompt_param=prompt_param, ema_coeff=ema_coeff)
            # Quan trọng: Gán apt vào backbone để vit.py có thể truy cập qua prompt.forward
            self.feat.apt = self.apt 
        else:
            self.apt = None

        # 3. Classifier
        self.last = nn.Linear(768, num_classes) 
        self.clf_norm = nn.LayerNorm(768)

        # 4. Thiết lập đóng băng/mở khóa tham số
        # Mở khóa các tham số của APT (bao gồm gate_net và prompt_tokens) và head
        for name, param in self.named_parameters():
            if "apt" in name or "last" in name or "clf_norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x, train=False):
        # Truyền module apt vào hàm forward của backbone (VisionTransformer)
        if self.apt is not None:
            out = self.feat(x, prompt=self.apt, train=train)
        else:
            out = self.feat(x, train=train)
            
        out = out[:, 0, :] # Lấy CLS token
        out = self.clf_norm(out)
        
        # Norm head (Cosine similarity classifier)
        wt_norm = F.normalize(self.last.weight, p=2, dim=1) 
        out = torch.matmul(out, wt_norm.t())
        return out

class MoCoZoo(ViTZoo):
    def __init__(self, num_classes=10, ema_coeff=0.5, pt=False, prompt_flag=False, prompt_param=None, tasks=[]):
        super(MoCoZoo, self).__init__(num_classes, ema_coeff, pt, prompt_flag, prompt_param, tasks)
        if pt:
            zoo_model = moco_base()
            ckpt = "/share/ckpt/cgn/vpt/model/mocov3_linear-vit-b-300ep.pth.tar"
            checkpoint = torch.load(ckpt, map_location="cpu")
            load_dict = checkpoint['state_dict']
            for k in list(load_dict.keys()):
                if k.startswith('module.'):
                    load_dict[k[len("module."):]] = load_dict[k]
                del load_dict[k]
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict, strict=False)
            self.feat = zoo_model
            # Re-assign apt to the new feat if exists
            if self.apt is not None:
                self.feat.apt = self.apt

def vit_pt_imnet(out_dim, ema_coeff, tasks=[], prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, ema_coeff=ema_coeff, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)

def moco_pt(out_dim, ema_coeff, tasks=[], prompt_flag = 'None', prompt_param=None):
    return MoCoZoo(num_classes=out_dim, ema_coeff=ema_coeff, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)