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
        self.num_layers = 12
        # Khởi tạo prompt_tokens
        self.prompt_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(self.prompt_token_number, emb_d)) 
            for _ in range(self.num_layers * 2)
        ])

        for p in self.prompt_tokens:
            trunc_normal_(p, std=0.02)

        self.register_buffer('global_merged_prompt', torch.zeros(self.num_layers * 2, self.prompt_token_number, emb_d))
        #self.prompt_tokens = create_prompt_with_init(12*2, emb_d) 
        #global_merged_prompt = torch.zeros(12*2, emb_d).cuda()
        #self.register_buffer('global_merged_prompt', global_merged_prompt.clone().detach()) 
        
        # Shared Gate
        self.gate_net = SharedGate(emb_d, self.num_layers)
        
        # Buffer lưu giá trị cổng trung bình
        self.register_buffer('avg_gate_values', torch.zeros(self.num_layers))
        self.current_gate_values = None 

        #trunc_normal_(self.prompt_tokens, std=0.02)

    def _init_smart(self, prompt_param):
        # 1. Hàm phụ để lấy tất cả các giá trị số ra khỏi các lớp list bọc nhau
        def flatten_params(nested_list):
            flat_list = []
            if not isinstance(nested_list, (list, tuple)):
                return [nested_list]
            for item in nested_list:
                if isinstance(item, (list, tuple)):
                    flat_list.extend(flatten_params(item))
                else:
                    flat_list.append(item)
            return flat_list

        # 2. Làm phẳng và ép kiểu
        try:
            temp_list = flatten_params(prompt_param)
            # Lọc bỏ các giá trị không phải là số hoặc chuỗi số (nếu có)
            # Và chỉ lấy 3 giá trị cuối cùng (thường là 10, 0.1, 768)
            p_param = [float(i) for i in temp_list if str(i).replace('.','',1).isdigit()]
            
            # Nếu sau khi lọc mà có nhiều hơn 3 số, ta lấy 3 số cuối cùng 
            # vì số 10 dư thừa thường nằm ở đầu do lỗi argparse
            if len(p_param) > 3:
                p_param = p_param[-3:]
                
            self.prompt_token_number = int(p_param[0]) 
            self.prompt_dropout_ratio = p_param[1]      
            self.prompt_len = int(p_param[2])
            
        except Exception as e:
            print(f"Lỗi định dạng prompt_param gốc: {prompt_param}")
            print(f"Dữ liệu sau khi làm phẳng: {temp_list}")
            raise e
        
        # 3. Kiểm tra an toàn cho Dropout
        if self.prompt_dropout_ratio > 1.0:
            self.prompt_dropout_ratio /= 100.0
            
        self.prompt_dropout = nn.Dropout(self.prompt_dropout_ratio)
    def process_task_count(self):
        self.task_count += 1

    def forward(self, l, x_block, train=False):
        B, N, D = x_block.shape

        # Tính toán Gate tại layer 0
        if l == 0 or self.current_gate_values is None:
            cls_token = x_block[:, 0]
            self.current_gate_values = self.gate_net(cls_token)
        
        current_g = self.current_gate_values[:, l].view(B, 1, 1)
        idx_k, idx_v = l*2, l*2 + 1
        # Chọn prompt
        if train or not self.merge_flag:
            prompt_k = self.prompt_tokens[idx_k]
            prompt_v = self.prompt_tokens[idx_v]
        else:
            prompt_k = self.global_merged_prompt[idx_k]
            prompt_v = self.global_merged_prompt[idx_v]

        # Áp dụng Gate
        prompt_k = prompt_k.unsqueeze(0) * current_g 
        prompt_v = prompt_v.unsqueeze(0) * current_g

        # Reshape sang Multi-head (12 heads * 64 dims)
        P_root_k = prompt_k.reshape(B, 10, 12, 64).permute(0, 2, 1, 3)
        P_root_v = prompt_v.reshape(B, 10, 12, 64).permute(0, 2, 1, 3)

        padding_k = torch.zeros((B, 12, N - 10, 64), device=x_block.device)
        padding_v = torch.zeros((B, 12, N - 10, 64), device=x_block.device)
        
        P_k = torch.cat((P_root_k, padding_k), dim=-2)
        P_v = torch.cat((P_root_v, padding_v), dim=-2)
        
        return [P_k, P_v]
    
    @torch.no_grad()
    def priority_fusion(self):
        print("Executing Priority Fusion based on Gate Importance...")
        for l in range(self.num_layers):
            alpha = self.avg_gate_values[l]
            idx_k, idx_v = l*2, l*2 + 1

            current_p_k = self.prompt_tokens[idx_k]
            current_p_v = self.prompt_tokens[idx_v]

            new_k = alpha * self.global_merged_prompt[idx_k] + (1 - alpha) * current_p_k
            self.global_merged_prompt[idx_k].copy_(new_k)    
            
            new_v = alpha * self.global_merged_prompt[idx_v] + (1 - alpha) * current_p_v
            self.global_merged_prompt[idx_v].copy_(new_v)

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
            # if self.feat is not None:
            self.prompt = self.apt 
        else:
            self.apt = None
            self.prompt = None

        # 3. Classifier
        self.last = nn.Linear(768, num_classes) 
        self.clf_norm = nn.LayerNorm(768)

        self._ensure_apt_attached()
        # 4. Thiết lập đóng băng/mở khóa tham số
        # Mở khóa các tham số của APT (bao gồm gate_net và prompt_tokens) và head
        for name, param in self.named_parameters():
            if "apt" in name or "last" in name or "clf_norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _ensure_apt_attached(self):
        """Đảm bảo backbone (feat) luôn mang theo module apt"""
        if self.apt is not None and self.feat is not None:
            self.feat.apt = self.apt
    def forward(self, x, train=False):
        self._ensure_apt_attached()
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
            self._ensure_apt_attached()

def vit_pt_imnet(out_dim, ema_coeff, tasks=[], prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, ema_coeff=ema_coeff, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)

def moco_pt(out_dim, ema_coeff, tasks=[], prompt_flag = 'None', prompt_param=None):
    return MoCoZoo(num_classes=out_dim, ema_coeff=ema_coeff, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)