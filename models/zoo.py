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


class APT(nn.Module):
    def __init__(self, emb_d, initial_components=10, key_dim=768, num_layers=12):

        super().__init__()
        self.task_count = 0
        self.num_layers = num_layers
        self.key_dim = key_dim
        self.emb_d = emb_d
        
        # Lưu danh sách các Parameter chứa Prompts của các component
        # Mỗi component gồm: prompt_k và prompt_v cho toàn bộ 12 layers
        self.components_k = nn.ModuleList()
        self.components_v = nn.ModuleList()

        # khởi tạo M components đầu tiên
        self.add_new_components(initial_components)

        # Khởi tạo Pool chứa Keys tương ứng với các component [M, key_dim]
        self.keys = nn.Parameter(torch.randn(initial_components, key_dim))
        trunc_normal_(self.keys, std=0.02)

        # Vector chú ý A dùng để ⊙ với query
        self.A = nn.Parameter(torch.ones(1, key_dim))

        # Đánh dấu số lượng component cũ để phục vụ việc freeze
        self.num_old_components = 0

    def add_new_components(self, M):
        """Hàm sinh thêm M components mới vào pool khi có task mới"""
        for _ in range(M):
            # Tạo prompt cho Key và Value: kích thước [num_layers, 1, emb_d]
            p_k = nn.Parameter(torch.FloatTensor(self.num_layers, 1, self.emb_d))
            p_v = nn.Parameter(torch.FloatTensor(self.num_layers, 1, self.emb_d))
            trunc_normal_(p_k, std=0.02)
            trunc_normal_(p_v, std=0.02)
            self.components_k.append(nn.ParameterDict({'param': p_k}))
            self.components_v.append(nn.ParameterDict({'param': p_v}))
   
    def freeze_old_components(self, M_new=10):
        """Hàm gọi đầu mỗi Task mới (ngoại trừ task 0) để đóng băng components cũ"""
        self.task_count += 1
        # 1. Đóng băng tất cả các component hiện tại
        for param in self.components_k.parameters():
            param.requires_grad = False
        for param in self.components_v.parameters():
            param.requires_grad = False
            
        self.num_old_components = len(self.components_k)
        
        # 2. Tạo M_new components mới cho task này
        self.add_new_components(M_new)
        
        # 3. Mở rộng ma trận Keys (giữ lại gradient của keys mới, đóng băng keys cũ)
        old_keys = self.keys.data
        new_keys = torch.randn(M_new, self.key_dim).to(old_keys.device)
        trunc_normal_(new_keys, std=0.02)
        self.keys = nn.Parameter(torch.cat([old_keys, new_keys], dim=0))

    def progressive_prompt_fusion(self, beta=0.9):
        """Hàm PPF thực hiện sau khi kết thúc một task"""
        with torch.no_grad():
            num_total = len(self.components_k)
            num_new = num_total - self.num_old_components
            if self.num_old_components > 0 and num_new > 0:
                # Ép tri thức từ component mới học vào component cũ (Ví dụ phối hợp tuần hoàn cyclic)
                for i in range(self.num_old_components):
                    idx_new = self.num_old_components + (i % num_new)
                    c_old_k = self.components_k[i]['param']
                    c_new_k = self.components_k[idx_new]['param']
                    c_old_k.copy_(beta * c_old_k + (1 - beta) * c_new_k)
                    
                    c_old_v = self.components_v[i]['param']
                    c_new_v = self.components_v[idx_new]['param']
                    c_old_v.copy_(beta * c_old_v + (1 - beta) * c_new_v)

    def forward(self, cls_token):
        """
        Nhận CLS token đầu vào làm Query, tính toán trọng số alpha và tổ hợp Prompt.
        cls_token: [B, emb_d]
        """
        B = cls_token.shape[0]
        
        # 1. Tính toán Query: query = CLS ⊙ A
        query = cls_token * self.A  # [B, key_dim]
        
        # 2. Tính Cosine Similarity giữa query và TẤT CẢ các keys trong pool
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(self.keys, p=2, dim=-1)
        sim = torch.matmul(query_norm, keys_norm.t())  # [B, Total_components]
        
        # 3. Tính trọng số alpha = softmax(sim)
        alpha = F.softmax(sim, dim=-1)  # [B, Total_components]
        
        # 4. Tổ hợp Prompts từ weighted sum của các components
        # Gom tất cả các component parameter thành một Tensor lớn
        all_p_k = torch.stack([c['param'] for c in self.components_k], dim=0) # [Total_components, 12, 1, emb_d]
        all_p_v = torch.stack([c['param'] for c in self.components_v], dim=0) # [Total_components, 12, 1, emb_d]
        
        # Chuẩn bị alpha cho việc broadcasting: [B, Total_components, 1, 1, 1]
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Thực hiện phép tính Σ(α_m * P^m)
        P_k_total = torch.sum(alpha_expanded * all_p_k.unsqueeze(0), dim=1) # [B, 12, 1, emb_d]
        P_v_total = torch.sum(alpha_expanded * all_p_v.unsqueeze(0), dim=1) # [B, 12, 1, emb_d]
        
        return P_k_total, P_v_total, alpha

# note - ortho init has not been found to help l2p/dual prompt
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
        # get last layer

        self.prompt_flag = prompt_flag
        self.task_id = None
    
        self.tasks = tasks

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict,strict=False)
        else:
            pass
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
     
        #classifier
        self.last = nn.Linear(768, num_classes) 
        self.clf_norm = nn.LayerNorm(768)

        # create prompting module
        if self.prompt_flag == 'apt':
            self.prompt = APT(768, initial_components=10)
        else:
            self.prompt = None

        if self.prompt_flag == "apt":
            tuned_params = [
            "clf_norm.weight","clf_norm.bias",
            "prompt.prompt_tokens",
            "last.weight",
            "last.bias", 
            ] 
        else:
            tuned_params = [
            "clf_norm.weight","clf_norm.bias",
            "last.weight",
            "last.bias", 
            ]

        for name, param in self.named_parameters():
            if name in tuned_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
           

    def get_attn_score_within_heads(self, attn_matrix, dim, method="mean"):
        if method == "mean":
            return attn_matrix.mean(dim=dim)

        elif method == "max":
            return attn_matrix.max(dim=dim)[0]
 
    def forward(self, x, train=False):
        if self.prompt is not None:
            if self.prompt_flag == 'apt':
                out = self.feat(x, prompt=self.prompt, train=train)
                out =  out[:,0,:]
            else: 
                raise ValueError("prompt flag not supported")
               
        else:
            out, _, _ = self.feat(x, train=train)
            out = out[:,0,:]

        out = self.clf_norm(out)
        wt_norm = F.normalize(self.last.weight, p=2, dim=1) 
        out = torch.matmul(out, wt_norm.t())

        return out
   

class MoCoZoo(ViTZoo):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, tasks=[]):
        super(MoCoZoo, self).__init__(num_classes, pt, prompt_flag, prompt_param, tasks)
       
        if pt:
            zoo_model = moco_base()#VisionTransformerMoCo(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                     #   num_heads=12,
                                    #    drop_path_rate=0
                                   #     )
            ckpt = "/share/ckpt/cgn/vpt/model/mocov3_linear-vit-b-300ep.pth.tar"

            checkpoint = torch.load(ckpt, map_location="cpu")
            load_dict = checkpoint['state_dict']
            for k in list(load_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    load_dict[k[len("module."):]] = load_dict[k]
                # delete renamed or unused k
                del load_dict[k]

            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict, strict=False)

        else:
            pass
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

def vit_pt_imnet(out_dim, ema_coeff, tasks=[], prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, ema_coeff=ema_coeff, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)
    
def moco_pt(out_dim, tasks=[], prompt_flag = 'None', prompt_param=None):
    return MoCoZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)