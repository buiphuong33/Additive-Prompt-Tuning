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
        # x là cls_token: [batch, emb_d]
        return self.net(x) # Output: [batch, num_layers]
class APT(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, ema_coeff):

        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.n_tasks = n_tasks
        self._init_smart(prompt_param)

        self.merge_flag = True

        self.ema_coeff = ema_coeff

        self.prompt_tokens = create_prompt_with_init(12*2, emb_d) 
        global_merged_prompt = torch.zeros(12*2, emb_d).cuda()
        self.register_buffer('global_merged_prompt', global_merged_prompt.clone().detach()) 
        
        self.gate_net = SharedGate(emb_d, 12)
        
        # 2. Buffer lưu giá trị cổng trung bình để dùng cho Priority Fusion sau này
        self.register_buffer('avg_gate_values', torch.zeros(12))
        self.current_gate_values = None # Lưu tạm trong 1 batch để tính loss

        trunc_normal_(self.prompt_tokens, std=0.02)

        for i in range(12):
            setattr(self, f'k_layer_proj{i}', nn.Linear(2, 2))
            setattr(self, f'v_layer_proj{i}', nn.Linear(2, 2))
         
   
    def merge_prompt(self, prompt1, prompt2):
        print("Merging prompt ... ")
        return prompt1*self.ema_coeff + prompt2*(1-self.ema_coeff)

    def _init_smart(self, prompt_param):
        self.prompt_dropout_ratio = float(prompt_param[0])
        self.prompt_dropout = nn.Dropout(self.prompt_dropout_ratio)

    def process_task_count(self):
        self.task_count += 1

    def forward(self, l, x_block, train=False):
        """
        l: layer_idx
        x_block: đầu vào của transformer block (B, N, D)
        """
        B, N, D = x_block.shape

        # 3. Tính toán Gate Values dựa trên CLS token (x_block[:, 0])
        # Chúng ta tính toán gate cho toàn bộ 12 lớp một lần tại layer 0
        # và tái sử dụng cho các layer sau trong cùng 1 forward pass.
        if l == 0:
            cls_token = x_block[:, 0] # [B, D]
            self.current_gate_values = self.gate_net(cls_token) # [B, 12]
        
        # Lấy trọng số cổng cho lớp hiện tại l
        # current_g: [B, 1]
        current_g = self.current_gate_values[:, l].view(B, 1)

        # 4. Chọn nguồn prompt (huấn luyện dùng prompt_tokens, test dùng global_merged)
        if train or not self.merge_flag:
            prompt_k = self.prompt_tokens[l*2]      # [D]
            prompt_v = self.prompt_tokens[l*2 + 1]  # [D]
        else:
            prompt_k = self.global_merged_prompt[l*2]
            prompt_v = self.global_merged_prompt[l*2 + 1]

        # 5. Áp dụng Gating (Nhân trọng số động vào prompt)
        # prompt_k/v: [B, D]
        prompt_k = prompt_k.unsqueeze(0) * current_g 
        prompt_v = prompt_v.unsqueeze(0) * current_g

        # 6. Biến đổi sang định dạng Multi-head để cộng vào Attention (giống logic cũ)
        # Giả sử: 12 heads * 64 dims = 768 (emb_d)
        P_root_k = prompt_k.reshape(B, 12, 1, 64)
        P_root_v = prompt_v.reshape(B, 12, 1, 64)

        # Tạo padding zero cho các token còn lại (không phải CLS)
        P_k = torch.cat((P_root_k, torch.zeros((B, 12, N-1, 64), device=x_block.device)), dim=-2)
        P_v = torch.cat((P_root_v, torch.zeros((B, 12, N-1, 64), device=x_block.device)), dim=-2)
        
        return [P_k, P_v]
    
    @torch.no_grad()
    def priority_fusion(self):
        """
        Cơ chế Hợp nhất dựa trên độ quan trọng của Cổng (thay thế merge_prompt cũ)
        Được gọi sau khi kết thúc 1 task.
        """
        print("Executing Priority Fusion based on Gate Importance...")
        for l in range(12):
            # Trọng số alpha dựa trên mức độ mở cổng trung bình của task cũ
            # Nếu avg_gate cao -> Task cũ dùng lớp này nhiều -> Giữ lại prompt cũ
            alpha = self.avg_gate_values[l]
            
            # Cập nhật cho cả K và V prompt của lớp l
            idx_k, idx_v = l*2, l*2 + 1
            
            self.global_merged_prompt[idx_k] = alpha * self.global_merged_prompt[idx_k] + \
                                               (1 - alpha) * self.prompt_tokens[idx_k]
                                               
            self.global_merged_prompt[idx_v] = alpha * self.global_merged_prompt[idx_v] + \
                                               (1 - alpha) * self.prompt_tokens[idx_v]

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
            self.prompt = APT(768, prompt_param[0], prompt_param[1], ema_coeff=ema_coeff)
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