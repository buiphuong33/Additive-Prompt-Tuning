# learners/prompt.py
from __future__ import print_function
import torch
import models
from utils.metric import accuracy, AverageMeter, Timer
from .default import NormalNN, weight_reset, accumulate_acc
from utils.schedulers import CosineSchedule

class Prompt_Learner(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']
        super(Prompt_Learner, self).__init__(learner_config)

    def update_model(self, inputs, targets):
        logits = self.model(inputs, train=True)
        
        logits = logits[:, :self.valid_out_dim]
        logits[:, :self.last_valid_out_dim] = -float('inf')
        
        # 2. Tính Loss phân lớp truyền thống (L_ce)
        loss_ce = self.criterion(logits, targets.long())       
        
        # 3. Tính Loss trực giao (L_ortho) cho các Keys của APT-D để tránh bùng nổ/trùng lặp tri thức
        # Trích xuất module prompt (Xử lý trường hợp có DataParallel hoặc không)
        prompt_module = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt
        
        keys = prompt_module.keys # Shape: [Total_components, key_dim]
        keys_norm = torch.nn.functional.normalize(keys, p=2, dim=-1)
        identity = torch.eye(keys.shape[0]).to(keys.device)
        
        # Phép tính ma trận tương đồng giữa các keys: ||K . K^T - I||^2
        loss_ortho = torch.mean((torch.matmul(keys_norm, keys_norm.t()) - identity) ** 2)
        
        # 4. Tổng hợp Loss tổng: L = L_ce + λ × L_ortho (Giả sử hệ số λ = 0.1, bạn có thể đưa vào config)
        lambda_ortho = self.config.get('lambda_ortho', 0.1)
        total_loss = loss_ce + lambda_ortho * loss_ortho
        
        # 5. Backward pass & Cập nhật trọng số
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits

    def get_attn_heatmap(self, inputs):
        return 

    # sets model optimizers
    # SỬA ĐỔI TẠI: learners/prompt.py -> class APT_Learner -> bổ sung/sửa hàm init_optimizer

    def init_optimizer(self):
        if hasattr(self, 'optimizer'):
            del self.optimizer
        # Phân tách cấu hình tham số Optimizer từ file cấu hình config
        optimizer_arg = {'lr': self.config['lr'], 'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] == 'SGD':
            optimizer_arg['momentum'] = self.config['momentum']
            optimizer_arg['nesterov'] = True
        elif self.config['optimizer'] == 'AdamW':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'AdamW'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'], 0.999)

        # === CHỈ LỌC CÁC THAM SỐ CÓ grad=True (Components mới + Keys + Attention vector A) ===
        # Việc lọc filter này đảm bảo các components cũ đã đóng băng sẽ KHÔNG bị update gradients
        params_to_opt = list(filter(lambda p: p.requires_grad, self.model.parameters()))

        # Tạo thực thể Optimizer mới
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](params_to_opt, **optimizer_arg)
        
        # Tái thiết lập bộ điều chỉnh Scheduler phù hợp với Optimizer mới
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
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
    # BỔ SUNG TẠI: learners/prompt.py -> class APT_Learner

    def pre_steps(self):
        """Hàm tự động chạy trước khi bắt đầu huấn luyện mỗi Task"""
        # Nếu không phải task đầu tiên (task_count > 0), tiến hành đóng băng components cũ và sinh components mới
        if self.task_count > 0:
            prompt_module = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt
            
            # Giả sử mỗi task mới bạn muốn sinh thêm 10 components mới
            num_new_components = self.config.get('num_components_per_task', 10)
            prompt_module.freeze_old_components(M_new=num_new_components)

            device = next(self.model.parameters()).device
            self.model.to(device)
            
            # CỰC KỲ QUAN TRỌNG: Gọi lại init_optimizer để làm mới danh sách tham số huấn luyện của Optimizer
            self.init_optimizer()
            
        self.task_count += 1

    def post_steps(self):
        """Hàm tự động chạy sau khi kết thúc huấn luyện một Task"""
        prompt_module = self.model.module.prompt if hasattr(self.model, 'module') else self.model.prompt
        
        # Thực hiện thuật toán Progressive Prompt Fusion để hòa trộn cấu trúc components
        beta_fusion = self.config.get('beta_ppf', 0.9)
        if hasattr(prompt_module, 'progressive_prompt_fusion'):
            prompt_module.progressive_prompt_fusion(beta=beta_fusion)
            print(f"==> Đã thực hiện đồng bộ Progressive Prompt Fusion với hệ số beta={beta_fusion}")

class APT_Learner(Prompt_Learner):

    def __init__(self, learner_config):
        super(APT_Learner, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, ema_coeff=self.ema_coeff, prompt_flag = 'apt', prompt_param=self.prompt_param, tasks=self.tasks)
        return model
