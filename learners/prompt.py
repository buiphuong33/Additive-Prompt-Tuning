#learners/prompt.py
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
        # logits
        logits = self.model(inputs, train=True)
        
        logits = logits[:,:self.valid_out_dim]
        logits[:,:self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logits, targets.long())       
        
        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits

    def get_attn_heatmap(self, inputs):
        return 

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            model_ref = self.model.module
        else:
            model_ref = self.model
        print('*****************************************')
        params_to_opt = []
        
        # Lấy tham số từ module APT (nơi chứa Prompt và Gate)
        if hasattr(model_ref.feat, 'apt'):
            params_to_opt += list(model_ref.feat.apt.parameters())
        else:
            # Backup trường hợp tên thuộc tính khác (nếu có)
            print("Cảnh báo: Không tìm thấy module 'apt' trong model_ref.feat")
        
        if hasattr(model_ref, 'last'):
            params_to_opt += list(model_ref.last.parameters())

        print(f'*** Số lượng nhóm tham số được tối ưu: {len(params_to_opt)} ***')    
        
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

class APT_Learner(Prompt_Learner):

    def __init__(self, learner_config):
        super(APT_Learner, self).__init__(learner_config)
        # Hệ số điều chỉnh độ thưa (sparsity), bạn có thể đưa vào config nếu muốn
        self.reg_lambda = 0.01 

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](
            out_dim=self.out_dim, 
            ema_coeff=self.ema_coeff, 
            prompt_flag = 'APT', # Đảm bảo viết hoa khớp với logic trong vit.py
            prompt_param=self.prompt_param, 
            tasks=self.tasks
        )
        return model

    def update_model(self, inputs, targets):
        # 1. Forward pass
        logits = self.model(inputs, train=True)
        
        # 2. Truy cập vào module APT để lấy gate_values vừa tính ở forward
        # Lưu ý: Xử lý cả trường hợp dùng DataParallel
        model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        # feat là VisionTransformer, apt là module APT chúng ta đã sửa ở zoo.py
        gate_vals = model_ref.feat.apt.current_gate_values 

        # 3. Tính toán Loss
        logits = logits[:,:self.valid_out_dim]
        logits[:,:self.last_valid_out_dim] = -float('inf')
        
        ce_loss = self.criterion(logits, targets.long())
        
        # Sparsity Loss: Ép trung bình các cổng về 0 (chuẩn L1)
        sparsity_loss = gate_vals.mean() 
        
        total_loss = ce_loss + self.reg_lambda * sparsity_loss
        
        # 4. Optimizer step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits

    # --- PHẦN THÊM MỚI ĐỂ XỬ LÝ TASK ---

    def after_task(self):
        """
        Hàm này được gọi sau khi kết thúc huấn luyện một task.
        Nó sẽ tính toán độ quan trọng của từng layer và hợp nhất prompt.
        """
        self.model.eval()
        model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
        print("Calculating average gate values for priority fusion...")
        
        all_gates = []
        # Chạy qua một phần dữ liệu (hoặc toàn bộ) task hiện tại để lấy gate trung bình
        # Ở đây ta tận dụng chính data_loader của task hiện tại
        with torch.no_grad():
            # Lấy khoảng 10-20 batches là đủ để ước lượng độ quan trọng
            for i, (inputs, targets) in enumerate(self.train_loader):
                if i > 20: break 
                inputs = inputs.cuda()
                _ = self.model(inputs, train=False)
                all_gates.append(model_ref.feat.apt.current_gate_values.mean(0)) # Mean theo batch
        
        # Tính trung bình cổng trên toàn bộ mẫu: kết quả là vector [12]
        avg_g = torch.stack(all_gates).mean(0)
        
        # Cập nhật vào buffer của APT
        model_ref.feat.apt.avg_gate_values.copy_(avg_g)
        
        # Thực hiện Priority Fusion (Hợp nhất dựa trên độ quan trọng)
        model_ref.feat.apt.priority_fusion()
        
        # Tăng task count
        model_ref.feat.apt.process_task_count()
        super().after_task()
