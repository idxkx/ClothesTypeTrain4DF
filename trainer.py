# Placeholder for training loop and logic 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import math

# 导入我们之前定义的类
# 假设 dataset.py 和 model.py 在同一个目录下
# 添加错误处理和占位符，以便在导入失败时仍能进行基本测试
try:
    from dataset import DeepFashionDataset
    from model import ClothesModel
except ImportError:
    print("警告：无法直接导入 dataset 或 model。请确保 trainer.py 与它们在同一目录或正确配置了 PYTHONPATH。")
    from torch.utils.data import Dataset
    from torch.utils.data import Subset # 用于创建小数据集测试
    class DummyDataset(Dataset):
        def __init__(self, length=100, num_categories=50, num_attributes=1000):
            self.length = length
            self.num_categories = num_categories
            self.num_attributes = num_attributes
        def __len__(self): return self.length
        def __getitem__(self, idx):
            # 确保标签在有效范围内
            category_label = torch.randint(0, self.num_categories, (1,)).squeeze()
            # 确保属性标签是 float 类型，符合 BCEWithLogitsLoss 要求
            attribute_label = torch.randint(0, 2, (self.num_attributes,)).float() 
            return {'image': torch.randn(3, 224, 224), 'category': category_label, 'attributes': attribute_label}
    class DummyModel(nn.Module):
        def __init__(self, num_categories=50, num_attributes=26, backbone='dummy', pretrained=False): # 添加参数以匹配 ClothesModel
            super().__init__()
            # 简化模型以加快测试
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            # 计算展平后的特征数量
            # 对于 (1, 1) 的池化输出和 16 个通道，展平后是 16
            num_ftrs = 16 
            self.category_head = nn.Linear(num_ftrs, num_categories)
            self.attribute_head = nn.Linear(num_ftrs, num_attributes)
        def forward(self, x):
            x = self.pool(torch.relu(self.conv(x)))
            x = self.flatten(x)
            cat_logits = self.category_head(x)
            attr_logits = self.attribute_head(x)
            return cat_logits, attr_logits
    # 使用占位符
    print("使用 DummyDataset 和 DummyModel 进行基本测试。")
    DeepFashionDataset = DummyDataset 
    ClothesModel = DummyModel
    # 创建一个小型虚拟数据集用于测试
    dummy_train_ds = DummyDataset(length=64) # 减少长度加速测试
    dummy_val_ds = DummyDataset(length=32)

class Trainer:
    """负责模型训练和验证过程的类"""

    def __init__(self, model, train_dataset, val_dataset, args):
        """
        初始化 Trainer。

        Args:
            model (nn.Module): 要训练的模型实例。
            train_dataset (Dataset): 训练数据集。
            val_dataset (Dataset): 验证数据集。
            args (dict): 包含训练超参数的字典，例如：
                'epochs': int
                'batch_size': int
                'learning_rate': float
                'device': str ('cuda' or 'cpu')
                'model_save_path': str (可选)
                'attribute_loss_weight': float (可选, 默认 1.0)
                'num_workers': int (可选, 默认 0)
        """
        self.args = args
        # 设备选择与确认
        if args.get('device'):
             self.device = torch.device(args['device'])
        else:
             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
             print(f"[Trainer] 未指定设备，自动选择: {self.device}")
             self.args['device'] = str(self.device) # 更新 args 以保持一致

        print(f"[Trainer] 使用设备: {self.device}")
        self.model = model.to(self.device)
        print(f"[Trainer] 模型已移至: {self.device}")

        # DataLoader 配置
        num_workers = args.get('num_workers', 0)
        batch_size = args.get('batch_size', 16) # 减小默认批次大小以适应虚拟测试
        pin_memory = self.device.type == 'cuda' # 仅在 CUDA 上启用 pin_memory

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=pin_memory
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        print(f"[Trainer] DataLoader 初始化完成. 批次: {batch_size}, workers: {num_workers}")

        # 损失函数
        self.criterion_category = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_attribute = nn.BCEWithLogitsLoss() # 适用于多标签
        self.attribute_loss_weight = args.get('attribute_loss_weight', 1.0)
        print(f"[Trainer] 损失函数初始化完成. Attr Weight: {self.attribute_loss_weight}")

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.get('learning_rate', 1e-4))
        print(f"[Trainer] 优化器 Adam 初始化完成. LR: {args.get('learning_rate', 1e-4)}")

        self.epochs = args.get('epochs', 5) # 减少默认轮次以加速测试
        self.model_save_path = args.get('model_save_path', './models')
        # 安全地创建目录
        if self.model_save_path:
            try:
                os.makedirs(self.model_save_path, exist_ok=True) # exist_ok=True 避免目录已存在时报错
                print(f"[Trainer] 模型将保存在: {self.model_save_path}")
            except OSError as e:
                print(f"错误：无法创建模型保存目录 {self.model_save_path}: {e}")
                self.model_save_path = None # 创建失败则不保存

        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'train_cat_acc': [], 'val_cat_acc': []}
        print(f"[Trainer] 将训练 {self.epochs} 个 Epochs.")

    def _train_epoch(self, epoch_num):
        """执行一个训练轮次"""
        self.model.train()
        total_loss, total_cat_loss, total_attr_loss = 0.0, 0.0, 0.0
        correct_categories, total_samples = 0, 0
        start_time = time.time()
        num_batches = len(self.train_loader)

        print(f"\n--- 开始训练 Epoch {epoch_num+1}/{self.epochs} ---")
        for i, batch in enumerate(self.train_loader):
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                cat_labels = batch['category'].to(self.device, non_blocking=True)
                attr_labels = batch['attributes'].to(self.device, non_blocking=True)
            except Exception as e:
                print(f"错误：移动数据到设备时出错 (batch {i+1}): {e}. 跳过。")
                continue

            self.optimizer.zero_grad()

            try:
                cat_logits, attr_logits = self.model(images)
                loss_cat = self.criterion_category(cat_logits, cat_labels)
                loss_attr = self.criterion_attribute(attr_logits, attr_labels)
                if not (torch.isfinite(loss_cat) and torch.isfinite(loss_attr)):
                    print(f"警告：检测到无效损失值 (batch {i+1}). 跳过。")
                    continue
                loss = loss_cat + self.attribute_loss_weight * loss_attr
            except Exception as e:
                print(f"错误：前向传播或损失计算出错 (batch {i+1}): {e}. 跳过。")
                continue

            try:
                loss.backward()
                self.optimizer.step()
            except Exception as e:
                print(f"错误：反向传播或优化器更新出错 (batch {i+1}): {e}. 跳过。")
                continue

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_cat_loss += loss_cat.item() * batch_size
            total_attr_loss += loss_attr.item() * batch_size

            _, predicted_cats = torch.max(cat_logits.data, 1)
            correct_categories += (predicted_cats == cat_labels).sum().item()
            total_samples += batch_size

            # 打印进度
            if (i + 1) % max(1, num_batches // 10) == 0 or (i + 1) == num_batches: # 大约打印10次
                elapsed_time = time.time() - start_time
                current_avg_loss = total_loss / total_samples if total_samples else 0
                current_cat_acc = 100.0 * correct_categories / total_samples if total_samples else 0
                print(f"  Batch {i+1}/{num_batches} | Loss: {current_avg_loss:.4f} | Cat Acc: {current_cat_acc:.2f}% | Time: {elapsed_time:.2f}s")
                # TODO: UI 更新点: 发送进度信息

        epoch_loss = total_loss / total_samples if total_samples else float('inf')
        epoch_cat_acc = 100.0 * correct_categories / total_samples if total_samples else 0.0
        print(f"--- Epoch {epoch_num+1} 训练完成 ---")
        return epoch_loss, epoch_cat_acc

    def _validate_epoch(self, epoch_num):
        """执行一个验证轮次"""
        self.model.eval() # 设置模型为评估模式
        total_loss = 0.0
        correct_categories = 0
        total_samples = 0
        start_time = time.time()
        num_batches = len(self.val_loader)

        print(f"--- 开始验证 Epoch {epoch_num+1}/{self.epochs} ---")
        with torch.no_grad(): # 验证时不需要计算梯度
            for i, batch in enumerate(self.val_loader):
                try:
                    images = batch['image'].to(self.device, non_blocking=True)
                    cat_labels = batch['category'].to(self.device, non_blocking=True)
                    attr_labels = batch['attributes'].to(self.device, non_blocking=True)
                except Exception as e:
                    print(f"错误：移动验证数据到设备时出错 (batch {i+1}): {e}. 跳过。")
                    continue

                try:
                    cat_logits, attr_logits = self.model(images)
                    loss_cat = self.criterion_category(cat_logits, cat_labels)
                    loss_attr = self.criterion_attribute(attr_logits, attr_labels)
                    if not (torch.isfinite(loss_cat) and torch.isfinite(loss_attr)):
                         print(f"警告：检测到无效验证损失值 (batch {i+1}). 跳过。")
                         continue
                    loss = loss_cat + self.attribute_loss_weight * loss_attr
                except Exception as e:
                     print(f"错误：验证时前向传播或损失计算出错 (batch {i+1}): {e}. 跳过。")
                     continue

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                _, predicted_cats = torch.max(cat_logits.data, 1)
                correct_categories += (predicted_cats == cat_labels).sum().item()
                total_samples += batch_size
                
                # 可以减少验证过程中的打印频率
                # if (i + 1) % max(1, num_batches // 5) == 0 or (i + 1) == num_batches:
                #     print(f"  Validation Batch {i+1}/{num_batches}")

        epoch_loss = total_loss / total_samples if total_samples else float('inf')
        epoch_cat_accuracy = 100.0 * correct_categories / total_samples if total_samples else 0.0
        epoch_time = time.time() - start_time
        print(f"--- Epoch {epoch_num+1} 验证完成 ---")
        return epoch_loss, epoch_cat_accuracy

    def train(self):
        """执行完整的训练流程"""
        print("\n==================== 开始训练 ====================")
        start_train_time = time.time()

        for epoch in range(self.epochs):
            train_loss, train_cat_acc = self._train_epoch(epoch)
            val_loss, val_cat_acc = self._validate_epoch(epoch)

            # 记录历史数据
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_cat_acc'].append(train_cat_acc)
            self.history['val_cat_acc'].append(val_cat_acc)

            print(f"\nEpoch {epoch+1}/{self.epochs} 结果:")
            print(f"  训练 - Loss: {train_loss:.4f}, Cat Acc: {train_cat_acc:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Cat Acc: {val_cat_acc:.2f}%")
            # TODO: UI 更新点: 发送 epoch 结果

            # 保存最佳模型 (基于验证损失)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.model_save_path:
                    save_filename = os.path.join(self.model_save_path, f"best_model_epoch_{epoch+1}.pth")
                    try:
                        torch.save(self.model.state_dict(), save_filename)
                        print(f"  ** 新的最佳模型已保存到: {save_filename} (Val Loss: {val_loss:.4f}) **")
                    except Exception as e:
                        print(f"错误：保存模型时出错: {e}")
                else:
                     print("  (模型保存路径未设置，跳过保存最佳模型)")
            
            # TODO: 实现提前停止逻辑 (Early Stopping)

        total_train_time = time.time() - start_train_time
        print("\n==================== 训练完成 ====================")
        print(f"总训练时间: {total_train_time:.2f} 秒")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        return self.history # 返回训练历史记录

# --- 示例用法 (用于基本测试) ---
if __name__ == '__main__':
    print("[Trainer Test] 开始测试 Trainer 类...")

    # --- 定义用于测试的虚拟数据和模型 ---
    # (这些定义现在放在测试块内部，确保总是使用它们进行测试)
    from torch.utils.data import Dataset
    from torch.utils.data import Subset
    class DummyDataset(Dataset):
        def __init__(self, length=100, num_categories=50, num_attributes=26):
            self.length = length
            self.num_categories = num_categories
            self.num_attributes = num_attributes
        def __len__(self): return self.length
        def __getitem__(self, idx):
            category_label = torch.randint(0, self.num_categories, (1,)).squeeze()
            attribute_label = torch.randint(0, 2, (self.num_attributes,)).float() 
            return {'image': torch.randn(3, 224, 224), 'category': category_label, 'attributes': attribute_label}
    class DummyModel(nn.Module):
        def __init__(self, num_categories=50, num_attributes=26, backbone='dummy', pretrained=False):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            num_ftrs = 16 
            self.category_head = nn.Linear(num_ftrs, num_categories)
            self.attribute_head = nn.Linear(num_ftrs, num_attributes)
        def forward(self, x):
            x = self.pool(torch.relu(self.conv(x)))
            x = self.flatten(x)
            cat_logits = self.category_head(x)
            attr_logits = self.attribute_head(x)
            return cat_logits, attr_logits
    # --- 虚拟数据和模型定义结束 ---

    print("[Trainer Test] 使用内部定义的 DummyDataset 和 DummyModel 进行测试。")
    num_categories = 50
    num_attributes = 26
    
    # 明确使用 Dummy 数据集和模型
    test_model = DummyModel(num_categories=num_categories, num_attributes=num_attributes)
    test_train_ds = DummyDataset(length=128, num_categories=num_categories, num_attributes=num_attributes)
    test_val_ds = DummyDataset(length=64, num_categories=num_categories, num_attributes=num_attributes)

    # 定义测试参数
    test_args = {
        'epochs': 2, 
        'batch_size': 16,
        'learning_rate': 1e-3,
        'device': 'cpu', # 强制 CPU 测试
        'model_save_path': './test_models', 
        'attribute_loss_weight': 0.5,
        'num_workers': 0
    }

    print(f"[Trainer Test] 测试参数: {test_args}")

    # 创建 Trainer 实例并开始训练
    try:
        trainer = Trainer(test_model, test_train_ds, test_val_ds, test_args)
        print("[Trainer Test] Trainer 初始化成功。")
        history = trainer.train()
        print("[Trainer Test] 训练过程完成。")
        print("[Trainer Test] 训练历史记录:")
        print(history)
        
        # ... (省略后续检查代码) ...
        assert len(history['train_loss']) == test_args['epochs']
        # ... (省略其他检查) ...
        print("[Trainer Test] 历史记录长度检查通过。")
        # ... (省略模型文件检查) ...
        
    except Exception as e:
        print(f"[Trainer Test] Trainer 测试过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

    print("[Trainer Test] 测试结束。") 