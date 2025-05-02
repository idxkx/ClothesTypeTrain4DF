# Placeholder for training loop and logic 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import math
import json
from datetime import datetime

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

        # 存储数据集，用于元数据生成
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

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

    def _save_metadata(self, model_name):
        """生成并保存模型元数据文件"""
        print(f"[Trainer] 生成模型元数据...")
        
        try:
            # 获取模型相关信息
            if isinstance(self.model, ClothesModel):
                backbone = getattr(self.model, 'backbone_name', 
                                  self.args.get('backbone', 'unknown_backbone'))
            else:
                backbone = "unknown_backbone"
            
            # 尝试获取类别和属性名称
            class_names = []
            feature_names = []
            
            # 从数据集中获取类别和属性名称
            try:
                # 尝试从train_dataset读取类别和属性名称
                if hasattr(self.train_dataset, 'category_names') and self.train_dataset.category_names:
                    # 类别名称通常是字典 {id: name}，需要转换为列表
                    category_dict = self.train_dataset.category_names
                    # 因为ID可能从1开始，我们需要确保列表索引正确
                    max_id = max(category_dict.keys()) if category_dict else 0
                    class_names = ["未知"] * max_id
                    for cat_id, cat_name in category_dict.items():
                        if 1 <= cat_id <= max_id:  # 确保ID在有效范围内
                            class_names[cat_id-1] = cat_name  # 调整为0-based索引
                
                if hasattr(self.train_dataset, 'attribute_names') and self.train_dataset.attribute_names:
                    # 属性名称也是字典，同样转换为列表
                    attribute_dict = self.train_dataset.attribute_names
                    max_id = max(attribute_dict.keys()) if attribute_dict else 0
                    feature_names = ["未知"] * max_id
                    for attr_id, attr_name in attribute_dict.items():
                        if 1 <= attr_id <= max_id:
                            feature_names[attr_id-1] = attr_name
            except Exception as e:
                print(f"[Trainer] 警告：从数据集获取类别/属性名称时出错: {e}")
                # 出错时使用空列表
                class_names = []
                feature_names = []
            
            # 尝试从name_mapping.json加载中文名称映射
            try:
                if os.path.exists('name_mapping.json'):
                    with open('name_mapping.json', 'r', encoding='utf-8') as f:
                        name_mapping = json.load(f)
                    
                    # 如果有映射且class_names非空，尝试应用映射
                    if 'categories' in name_mapping and class_names:
                        for i, en_name in enumerate(class_names):
                            if en_name in name_mapping['categories']:
                                class_names[i] = name_mapping['categories'][en_name]
                    
                    # 对属性名称也进行同样操作
                    if 'attributes' in name_mapping and feature_names:
                        for i, en_name in enumerate(feature_names):
                            if en_name in name_mapping['attributes']:
                                feature_names[i] = name_mapping['attributes'][en_name]
            except Exception as e:
                print(f"[Trainer] 警告：应用中文名称映射时出错: {e}")
            
            # 输入形状采用标准尺寸，如果使用了其他尺寸，应从配置获取
            input_shape = [3, 224, 224]  # 默认值
            
            # 当前时间作为模型创建日期
            date_created = datetime.now().strftime("%Y-%m-%d")
            
            # 获取特征数量
            num_categories = len(class_names)
            
            # 构建元数据 (使用英文key)
            metadata = {
                "model_name": model_name,
                "version": "1.0.0",  # 可根据实际情况调整
                "description": f"基于{backbone}的服装分类模型，支持{num_categories}类服装分类",
                "input_shape": input_shape,
                "architecture": backbone,
                "class_names": class_names,
                "feature_names": feature_names,
                "date_created": date_created,
                "framework": "PyTorch",
                "trained_by": "服装类别与属性识别训练平台"
            }
            
            # 保存元数据文件
            metadata_path = os.path.join(self.model_save_path, f"{model_name}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            
            print(f"[Trainer] 元数据文件已保存至: {metadata_path}")
            return metadata_path
        
        except Exception as e:
            print(f"[Trainer] 错误：生成元数据文件时出错: {e}")
            # 出错时返回None
            return None

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

            # 计算训练时间
            epoch_time = time.time() - start_train_time
            avg_time_per_epoch = epoch_time / (epoch + 1)
            remaining_epochs = self.epochs - (epoch + 1)
            estimated_remaining_time = avg_time_per_epoch * remaining_epochs

            # 打印本轮结果
            print("\n--- 训练摘要 ---")
            print(f"Epoch {epoch+1}/{self.epochs} - 耗时: {epoch_time:.2f}s (估计剩余: {estimated_remaining_time:.2f}s)")
            print(f"  训练损失: {train_loss:.4f}, 类别准确率: {train_cat_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f}, 类别准确率: {val_cat_acc:.2f}%")

            # 保存最佳模型
            if self.model_save_path:
                # 始终保存最新的模型
                latest_model_file = os.path.join(self.model_save_path, f"latest_model_epoch{epoch+1}.pth")
                try:
                    torch.save(self.model.state_dict(), latest_model_file)
                    print(f"最新模型已保存: {latest_model_file}")
                except Exception as e:
                    print(f"错误：保存最新模型时出错: {e}")

                # 保存性能最佳的模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_model_file = os.path.join(self.model_save_path, f"best_model_epoch{epoch+1}.pth")
                    try:
                        torch.save(self.model.state_dict(), best_model_file)
                        print(f"新的最佳模型已保存: {best_model_file} (验证损失: {val_loss:.4f})")
                        
                        # 为最佳模型生成metadata.json文件
                        model_name = os.path.basename(self.model_save_path)
                        self._save_metadata(model_name)
                    except Exception as e:
                        print(f"错误：保存最佳模型时出错: {e}")

        # 训练完成
        total_time = time.time() - start_train_time
        print("\n==================== 训练完成 ====================")
        print(f"训练时间总计: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        # 返回训练历史和最佳验证损失
        return self.history, self.best_val_loss

# --- 仅用于测试 ---
if __name__ == "__main__":
    print("\n--- 测试 Trainer 类 (使用模拟数据) ---")
    
    # 创建简单模型和数据集用于测试
    model = DummyModel(num_categories=10, num_attributes=5)
    train_ds = DummyDataset(length=32, num_categories=10, num_attributes=5)
    val_ds = DummyDataset(length=16, num_categories=10, num_attributes=5)
    
    # 测试参数
    args = {
        'epochs': 2,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'model_save_path': './test_models',
        'attribute_loss_weight': 0.5
    }
    
    # 初始化 Trainer
    try:
        trainer = Trainer(model, train_ds, val_ds, args)
        
        # 执行训练
        history, best_val_loss = trainer.train()
        
        print("\n--- 训练历史摘要 ---")
        for epoch in range(len(history['train_loss'])):
            print(f"Epoch {epoch+1}: Train Loss={history['train_loss'][epoch]:.4f}, "
                  f"Val Loss={history['val_loss'][epoch]:.4f}, "
                  f"Train Cat Acc={history['train_cat_acc'][epoch]:.2f}%, "
                  f"Val Cat Acc={history['val_cat_acc'][epoch]:.2f}%")
    
    except Exception as e:
        import traceback
        print(f"测试失败: {e}")
        print(traceback.format_exc())
    
    print("\n--- 测试完成 ---") 