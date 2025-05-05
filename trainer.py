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

# 导入我们之前定义的类和工具函数
try:
    from dataset import DeepFashionDataset
    from model import ClothesModel
    from utils.file_utils import load_results, save_results
except ImportError:
    print("警告：无法直接导入必要模块。请确保所有依赖模块在正确的位置。")
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
                # 直接暴露错误，不使用伪数据
                error_msg = f"[Trainer] 错误：从数据集获取类别/属性名称时出错: {e}"
                print(error_msg)
                # 如果没有类别和属性名称，我们不应该继续生成元数据
                if not class_names and not feature_names:
                    raise ValueError(f"无法获取类别和属性名称: {e}")
            
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
            
            # 验证元数据的完整性
            if num_categories == 0:
                raise ValueError("模型元数据缺少必要的类别信息")
            
            if len(feature_names) == 0:
                raise ValueError("模型元数据缺少必要的属性信息")
            
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
            
            # 记录详细错误信息
            import traceback
            error_trace = traceback.format_exc()
            print(f"[Trainer] 错误详情：\n{error_trace}")
            
            # 不再尝试生成紧急备份的伪元数据
            # 直接返回None，让调用者知道元数据生成失败
            return None

    def train(self):
        """执行完整的训练过程"""
        print(f"\n=== 开始训练 ===")
        print(f"模型: {self.model.__class__.__name__}")
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.val_dataset)}")
        print(f"批次大小: {self.args.get('batch_size')}")
        print(f"学习率: {self.args.get('learning_rate')}")
        print(f"设备: {self.device}")
        
        start_time = time.time()
        best_model_path = None
        training_record = {
            "model_name": self.args.get("model_name", "unnamed_model"),
            "backbone": self.args.get("backbone", "unknown"),
            "image_size": self.args.get("image_size", 224),
            "batch_size": self.args.get("batch_size", 32),
            "learning_rate": self.args.get("learning_rate", 0.0001),
            "num_categories": self.args.get("num_categories", 13),
            "total_epochs": self.epochs,
            "completed_epochs": 0,
            "best_val_loss": float('inf'),
            "best_val_accuracy": 0.0,
            "best_epoch": 0,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "training",
            "best_model_path": ""
        }
        
        try:
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                
                # 训练和验证
                train_loss, train_cat_acc = self._train_epoch(epoch)
                val_loss, val_cat_acc = self._validate_epoch(epoch)
                
                # 更新历史记录
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_cat_acc'].append(train_cat_acc)
                self.history['val_cat_acc'].append(val_cat_acc)
                
                # 更新训练记录
                training_record["completed_epochs"] = epoch + 1
                
                # 检查是否是最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    training_record["best_val_loss"] = val_loss
                    training_record["best_val_accuracy"] = val_cat_acc
                    training_record["best_epoch"] = epoch + 1
                    
                    # 保存最佳模型
                    if self.model_save_path:
                        model_name = training_record["model_name"]
                        best_model_path = os.path.join(
                            self.model_save_path,
                            f"best_model_{model_name}_epoch{epoch+1}.pth"
                        )
                        torch.save(self.model.state_dict(), best_model_path)
                        training_record["best_model_path"] = best_model_path
                        print(f"保存最佳模型到: {best_model_path}")
                
                # 打印轮次总结
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch+1}/{self.epochs} 总结:")
                print(f"训练损失: {train_loss:.4f} | 训练类别准确率: {train_cat_acc:.2f}%")
                print(f"验证损失: {val_loss:.4f} | 验证类别准确率: {val_cat_acc:.2f}%")
                print(f"轮次用时: {epoch_time:.2f}s")
                
                # 保存当前训练记录
                try:
                    # 读取现有记录
                    existing_records = load_results("training_results.json")
                    if not isinstance(existing_records, list):
                        existing_records = []
                    
                    # 更新或添加当前记录
                    record_updated = False
                    for i, record in enumerate(existing_records):
                        if record.get("model_name") == training_record["model_name"]:
                            existing_records[i] = training_record
                            record_updated = True
                            break
                    
                    if not record_updated:
                        existing_records.append(training_record)
                    
                    # 保存更新后的记录
                    save_results(existing_records, "training_results.json")
                except Exception as e:
                    print(f"警告：保存训练记录时出错: {e}")
            
            # 训练完成，更新状态
            training_record["status"] = "completed"
            total_time = time.time() - start_time
            print(f"\n=== 训练完成 ===")
            print(f"总用时: {total_time:.2f}s")
            print(f"最佳验证损失: {self.best_val_loss:.4f} (Epoch {training_record['best_epoch']})")
            if best_model_path:
                print(f"最佳模型保存在: {best_model_path}")
            
            # 保存最终训练记录
            try:
                existing_records = load_results("training_results.json")
                if not isinstance(existing_records, list):
                    existing_records = []
                
                # 更新或添加最终记录
                record_updated = False
                for i, record in enumerate(existing_records):
                    if record.get("model_name") == training_record["model_name"]:
                        existing_records[i] = training_record
                        record_updated = True
                        break
                
                if not record_updated:
                    existing_records.append(training_record)
                
                save_results(existing_records, "training_results.json")
            except Exception as e:
                print(f"警告：保存最终训练记录时出错: {e}")
            
            return self.history
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
            training_record["status"] = "failed"
            try:
                # 保存失败记录
                existing_records = load_results("training_results.json")
                if not isinstance(existing_records, list):
                    existing_records = []
                
                # 更新或添加失败记录
                record_updated = False
                for i, record in enumerate(existing_records):
                    if record.get("model_name") == training_record["model_name"]:
                        existing_records[i] = training_record
                        record_updated = True
                        break
                
                if not record_updated:
                    existing_records.append(training_record)
                
                save_results(existing_records, "training_results.json")
            except Exception as save_error:
                print(f"警告：保存失败记录时出错: {save_error}")
            raise  # 重新抛出异常

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
        history = trainer.train()
        
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