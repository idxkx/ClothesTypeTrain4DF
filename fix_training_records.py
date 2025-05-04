import os
import json
import re
from datetime import datetime

def extract_model_info(model_dir):
    """从模型目录名称中提取训练信息"""
    # 示例: MD_RESNET18_1_128_50E04_0504_1821
    pattern = r'MD_(\w+)_(\d+)_(\d+)_(\d+E\d+)_(\d{4})_(\d{4})'
    match = re.match(pattern, os.path.basename(model_dir))
    
    if not match:
        return None
        
    backbone, version, image_size, lr_code, date, time = match.groups()
    
    # 解析学习率（例如：50E04 -> 5e-4）
    lr_match = re.match(r'(\d+)E(\d+)', lr_code)
    if lr_match:
        lr_base, lr_exp = lr_match.groups()
        learning_rate = float(lr_base) * (10 ** (-int(lr_exp)))
    else:
        learning_rate = 0.0001  # 默认值
    
    # 构建日期时间
    date_str = f"2024-05-{date[:2]}"  # 假设是2024年
    time_str = f"{time[:2]}:{time[2:]}"
    timestamp = f"{date_str} {time_str}"
    
    return {
        "model_name": os.path.basename(model_dir),
        "backbone": backbone,
        "version": version,
        "image_size": int(image_size),
        "learning_rate": learning_rate,
        "date_created": timestamp,
        "best_model_path": "",  # 将在后面填充
        "completed_epochs": 0,  # 将在后面填充
        "status": "completed",
        "num_categories": 13,  # 默认值
        "batch_size": 32,      # 默认值
        "total_epochs": 50,    # 默认值
    }

def find_best_model_file(model_dir):
    """在模型目录中查找最佳模型文件"""
    for file in os.listdir(model_dir):
        if file.startswith("best_model_") and file.endswith(".pth"):
            # 提取epoch数
            epoch_match = re.search(r'epoch(\d+)\.pth$', file)
            epoch = int(epoch_match.group(1)) if epoch_match else 0
            return os.path.join(model_dir, file), epoch
    return None, 0

def fix_training_records():
    """修复训练记录"""
    models_dir = "models"
    results_file = "training_results.json"
    
    # 读取现有记录
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            existing_records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_records = []
    
    # 确保existing_records是列表
    if not isinstance(existing_records, list):
        existing_records = []
    
    # 获取所有模型目录
    model_dirs = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("MD_")]
    
    # 现有记录的模型名称集合
    existing_models = {record.get("model_name", "") for record in existing_records}
    
    # 处理每个模型目录
    for model_dir in model_dirs:
        full_dir = os.path.join(models_dir, model_dir)
        
        # 如果模型已经在记录中，跳过
        if model_dir in existing_models:
            continue
        
        # 提取模型信息
        model_info = extract_model_info(full_dir)
        if not model_info:
            print(f"无法解析模型信息: {model_dir}")
            continue
        
        # 查找最佳模型文件
        best_model_file, epoch = find_best_model_file(full_dir)
        if not best_model_file:
            print(f"未找到模型文件: {model_dir}")
            continue
        
        # 更新模型信息
        model_info["best_model_path"] = best_model_file
        model_info["completed_epochs"] = epoch
        
        # 添加到记录中
        existing_records.append(model_info)
        print(f"添加模型记录: {model_dir}")
    
    # 保存更新后的记录
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(existing_records, f, indent=4, ensure_ascii=False)
    
    print(f"已更新训练记录，共 {len(existing_records)} 条记录")

if __name__ == "__main__":
    fix_training_records() 