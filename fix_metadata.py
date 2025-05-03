import os
import json
import traceback
from datetime import datetime
import sys

def load_results(file_path="training_results.json"):
    """加载历史训练结果，不依赖streamlit"""
    try:
        absolute_path = os.path.abspath(file_path)
        print(f"正在加载训练结果文件: {absolute_path}")
        
        if not os.path.exists(absolute_path):
            print(f"训练结果文件不存在: {absolute_path}")
            return []
            
        with open(absolute_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        if not isinstance(results, list):
            print(f"警告: 训练结果不是列表格式，将转换为列表")
            results = [results] if results else []
            
        print(f"已成功加载 {len(results)} 条训练记录")
        return results
    except Exception as e:
        print(f"加载训练结果时出错: {e}")
        return []

def get_default_categories():
    """获取默认类别名称列表"""
    return [
        "T恤", "衬衫", "卫衣", "毛衣", "西装", "夹克", "羽绒服", "风衣",
        "牛仔裤", "休闲裤", "西裤", "短裤", "运动裤", "连衣裙", "半身裙",
        "旗袍", "礼服", "运动鞋", "皮鞋", "高跟鞋", "靴子", "凉鞋", "拖鞋",
        "帽子", "围巾", "领带", "手套", "袜子", "腰带", "眼镜", "手表",
        "项链", "手链", "耳环", "戒指", "包包", "背包", "手提包", "钱包", "行李箱"
    ]

def get_default_features():
    """获取默认特征名称列表"""
    return [
        "颜色", "材质", "样式", "花纹", "季节", "正式度", "领型", "袖长",
        "长度", "裤型", "鞋型", "高度", "闭合方式"
    ]

def generate_metadata_for_model(model_result):
    """为单个模型生成元数据文件"""
    try:
        model_name = model_result.get("model_name", "")
        model_path = model_result.get("best_model_path", "")
        
        if not model_name:
            return False, "模型名称为空"
            
        if not model_path:
            return False, "模型路径为空"
            
        if not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}"
        
        # 提取模型目录
        model_dir = os.path.dirname(model_path)
        if not model_dir:
            return False, "无法提取模型目录"
            
        # 元数据文件路径
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        # 检查是否已存在
        if metadata_file and os.path.exists(metadata_file):
            print(f"[{model_name}] 元数据已存在: {metadata_file}")
            return True, "元数据已存在"
        
        # 从训练结果中提取基本信息
        backbone = model_result.get("backbone", "unknown")
        num_categories = model_result.get("num_categories", 50)  # 默认50个类别
        input_size = model_result.get("image_size", 224)
        learning_rate = model_result.get("parameters", {}).get("learning_rate", 0.0001)
        batch_size = model_result.get("parameters", {}).get("batch_size", 32)
        completed_epochs = model_result.get("completed_epochs", 0)
        best_val_loss = model_result.get("best_val_loss", 0.0)
        best_val_acc = model_result.get("best_val_accuracy", 0.0)
        best_epoch = model_result.get("best_epoch", 0)
        
        # 获取默认的类别和特征名称
        class_names = get_default_categories()
        feature_names = get_default_features()
        
        # 创建元数据字典
        metadata = {
            "model_name": model_name,
            "version": "1.0",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": f"ClothesModel({backbone})",
            "input_shape": [3, input_size, input_size],
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize": input_size
            },
            "class_names": class_names,
            "feature_names": feature_names,
            "training_info": {
                "backbone": backbone,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": completed_epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc
            },
            "export_format": "PyTorch",
            "usage_examples": {
                "python": "# 请参考示例代码使用此模型"
            }
        }
        
        # 保存元数据文件
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"[{model_name}] 元数据创建成功: {metadata_file}")
            return True, "元数据创建成功"
        except Exception as e:
            return False, f"保存元数据文件时出错: {e}"
            
    except Exception as e:
        traceback_info = traceback.format_exc()
        print(f"生成元数据时出错: {traceback_info}")
        return False, f"生成元数据时出错: {e}"

def batch_generate_metadata():
    """批量生成缺失的元数据"""
    results = []
    
    # 加载所有训练记录
    all_results = load_results()
    
    for model_data in all_results:
        model_name = model_data.get("model_name", "")
        model_path = model_data.get("best_model_path", "")
        
        if not model_name or not model_path:
            results.append((model_name or "未命名模型", False, "模型信息不完整"))
            continue
            
        # 检查模型文件是否存在
        if model_path and not os.path.exists(model_path):
            results.append((model_name, False, f"模型文件不存在: {model_path}"))
            continue
        
        # 检查元数据文件是否已存在
        model_dir = os.path.dirname(model_path) if model_path else ""
        if not model_dir:
            results.append((model_name, False, "模型路径不完整"))
            continue
            
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        # 如果元数据已存在，跳过
        if metadata_file and os.path.exists(metadata_file):
            results.append((model_name, True, "元数据已存在"))
            continue
        
        # 生成元数据
        try:
            success, message = generate_metadata_for_model(model_data)
            results.append((model_name, success, message))
        except Exception as e:
            results.append((model_name, False, f"生成元数据时出错: {str(e)}"))
    
    return results

if __name__ == "__main__":
    print("开始批量生成模型元数据...")
    results = batch_generate_metadata()
    
    success_count = sum(1 for _, success, _ in results if success)
    fail_count = len(results) - success_count
    
    print("\n--- 处理结果 ---")
    print(f"总模型数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    
    if fail_count > 0:
        print("\n失败详情:")
        for model_name, success, message in results:
            if not success:
                print(f"- {model_name}: {message}") 