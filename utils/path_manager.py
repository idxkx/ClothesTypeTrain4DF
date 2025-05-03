import os
import json
import streamlit as st
from utils.file_utils import safe_path

def load_config_paths(config_path):
    """从配置文件加载路径"""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'未找到配置文件: {config_path}，请参考README.md在项目根目录下创建，并填写数据路径。')
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        anno_dir = safe_path(config.get('anno_dir', None))
        img_dir = safe_path(config.get('img_dir', None))
        
        if not anno_dir or not os.path.exists(anno_dir):
            system_type = "Windows" if os.name == 'nt' else "Linux"
            raise FileNotFoundError(f'标注文件目录不存在或未配置，请检查 config.json 中的 "anno_dir" 路径: {anno_dir} (当前系统: {system_type})')
            
        if not img_dir or not os.path.exists(img_dir):
            system_type = "Windows" if os.name == 'nt' else "Linux"
            raise FileNotFoundError(f'高分辨率图片目录不存在或未配置，请检查 config.json 中的 "img_dir" 路径: {img_dir} (当前系统: {system_type})')
            
        return anno_dir, img_dir
        
    except FileNotFoundError as e:
        st.error(str(e))
        return None, None
        
    except json.JSONDecodeError:
        st.error(f"配置文件 {config_path} 格式错误，无法解析JSON")
        return None, None
        
    except Exception as e:
        st.error(f"加载配置文件时发生未知错误: {e}")
        return None, None

def update_config_paths(config_path, anno_dir=None, img_dir=None):
    """更新配置文件中的路径"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
            
        if anno_dir is not None:
            config['anno_dir'] = safe_path(anno_dir)
            
        if img_dir is not None:
            config['img_dir'] = safe_path(img_dir)
            
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        return True, "配置文件更新成功"
        
    except Exception as e:
        return False, f"更新配置文件时发生错误: {e}"

def get_model_save_dir(model_name):
    """获取模型保存目录"""
    return os.path.join('.', 'models', model_name)

def get_model_path(model_dir, model_name, epoch=None):
    """获取模型文件路径"""
    if epoch:
        return os.path.join(model_dir, f"best_model_{model_name}_epoch{epoch}.pth")
    else:
        # 查找最佳模型文件
        possible_files = [f for f in os.listdir(model_dir) 
                           if f.startswith(f"best_model_{model_name}") and f.endswith(".pth")]
        if not possible_files:
            return None
            
        # 按文件名中的epoch排序，取最新的
        possible_files.sort(key=lambda x: int(x.split('_epoch')[-1].split('.')[0]), reverse=True)
        return os.path.join(model_dir, possible_files[0])

def get_metadata_path(model_dir, model_name):
    """获取模型元数据文件路径"""
    return os.path.join(model_dir, f"{model_name}_metadata.json") 