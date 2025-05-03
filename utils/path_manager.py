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
    if not model_name:
        print("警告：模型名称为空，将使用默认目录")
        model_name = f"model_{int(os.path.getmtime('.'))}"  # 使用当前时间戳生成默认名称
        
    base_dir = os.path.join('.', 'models')
    
    # 确保基础目录存在
    try:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"创建模型目录: {base_dir}")
    except Exception as e:
        print(f"创建基础模型目录失败: {e}")
        
    full_path = os.path.join(base_dir, model_name)
    
    # 确保完整路径存在
    try:
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"创建模型保存目录: {full_path}")
    except Exception as e:
        print(f"创建模型保存目录失败: {e}")
        
    return full_path

def get_model_path(model_dir, model_name, epoch=None):
    """获取模型文件路径"""
    # 检查参数有效性
    if not model_dir or not os.path.exists(model_dir):
        print(f"警告：模型目录无效或不存在: {model_dir}")
        return None
        
    if not model_name:
        print("警告：模型名称为空")
        return None
        
    if epoch:
        model_path = os.path.join(model_dir, f"best_model_{model_name}_epoch{epoch}.pth")
        if os.path.exists(model_path):
            return model_path
        else:
            print(f"警告：指定轮次的模型文件不存在: {model_path}")
            return None
    else:
        # 查找最佳模型文件
        try:
            possible_files = [f for f in os.listdir(model_dir) 
                             if f.startswith(f"best_model_{model_name}") and f.endswith(".pth")]
            if not possible_files:
                print(f"警告：未找到匹配的模型文件，目录: {model_dir}, 名称: {model_name}")
                return None
                
            # 按文件名中的epoch排序，取最新的
            try:
                possible_files.sort(key=lambda x: int(x.split('_epoch')[-1].split('.')[0]), reverse=True)
            except (ValueError, IndexError):
                # 如果排序失败，按文件修改时间排序
                possible_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                
            model_path = os.path.join(model_dir, possible_files[0])
            if os.path.exists(model_path):
                return model_path
            else:
                print(f"警告：选择的模型文件不存在: {model_path}")
                return None
        except Exception as e:
            print(f"获取模型路径时出错: {e}")
            return None

def get_metadata_path(model_dir, model_name):
    """获取模型元数据文件路径"""
    if not model_dir or not os.path.exists(model_dir):
        print(f"警告：模型目录无效或不存在: {model_dir}")
        return None
        
    if not model_name:
        print("警告：模型名称为空")
        return None
        
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    
    # 验证路径有效性
    if os.path.exists(metadata_path):
        return metadata_path
    else:
        # 元数据可能尚未创建，返回期望路径但打印警告
        print(f"提示：元数据文件尚未存在: {metadata_path}")
        return metadata_path 