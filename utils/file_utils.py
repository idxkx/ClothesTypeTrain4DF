import os
import json
import streamlit as st

def safe_path(path):
    """自动规范化路径，兼容Windows和Linux"""
    return os.path.normpath(path) if path else ""

def load_results(file_path="training_results.json"):
    """加载历史训练结果"""
    try:
        # 确保使用绝对路径
        absolute_path = os.path.abspath(file_path)
        print(f"正在加载训练结果文件: {absolute_path}")
        
        if not os.path.exists(absolute_path):
            print(f"训练结果文件不存在: {absolute_path}")
            return []
            
        with open(absolute_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # 确保返回的是列表
        if not isinstance(results, list):
            print(f"警告: 训练结果不是列表格式，将转换为列表")
            results = [results] if results else []
            
        print(f"已成功加载 {len(results)} 条训练记录")
        return results
    except FileNotFoundError:
        print(f"训练结果文件不存在: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"训练结果文件JSON解析错误: {e}")
        # 文件损坏或为空，返回空列表
        return []
    except Exception as e:
        print(f"加载训练结果时出错: {e}")
        return []

def save_results(results, file_path="training_results.json"):
    """保存训练结果列表到 JSON 文件"""
    try:
        # 确保使用绝对路径
        absolute_path = os.path.abspath(file_path)
        print(f"正在保存训练结果到文件: {absolute_path}")
        
        # 确保结果是列表类型
        if not isinstance(results, list):
            print(f"警告: 训练结果不是列表格式，将转换为列表")
            results = [results] if results else []
        
        # 确保目录存在
        directory = os.path.dirname(absolute_path)
        if directory and not os.path.exists(directory):
            print(f"创建目录: {directory}")
            os.makedirs(directory)
        
        with open(absolute_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        print(f"已成功保存 {len(results)} 条训练记录")
        return True
    except Exception as e:
        error_msg = f"无法保存训练结果到 {file_path}: {e}"
        print(error_msg)
        st.error(error_msg)
        return False

def ensure_dir_exists(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return True
        except Exception as e:
            st.error(f"创建目录失败 {directory}: {e}")
            return False
    return True

def file_exists(file_path):
    """检查文件是否存在"""
    return os.path.isfile(file_path)

def get_file_extension(file_path):
    """获取文件扩展名"""
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def get_directory(file_path):
    """获取文件所在目录"""
    return os.path.dirname(file_path)

def join_paths(*paths):
    """连接多个路径"""
    return os.path.join(*paths)

def find_files(directory, pattern=None, extension=None):
    """查找目录中符合条件的文件"""
    result = []
    if not os.path.exists(directory):
        return result
        
    for file in os.listdir(directory):
        if extension and not file.endswith(extension):
            continue
        if pattern and pattern not in file:
            continue
        result.append(os.path.join(directory, file))
    
    return result 