import streamlit as st
import time
import json
import os

def initialize_session_state():
    """初始化应用的会话状态"""
    # 训练控制相关
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False
    
    # 日志相关
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    # 训练历史相关
    if 'history_df_list' not in st.session_state:
        st.session_state.history_df_list = []
    
    # 训练策略相关
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = "均衡推荐 (Balanced)"
    
    # GPU监控相关
    if 'gpu_metrics_history' not in st.session_state:
        st.session_state.gpu_metrics_history = []
    if 'gpu_poll_step' not in st.session_state:
        st.session_state.gpu_poll_step = 0
    
    # 时间统计相关
    if 'training_start_time' not in st.session_state:
        st.session_state.training_start_time = None
    if 'epoch_durations' not in st.session_state:
        st.session_state.epoch_durations = []
    
    # 元数据和模型相关
    if 'selected_model_for_details' not in st.session_state:
        st.session_state.selected_model_for_details = None
    
    # 显示选项相关
    if 'show_failed' not in st.session_state:
        st.session_state.show_failed = True
    
    # 错误标记状态
    if 'gpu_error_logged' not in st.session_state:
        st.session_state.gpu_error_logged = False
    if 'chart_error_logged' not in st.session_state:
        st.session_state.chart_error_logged = False

def append_log(message):
    """将带时间戳的消息追加到 session_state 的日志列表中"""
    timestamp = time.strftime('%H:%M:%S')
    st.session_state.log_messages.append(f"[{timestamp}] {message}")

def reset_training_state():
    """重置训练相关的状态，用于开始新的训练"""
    st.session_state.log_messages = []
    st.session_state.history_df_list = []
    st.session_state.stop_requested = False
    st.session_state.gpu_error_logged = False
    st.session_state.gpu_metrics_history = []
    st.session_state.gpu_poll_step = 0
    st.session_state.chart_error_logged = False
    st.session_state.training_start_time = time.time()
    st.session_state.epoch_durations = []

def update_session_state(key, value):
    """更新会话状态中的指定键值"""
    st.session_state[key] = value 

def get_default_categories():
    """获取默认的服装类别列表"""
    try:
        # 首先尝试从name_mapping.json文件获取类别
        if os.path.exists('name_mapping.json'):
            with open('name_mapping.json', 'r', encoding='utf-8') as f:
                categories = json.load(f)
                if isinstance(categories, dict) and 'categories' in categories:
                    return categories['categories']
                return list(categories.keys()) if isinstance(categories, dict) else categories
        
        # 如果文件不存在或格式不对，返回默认类别列表
        return [
            "T恤", "连衣裙", "上衣", "衬衫", "毛衣", "夹克", "风衣", 
            "牛仔裤", "休闲裤", "短裤", "西装", "裙子", "大衣"
        ]
    except Exception as e:
        print(f"获取默认类别时出错: {e}")
        # 出错时返回基本默认值
        return ["类别1", "类别2", "类别3", "类别4", "类别5"]

def get_default_features():
    """获取默认的服装特征列表"""
    try:
        # 首先尝试从custom_attributes.json文件获取特征
        if os.path.exists('custom_attributes.json'):
            with open('custom_attributes.json', 'r', encoding='utf-8') as f:
                features = json.load(f)
                return features
        
        # 如果文件不存在，返回默认特征列表
        return [
            "纽扣", "拉链", "V领", "圆领", "翻领", "格子", "条纹", "纯色", 
            "图案", "蕾丝", "水洗", "亮片", "刺绣", "印花", "束腰", "松紧", 
            "褶皱", "抽绳", "背带", "拼接", "开衩", "系带", "荷叶边", "褶边", 
            "镂空", "口袋"
        ]
    except Exception as e:
        print(f"获取默认特征时出错: {e}")
        # 出错时返回基本默认值
        return ["特征1", "特征2", "特征3", "特征4", "特征5"] 