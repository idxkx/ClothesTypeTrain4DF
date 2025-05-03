import streamlit as st
import time

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