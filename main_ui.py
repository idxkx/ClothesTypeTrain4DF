# Placeholder for Streamlit UI code 

# 解决PyTorch和Streamlit之间的兼容性问题
import torch_patch  # 导入补丁
import warnings
import sys
import os

# 忽略PyTorch相关警告和错误
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Tried to instantiate class \"__path__._path\".*")
warnings.filterwarnings("ignore", message=".*torch.classes.__path__.*")
warnings.filterwarnings("ignore", message=".*torch._C.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")

# 添加当前目录到Python模块搜索路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 处理asyncio事件循环
import asyncio
import nest_asyncio

# 应用nest_asyncio以允许嵌套事件循环
try:
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 导入框架和库
import streamlit as st
import torch
import time
import pandas as pd
import math
import traceback
import json
import platform
import subprocess
import uuid
from datetime import datetime, timedelta
from torchvision import transforms

# --- 组件导入 ---
from components.config_panel import create_config_panel
from components.training_panel import start_training_process, update_training_ui
from components.history_viewer import display_history, render_action_buttons
from components.gpu_monitor import update_gpu_info, initialize_gpu
from components.metadata_manager import display_metadata_viewer, display_metadata_creator, create_metadata_form
from components.report_generator import batch_generate_metadata
from utils.state_manager import initialize_session_state
from utils.file_utils import safe_path, load_results, save_results
from utils.time_utils import format_time_delta
from utils.path_manager import load_config_paths

# --- 环境检测 ---
IS_WINDOWS = platform.system().lower().startswith('win')
IS_LINUX = platform.system().lower() == 'linux'

# --- 配置文件常量 ---
RESULTS_FILE = "training_results.json"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

# --- GPU初始化 ---
nvml_available, pynvml, gpu_names = initialize_gpu()

# --- 路径配置 ---
ANNO_DIR, IMG_DIR = load_config_paths(CONFIG_PATH)

# --- 模型和训练器导入 ---
try:
    from model import ClothesModel
    from trainer import Trainer
    from dataset import DeepFashionDataset
except ImportError as e:
    st.error(f"错误：无法导入必要的模块。请确保 model.py, trainer.py, dataset.py 与 main_ui.py 在同一目录下。错误: {e}")
    st.stop()

# 修复torch.classes.__path__问题
if hasattr(torch, 'classes'):
    # 定义补丁类
    if not hasattr(sys, '_ModulePathPatchDefined'):
        # 只在第一次定义
        class ModulePathPatch:
            def __init__(self):
                self._path = []
            
            def __iter__(self):
                return iter(self._path)
        
        # 标记已定义
        sys._ModulePathPatchDefined = True
        # 保存类引用以便重用
        sys._ModulePathPatchClass = ModulePathPatch
    else:
        # 重用已定义的类
        ModulePathPatch = sys._ModulePathPatchClass
    
    # 应用补丁
    if not hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = ModulePathPatch()

def main():
    """主应用入口函数"""
    # --- 修复PyTorch和Streamlit兼容性问题 ---
    if hasattr(torch, 'classes') and not hasattr(torch.classes, '__path__'):
        if hasattr(sys, '_ModulePathPatchClass'):
            # 使用已定义的补丁类
            torch.classes.__path__ = sys._ModulePathPatchClass()
    
    # --- 增加额外的asyncio事件循环处理 ---
    try:
        # 尝试获取当前事件循环，如果不存在则创建一个
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # 再次应用nest_asyncio以处理嵌套循环
        nest_asyncio.apply()
    except Exception as e:
        # 异常不会影响主程序运行，只打印警告
        print(f"事件循环初始化警告（可忽略）: {e}")
    
    # --- 页面配置 ---
    st.set_page_config(page_title="喵搭服装识别训练场", layout="wide")
    st.title("👕喵搭👗服装识别模型训练场")
    st.markdown("--- ")
    
    # --- 初始化会话状态 ---
    initialize_session_state()
    
    # --- 训练控制移到侧边栏最顶部 ---
    st.sidebar.subheader("🚀 训练控制")
    train_button_col, stop_button_col = st.sidebar.columns(2)
    start_training = train_button_col.button("开始训练！", use_container_width=True, key="start_training_btn")
    stop_training = stop_button_col.button("停止训练", use_container_width=True, key="stop_training_btn")
    
    if stop_training:
        st.session_state.stop_requested = True
        st.sidebar.warning("收到停止请求...将在当前轮次结束后尝试停止。")
    
    st.sidebar.markdown("---")
    
    # --- 创建侧边栏配置面板 ---
    selected_device, selected_gpu_index, training_params = create_config_panel(
        ANNO_DIR, IMG_DIR, CONFIG_PATH, nvml_available, gpu_names)
    
    # --- 创建主界面 ---
    col_main_1, col_main_2 = st.columns([2, 1])  # 状态/图表区 vs 日志区
    
    with col_main_1:
        st.subheader("📊 训练状态与指标")
        status_placeholder = st.empty()
        overall_progress_bar = st.progress(0.0)
        epoch_info_placeholder = st.empty()
        time_info_placeholder = st.empty()
        diagnostic_report_placeholder = st.empty()
        functional_test_placeholder = st.empty()
        loss_chart_placeholder = st.empty()
        acc_chart_placeholder = st.empty()
    
    with col_main_2:
        st.subheader("📜 训练日志")
        log_placeholder = st.empty()
        log_placeholder.code("", language='log')
    
    # --- 创建GPU监控区域 ---
    st.sidebar.markdown("--- ")
    st.sidebar.subheader("📈 GPU 监控")
    gpu_info_placeholder = st.sidebar.empty()
    gpu_util_chart_placeholder = st.sidebar.empty()
    gpu_mem_chart_placeholder = st.sidebar.empty()
    
    # 更新GPU监控
    if selected_gpu_index is not None:
        gpu_chart_placeholders = (gpu_util_chart_placeholder, gpu_mem_chart_placeholder)
        update_gpu_info(selected_gpu_index, gpu_info_placeholder, gpu_chart_placeholders)
    
    # --- 触发训练过程 ---
    if start_training:
        ui_components = {
            'status': status_placeholder,
            'progress': overall_progress_bar,
            'epoch_info': epoch_info_placeholder,
            'time_info': time_info_placeholder,
            'diagnostic': diagnostic_report_placeholder,
            'functional_test': functional_test_placeholder,
            'loss_chart': loss_chart_placeholder,
            'acc_chart': acc_chart_placeholder,
            'log': log_placeholder,
            'gpu_info': gpu_info_placeholder,
            'gpu_charts': (gpu_util_chart_placeholder, gpu_mem_chart_placeholder)
        }
        
        start_training_process(
            training_params,
            selected_device,
            selected_gpu_index,
            ui_components,
            ANNO_DIR,
            IMG_DIR,
            RESULTS_FILE
        )
    elif not st.session_state.get('log_messages'):
        status_placeholder.info("请在左侧设置参数并点击 '开始训练！' 按钮。")
        log_placeholder.code("尚未开始训练...", language='log')
    
    # --- 历史记录区域 ---
    st.markdown("--- ")
    st.subheader("📜 历史训练对比")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔄 批量生成缺失的元数据", key="batch_generate_metadata_btn"):
            with st.spinner("正在生成元数据..."):
                results = batch_generate_metadata()
                st.write("生成结果:")
                for model_name, success, message in results:
                    if success:
                        st.success(f"✅ {model_name}: {message}")
                    else:
                        st.warning(f"⚠️ {model_name}: {message}")
            st.rerun()  # 刷新显示
    
    # 确保会话状态中有show_failed
    if "show_failed" not in st.session_state:
        st.session_state.show_failed = True
    
    with col2:
        st.session_state.show_failed = st.checkbox(
            "显示失败的训练",
            value=st.session_state.show_failed,
            key="main_ui_show_failed_checkbox",
            help="勾选显示训练失败的记录"
        )
    
    # 显示历史记录
    try:
        display_history()
    except Exception as e:
        st.error(f"显示历史记录时出错: {e}")
        st.info("请尝试重新启动应用或检查训练结果文件是否损坏")
    
    # --- 元数据查看区域 ---
    with st.expander("🔍 查看模型元数据", expanded=False):
        display_metadata_viewer()
    
    # --- 手动创建元数据区域 ---
    with st.expander("🛠️ 手动创建元数据", expanded=False):
        display_metadata_creator()

if __name__ == "__main__":
    main()
