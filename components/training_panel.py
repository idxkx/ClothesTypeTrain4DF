import streamlit as st
import torch
import time
import pandas as pd
import math
import traceback
import os
import json
from datetime import datetime, timedelta
from torchvision import transforms

from utils.state_manager import append_log, reset_training_state
from utils.time_utils import format_time_delta, format_datetime
from utils.file_utils import ensure_dir_exists, load_results, save_results
from utils.path_manager import get_model_save_dir
from components.gpu_monitor import update_gpu_info, shutdown_gpu
from components.report_generator import generate_diagnostic_report, run_functional_test

def start_training_process(training_params, selected_device, selected_gpu_index, 
                          ui_components, anno_dir, img_dir, results_file):
    """启动训练流程"""
    # 检查GPU设备是否有效
    if selected_device is None or not selected_device.startswith('cuda'):
        error_msg = "错误：必须选择一个有效的 GPU 设备才能开始训练。请检查侧边栏的设备选择。"
        st.error(error_msg)
        append_log(error_msg)
        ui_components['status'].error("❌ 训练无法开始：未选择有效 GPU。")
        return
    
    # 重置训练状态
    reset_training_state()
    
    # 清空UI显示区域
    for key in ['diagnostic', 'functional_test', 'time_info']:
        if key in ui_components:
            ui_components[key].empty()
    
    # 初始化显示
    ui_components['status'].info("⏳ 正在准备训练环境...")
    ui_components['progress'].progress(0.0)
    ui_components['epoch_info'].empty()
    ui_components['log'].code("\n".join(st.session_state.log_messages), language='log')
    
    # 更新GPU监控
    if selected_gpu_index is not None:
        update_gpu_info(selected_gpu_index, ui_components['gpu_info'], ui_components['gpu_charts'])
    
    # 准备模型保存目录
    model_name = training_params['model_name']
    model_save_dir = get_model_save_dir(model_name)
    ensure_dir_exists(model_save_dir)
    
    # 收集训练参数
    args = {
        'epochs': training_params['epochs'],
        'batch_size': training_params['batch_size'],
        'learning_rate': training_params['learning_rate'],
        'device': selected_device,
        'model_save_path': model_save_dir,
        'attribute_loss_weight': training_params['attribute_loss_weight'],
        'num_workers': training_params['num_workers']
    }
    
    append_log(f"使用的训练参数: {args}")
    append_log(f"使用的骨干网络: {training_params['backbone']}")
    append_log(f"Anno Dir Path: {anno_dir}")
    append_log(f"Image Dir Path: {img_dir}")
    
    # 定义图像转换
    transforms_config = _prepare_transforms()
    
    # 运行训练过程
    try:
        _run_training_pipeline(
            training_params, args, transforms_config, 
            anno_dir, img_dir, model_save_dir, selected_device, selected_gpu_index,
            ui_components, results_file
        )
    except Exception as e:
        error_msg = f"训练过程中发生错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        traceback.print_exc()

def _prepare_transforms():
    """准备图像转换"""
    # 对于预训练模型，通常使用 ImageNet 的均值和标准差
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # 定义训练和验证的转换流程
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # 可以在这里添加数据增强，例如：
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    append_log("已定义图像转换流程 (Resize, ToTensor, Normalize)")
    
    return {
        'train': train_transform,
        'val': val_transform
    }

def _run_training_pipeline(training_params, args, transforms_config, 
                         anno_dir, img_dir, model_save_dir, selected_device, selected_gpu_index,
                         ui_components, results_file):
    """运行完整的训练流程"""
    # 初始化数据集
    train_dataset, val_dataset = _initialize_datasets(
        anno_dir, img_dir, transforms_config, ui_components
    )
    if train_dataset is None or val_dataset is None:
        return
    
    # 初始化模型
    model = _initialize_model(
        training_params['backbone'], selected_device, ui_components
    )
    if model is None:
        return
    
    # 初始化训练器
    trainer = _initialize_trainer(
        model, train_dataset, val_dataset, args, ui_components
    )
    if trainer is None:
        return
    
    # 开始训练
    current_run_result = _initialize_run_result(training_params, selected_device)
    
    training_success, best_val_loss, history_df = _execute_training(
        trainer, selected_device, selected_gpu_index,
        ui_components, current_run_result
    )
    
    # 完成后处理
    _finalize_training(
        training_success, history_df, best_val_loss, model_save_dir, training_params, 
        current_run_result, ui_components, selected_device, results_file
    )
    
def update_training_ui(epoch, epochs, train_metrics, val_metrics, epoch_time, ui_components):
    """更新训练UI"""
    # 更新摘要信息
    epoch_summary = (
        f"**Epoch {epoch+1}/{epochs}** | "
        f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}% | "
        f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )
    ui_components['epoch_info'].markdown(epoch_summary)
    
    # 更新历史数据
    current_epoch_history = {
        'epoch': epoch + 1,
        'Train Loss': train_metrics['loss'] if math.isfinite(train_metrics['loss']) else None,
        'Validation Loss': val_metrics['loss'] if math.isfinite(val_metrics['loss']) else None,
        'Train Accuracy (%)': train_metrics['accuracy'],
        'Validation Accuracy (%)': val_metrics['accuracy']
    }
    st.session_state.history_df_list.append(current_epoch_history)
    
    # 更新图表
    history_df = pd.DataFrame(st.session_state.history_df_list).dropna(subset=['epoch'])
    if not history_df.empty:
        plot_df_loss = history_df[['epoch', 'Train Loss', 'Validation Loss']].set_index('epoch')
        plot_df_acc = history_df[['epoch', 'Train Accuracy (%)', 'Validation Accuracy (%)']].set_index('epoch')
        with ui_components['loss_chart'].container():
            st.line_chart(plot_df_loss)
        with ui_components['acc_chart'].container():
            st.line_chart(plot_df_acc)
    
    # 更新进度条
    ui_components['progress'].progress((epoch + 1) / epochs)
    
    # 更新时间信息
    _update_time_info(epoch, epochs, epoch_time, ui_components['time_info'])
    
    # 更新日志
    ui_components['log'].code("\n".join(st.session_state.log_messages), language='log')

def _update_time_info(epoch, total_epochs, epoch_time, time_info_placeholder):
    """更新时间信息"""
    # 记录当前轮次耗时
    st.session_state.epoch_durations.append(epoch_time)
    
    # 计算时间数据
    if st.session_state.training_start_time is not None:
        elapsed_time = time.time() - st.session_state.training_start_time
        avg_epoch_time = sum(st.session_state.epoch_durations) / len(st.session_state.epoch_durations)
        remaining_epochs = total_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs if avg_epoch_time is not None and remaining_epochs > 0 else None
        eta_time = datetime.now() + timedelta(seconds=int(eta_seconds)) if eta_seconds is not None else None
        
        time_info_str = (
            f"⏱️ **时间统计:**  "
            f"已运行: {format_time_delta(elapsed_time)} | "
            f"平均每轮: {avg_epoch_time:.2f} 秒 | "
            f"预计剩余: {format_time_delta(eta_seconds)} | "
            f"预计完成时间: {eta_time.strftime('%Y-%m-%d %H:%M:%S') if eta_time else 'N/A'}"
        )
        time_info_placeholder.markdown(time_info_str)

# 训练流程子函数部分导入（从另一个文件中定义）
from components.training_functions import (
    _initialize_datasets,
    _initialize_model,
    _initialize_trainer,
    _initialize_run_result,
    _execute_training,
    _finalize_training
) 