import streamlit as st
import os
import pandas as pd
import json
import uuid
import time
import math
from datetime import datetime

from utils.file_utils import load_results, save_results
from components.report_generator import generate_metadata_for_model
from utils.time_utils import format_time_delta

def load_record_sets():
    """加载所有可用的记录集"""
    record_sets = {}
    record_sets_dir = "training_record_sets"
    
    if os.path.exists(record_sets_dir):
        for file in os.listdir(record_sets_dir):
            if file.endswith('.json'):
                name = file[:-5]  # 移除.json后缀
                try:
                    with open(os.path.join(record_sets_dir, file), 'r', encoding='utf-8') as f:
                        records = json.load(f)
                    record_sets[name] = records
                except Exception as e:
                    st.warning(f"无法加载记录集 {name}: {e}")
    
    return record_sets

def switch_to_record_set(name, records):
    """切换到指定的记录集"""
    try:
        # 备份当前记录
        backup_dir = "training_records_backup"
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"training_results_{timestamp}.json")
        
        with open("training_results.json", 'r', encoding='utf-8') as f:
            current_records = json.load(f)
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(current_records, f, indent=4, ensure_ascii=False)
        
        # 切换到新记录集
        with open("training_results.json", 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        
        return True, f"已切换到记录集: {name}"
    except Exception as e:
        return False, f"切换记录集失败: {e}"

def display_record_sets_manager():
    """显示记录集管理界面"""
    st.subheader("📚 训练记录集管理")
    
    # 加载所有记录集
    record_sets = load_record_sets()
    
    if not record_sets:
        st.info("还没有创建任何记录集。记录集将在训练完成后自动创建。")
        return
    
    # 创建两列布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 显示记录集选择器
        selected_set = st.selectbox(
            "选择记录集",
            options=list(record_sets.keys()),
            format_func=lambda x: f"{x} ({len(record_sets[x])} 个模型记录)"
        )
        
        if selected_set:
            records = record_sets[selected_set]
            st.write(f"记录集 '{selected_set}' 包含以下模型：")
            
            # 创建数据表格
            if records:
                df = pd.DataFrame(records)
                if 'date_created' in df.columns:
                    df['date_created'] = pd.to_datetime(df['date_created'], errors='coerce')
                    df = df.sort_values('date_created', ascending=False)
                
                # 选择要显示的列
                display_columns = [
                    'model_name', 'backbone', 'completed_epochs',
                    'date_created', 'learning_rate', 'status'
                ]
                display_columns = [col for col in display_columns if col in df.columns]
                
                st.dataframe(df[display_columns])
            else:
                st.info("此记录集为空")
    
    with col2:
        # 切换记录集的按钮
        if st.button(f"切换到 '{selected_set}'", key="switch_record_set"):
            success, message = switch_to_record_set(selected_set, record_sets[selected_set])
            if success:
                st.success(message)
                st.rerun()  # 重新加载页面以显示新记录
            else:
                st.error(message)

def display_history():
    """显示训练历史记录"""
    # 首先显示记录集管理器
    display_record_sets_manager()
    
    st.markdown("---")
    st.subheader("📊 当前训练记录")
    
    # 加载当前的训练记录
    results = load_results()
    
    if not results:
        st.info("还没有任何训练记录。")
        return
    
    # 转换为DataFrame以便显示
    df = pd.DataFrame(results)
    
    # 确保date_created列是datetime类型
    if 'date_created' in df.columns:
        df['date_created'] = pd.to_datetime(df['date_created'], errors='coerce')
        df = df.sort_values('date_created', ascending=False)
    
    # 添加筛选选项
    col1, col2 = st.columns([2, 1])
    with col1:
        # 添加backbone筛选
        if 'backbone' in df.columns:
            backbones = ['全部'] + sorted(df['backbone'].unique().tolist())
            selected_backbone = st.selectbox('选择backbone:', backbones)
    
    with col2:
        # 显示失败记录的选项
        show_failed = st.checkbox('显示失败的训练', value=st.session_state.get('show_failed', True))
    
    # 应用筛选
    if selected_backbone != '全部':
        df = df[df['backbone'] == selected_backbone]
    if not show_failed:
        df = df[df['status'] != 'failed']
    
    # 选择要显示的列
    display_columns = [
        'model_name', 'backbone', 'completed_epochs',
        'date_created', 'learning_rate', 'status'
    ]
    display_columns = [col for col in display_columns if col in df.columns]
    
    # 显示数据表格
    st.dataframe(df[display_columns])
    
    # 显示详细信息
    if len(df) > 0:
        st.markdown("### 📝 详细信息")
        for _, row in df.iterrows():
            with st.expander(f"🔍 {row['model_name']}"):
                st.json(row.to_dict())

def delete_training_record(model_name, results_file="training_results.json"):
    """删除指定的训练记录
    
    参数:
        model_name: 要删除的模型名称
        results_file: 训练结果文件路径
    
    返回:
        (success, message): 操作是否成功的布尔值和相关消息
    """
    try:
        print(f"正在尝试删除模型记录: {model_name}")
        
        # 加载所有训练记录
        all_results = load_results(results_file)
        print(f"已加载训练记录，共 {len(all_results)} 条")
        
        # 查找要删除的记录索引
        index_to_delete = None
        for i, record in enumerate(all_results):
            if record.get("model_name") == model_name:
                index_to_delete = i
                break
        
        # 如果没有找到记录，返回错误
        if index_to_delete is None:
            print(f"未找到模型 '{model_name}' 的训练记录")
            return False, f"未找到模型 '{model_name}' 的训练记录"
        
        # 获取记录信息，用于确认和显示
        record_to_delete = all_results[index_to_delete]
        print(f"找到要删除的记录，索引: {index_to_delete}")
        
        # 删除记录
        del all_results[index_to_delete]
        print(f"已从内存中删除记录，现在尝试保存到文件")
        
        # 保存更新后的记录列表
        success = save_results(all_results, results_file)
        if not success:
            print("保存更新后的训练记录失败")
            return False, "保存更新后的训练记录失败"
        
        print(f"成功删除模型 '{model_name}' 的训练记录并保存")
        return True, f"已成功删除模型 '{model_name}' 的训练记录"
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"删除训练记录时出错: {e}")
        print(f"详细错误: {error_trace}")
        return False, f"删除训练记录时出错: {e}"

def render_action_buttons(row_data):
    """为每个训练记录渲染操作按钮"""
    model_info = row_data["操作"]
    model_name = model_info["model_name"]
    has_metadata = model_info["metadata_exists"]
    model_path = model_info["model_path"]
    original_data = model_info["original_data"]
    
    # 确保模型路径有效
    if not model_path or not os.path.exists(model_path):
        has_metadata = False
    
    # 获取行索引，确保UI元素的唯一性
    # 使用完成时间作为区分不同记录的标识符
    end_time = row_data.get("完成时间", "")
    record_id = f"{model_name}_{end_time}"
    
    # 初始化会话状态变量
    if "deletion_target" not in st.session_state:
        st.session_state.deletion_target = None
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        if has_metadata:
            # 如果已有元数据，显示查看按钮
            if st.button("📄 查看元数据", key=f"view_metadata_{record_id}", help=f"查看 {model_name} 的元数据文件", use_container_width=True):
                view_model_metadata(model_name, model_path)
        else:
            # 如果没有元数据，显示创建按钮
            if st.button("➕ 创建元数据", key=f"create_metadata_{record_id}", help=f"为 {model_name} 创建元数据文件", use_container_width=True):
                try:
                    success, message = generate_metadata_for_model(original_data)
                    if success:
                        st.success(f"✅ {model_name} 的元数据创建成功！")
                        # 刷新显示
                        st.rerun()
                    else:
                        st.error(f"❌ 创建失败: {message}")
                except Exception as e:
                    st.error(f"创建元数据时发生错误: {e}")
    
    with col2:
        # 添加查看详情按钮
        if st.button("📊 查看详情", key=f"view_details_{record_id}", help=f"查看 {model_name} 的训练详情", use_container_width=True):
            st.session_state.selected_model_for_details = model_name
            st.rerun()
    
    # 检查是否是当前选中的删除目标
    is_delete_target = st.session_state.deletion_target == record_id
    
    # 使用单独的容器而不是嵌套列来显示删除按钮和确认操作
    delete_container = st.container()
    
    if not is_delete_target:
        # 显示删除按钮
        if delete_container.button("🗑️ 删除记录", key=f"delete_{record_id}", help=f"删除 {model_name} 的训练记录", use_container_width=True):
            # 设置当前记录为删除目标
            st.session_state.deletion_target = record_id
            st.rerun()
    else:
        # 显示确认对话框
        delete_container.warning(f"确定要删除模型 '{model_name}' 的训练记录吗？此操作不可撤销。")
        
        # 使用水平布局放置按钮，但不使用列
        confirm_btn = delete_container.button("✓ 确认删除", key=f"confirm_yes_{record_id}", help=f"确认删除此记录", use_container_width=False)
        cancel_btn = delete_container.button("✗ 取消", key=f"confirm_no_{record_id}", help=f"取消删除操作", use_container_width=False)
        
        if confirm_btn:
            # 执行删除操作
            try:
                success, message = delete_training_record(model_name)
                if success:
                    st.success(message)
                    # 重置删除目标
                    st.session_state.deletion_target = None
                    # 刷新显示
                    st.rerun()
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"删除失败: {e}")
                # 记录错误到控制台以便调试
                print(f"删除记录时出现错误: {e}")
        
        if cancel_btn:
            # 取消删除，重置目标
            st.session_state.deletion_target = None
            st.rerun()

def view_model_metadata(model_name, model_path):
    """查看模型元数据"""
    try:
        if not model_path or not os.path.exists(model_path):
            st.warning(f"⚠️ 模型文件不存在: {model_path}")
            return
            
        model_dir = os.path.dirname(model_path)
        if not model_dir:
            st.warning(f"⚠️ 无法获取模型目录")
            return
            
        if not model_name:
            st.warning(f"⚠️ 模型名称为空")
            return
            
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        if not metadata_file or not os.path.exists(metadata_file):
            st.warning(f"⚠️ 未找到 {model_name} 的元数据文件 ({metadata_file})")
            return
            
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # 使用expander显示元数据
        with st.expander("📋 元数据内容", expanded=True):
            st.json(metadata)
            
        st.success(f"✅ 已加载 {model_name} 的元数据")
    except Exception as e:
        st.error(f"读取元数据时发生错误: {e}")

def show_model_details(model_name, all_results):
    """显示模型详细信息"""
    if not model_name:
        st.warning("无法显示详情：模型名称为空")
        return
        
    # 查找选中的模型记录
    selected_result = next(
        (r for r in all_results if r.get("model_name") == model_name),
        None
    )
    
    if not selected_result:
        st.warning(f"未找到 {model_name} 的训练记录")
        return
        
    # 显示基本信息
    st.markdown(f"#### 📌 基本信息")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown(f"**模型名称:** {selected_result.get('model_name', 'N/A')}")
        st.markdown(f"**训练策略:** {selected_result.get('strategy', 'N/A')}")
    with info_col2:
        st.markdown(f"**开始时间:** {selected_result.get('start_time_str', 'N/A')}")
        st.markdown(f"**结束时间:** {selected_result.get('end_time_str', 'N/A')}")
    with info_col3:
        st.markdown(f"**训练状态:** {selected_result.get('status', 'N/A')}")
        st.markdown(f"**总耗时:** {selected_result.get('duration_str', 'N/A')}")

    # 显示训练参数
    with st.expander("🔧 训练参数", expanded=False):
        params = selected_result.get('parameters', {})
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            st.markdown(f"**学习率:** {params.get('learning_rate', 'N/A')}")
            st.markdown(f"**批次大小:** {params.get('batch_size', 'N/A')}")
        with param_col2:
            st.markdown(f"**训练轮数:** {params.get('epochs', 'N/A')}")
            st.markdown(f"**设备:** {params.get('device', 'N/A')}")

    # 添加图表类型选择
    chart_type = st.selectbox(
        "选择查看的指标:",
        ["损失曲线", "准确率曲线", "GPU使用率", "训练日志"],
        key="history_chart_type"
    )

    # 显示选择的图表或日志
    st.markdown(f"#### 📈 {chart_type}")
    
    if chart_type == "损失曲线":
        show_loss_chart(selected_result)
    elif chart_type == "准确率曲线":
        show_accuracy_chart(selected_result)
    elif chart_type == "GPU使用率":
        show_gpu_chart(selected_result)
    else:  # 训练日志
        show_training_log(selected_result)

    # 显示诊断报告
    with st.expander("🩺 训练诊断报告", expanded=False):
        diagnostic_summary = selected_result.get('diagnostic_summary', None)
        if diagnostic_summary:
            st.markdown(diagnostic_summary)
        else:
            st.info("没有可用的诊断报告")

def show_loss_chart(model_result):
    """显示损失曲线"""
    history_data = model_result.get('training_history', [])
    if not history_data:
        st.info("没有保存训练历史数据")
        return
        
    history_df = pd.DataFrame(history_data)
    if history_df.empty or 'Train Loss' not in history_df.columns or 'Validation Loss' not in history_df.columns:
        st.info("没有可用的损失数据")
        return
        
    st.line_chart(history_df[['Train Loss', 'Validation Loss']].set_index('epoch'))

def show_accuracy_chart(model_result):
    """显示准确率曲线"""
    history_data = model_result.get('training_history', [])
    if not history_data:
        st.info("没有保存训练历史数据")
        return
        
    history_df = pd.DataFrame(history_data)
    if history_df.empty or 'Train Accuracy (%)' not in history_df.columns or 'Validation Accuracy (%)' not in history_df.columns:
        st.info("没有可用的准确率数据")
        return
        
    st.line_chart(history_df[['Train Accuracy (%)', 'Validation Accuracy (%)']].set_index('epoch'))

def show_gpu_chart(model_result):
    """显示GPU使用率图表"""
    gpu_history = model_result.get('gpu_metrics_history', [])
    if not gpu_history:
        st.info("没有保存GPU监控数据")
        return
        
    gpu_df = pd.DataFrame(gpu_history)
    if gpu_df.empty:
        st.info("没有可用的GPU使用率数据")
        return
        
    st.line_chart(gpu_df[['GPU Utilization (%)', 'Memory Utilization (%)']].set_index('step'))

def show_training_log(model_result):
    """显示训练日志"""
    log_messages = model_result.get('log_messages', [])
    if not log_messages:
        st.info("没有保存训练日志")
        return
        
    st.code("\n".join(log_messages), language="log") 