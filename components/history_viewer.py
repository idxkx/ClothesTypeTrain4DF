import streamlit as st
import os
import pandas as pd
import json
import uuid
import time

from utils.file_utils import load_results, save_results
from components.report_generator import generate_metadata_for_model

def display_history():
    """显示历史训练记录"""
    all_results = load_results()
    
    # 设置显示失败记录的选项
    if "show_failed" not in st.session_state:
        st.session_state.show_failed = False
        
    show_failed = st.checkbox(
        "显示失败的训练记录", 
        value=st.session_state.show_failed,
        key="show_failed_checkbox",
        help="勾选此项可显示状态为'失败'或'错误'的训练记录"
    )
    
    # 更新会话状态
    st.session_state.show_failed = show_failed
    
    if not all_results:
        st.info("尚未有训练记录。开始一次训练后，结果将显示在这里。")
        return

    # 转换数据以便更好地显示
    display_data = []
    for r in reversed(all_results):  # 显示最新的在前面
        # 如果未勾选显示失败记录，则跳过失败的记录
        if not st.session_state.show_failed and r.get("status", "").lower() in ["失败", "错误", "failed", "error"]:
            continue

        best_epoch_info = f"{r.get('best_val_loss', 'N/A'):.4f} @ E{r.get('best_epoch', 'N/A')}" if isinstance(r.get('best_val_loss'), (int, float)) else "N/A"
        
        # 检查是否存在元数据文件
        metadata_status = "⚠️ 未找到"
        model_path = r.get("best_model_path", "")
        metadata_file = None
        if model_path and os.path.exists(model_path):
            model_dir = os.path.dirname(model_path)
            model_name = r.get("model_name", "")
            metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
            if os.path.exists(metadata_file):
                metadata_status = "✅ 已生成"
        
        display_data.append({
            "完成时间": r.get("end_time_str", "N/A"),
            "模型名称": r.get("model_name", "N/A"),
            "策略": r.get("strategy", "N/A"),
            "骨干网络": r.get("backbone", "N/A"),
            "轮数": f"{r.get('completed_epochs', 'N/A')}/{r.get('total_epochs', 'N/A')}",
            "最佳验证损失 (轮)": best_epoch_info,
            "状态": r.get("status", "N/A").split('.')[0],  # 取第一句
            "总耗时": r.get("duration_str", "N/A"),
            "元数据": metadata_status,
            "功能测试": r.get("functional_test_result", "未执行"),
            "操作": {
                "model_name": r.get("model_name", ""),
                "metadata_exists": os.path.exists(metadata_file) if metadata_file else False,
                "model_path": model_path,
                "original_data": r  # 保存原始数据以供后续使用
            }
        })

    if display_data:
        # 创建DataFrame
        results_df = pd.DataFrame(display_data)
        
        # 显示主表格（不包含操作列）
        st.markdown("### 📋 历史训练记录")
        display_cols = [col for col in results_df.columns if col != "操作"]
        st.dataframe(
            results_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # 显示操作区域
        st.markdown("### ⚡ 快速操作")
        
        # 创建三列布局显示操作按钮
        num_records = len(results_df)
        cols_per_row = 3
        num_rows = (num_records + cols_per_row - 1) // cols_per_row
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                idx = row * cols_per_row + col
                if idx < num_records:
                    with cols[col]:
                        record = results_df.iloc[idx]
                        # 显示模型基本信息
                        st.markdown(f"""
                        **{record['模型名称']}**  
                        训练时间: {record['完成时间']}  
                        策略: {record['策略']}  
                        状态: {record['状态']}
                        """)
                        # 显示操作按钮
                        render_action_buttons(record)
                        # 添加分隔线
                        st.markdown("---")

        # 添加详细信息查看区域
        if 'selected_model_for_details' in st.session_state and st.session_state.selected_model_for_details:
            st.markdown("### 📊 训练详情查看")
            show_model_details(st.session_state.selected_model_for_details, all_results)
    else:
        st.info("没有符合筛选条件的训练记录。")

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
        model_dir = os.path.dirname(model_path)
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_file):
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