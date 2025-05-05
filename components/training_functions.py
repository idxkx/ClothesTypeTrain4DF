import streamlit as st
import torch
import time
import pandas as pd
import math
import traceback
import os
import json
from datetime import datetime, timedelta
import re

from utils.state_manager import append_log
from utils.time_utils import format_time_delta, format_datetime
from utils.file_utils import save_results, load_results
from components.gpu_monitor import update_gpu_info

def _initialize_datasets(anno_dir, img_dir, transforms_config, ui_components):
    """初始化训练和验证数据集"""
    try:
        ui_components['status'].info("⏳ 正在加载数据集...")
        append_log("初始化训练数据集...")
        
        from dataset import DeepFashionDataset
        
        train_dataset = DeepFashionDataset(
            anno_dir_path=anno_dir,
            image_dir_path=img_dir,
            partition='train',
            transform=transforms_config['train']
        )
        append_log(f"训练集加载成功，样本数: {len(train_dataset)}")
        
        append_log("初始化验证数据集...")
        val_dataset = DeepFashionDataset(
            anno_dir_path=anno_dir,
            image_dir_path=img_dir,
            partition='val',
            transform=transforms_config['val']
        )
        append_log(f"验证集加载成功，样本数: {len(val_dataset)}")
        
        ui_components['status'].success("✅ 数据集加载完成!")
        return train_dataset, val_dataset
        
    except FileNotFoundError as e:
        error_msg = f"错误：找不到必要的数据文件或目录！请检查你输入的 Anno_fine 目录 '{anno_dir}' 和图片目录 '{img_dir}' 是否正确且存在。详细错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        ui_components['status'].error("❌ 数据集加载失败！")
        return None, None
        
    except ValueError as e:
        error_msg = f"错误：加载或解析分区文件时出错。请检查 '{anno_dir}' 目录下的分区文件 (如 train.txt, train_cate.txt) 是否存在且格式正确。详细错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        ui_components['status'].error("❌ 数据集加载失败！")
        return None, None
        
    except Exception as e:
        error_msg = f"错误：加载数据集时发生未知错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        ui_components['status'].error("❌ 数据集加载失败！")
        traceback.print_exc()
        return None, None

def _initialize_model(backbone, device, ui_components):
    """初始化模型"""
    try:
        ui_components['status'].info("⏳ 正在初始化模型...")
        append_log(f"初始化模型 (Backbone: {backbone})...")
        
        from model import ClothesModel
        
        num_categories = 50  # 可能需要修改为动态获取
        model = ClothesModel(
            num_categories=num_categories,
            backbone=backbone
        )
        
        append_log(f"模型初始化成功。将模型移动到设备: {device}")
        model.to(device)
        
        ui_components['status'].success("✅ 模型初始化完成!")
        return model
        
    except Exception as e:
        error_msg = f"错误：初始化模型 '{backbone}' 时发生错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        ui_components['status'].error("❌ 模型初始化失败！请检查选择的骨干网络是否可用或已安装 `timm` 库。")
        traceback.print_exc()
        return None

def _initialize_trainer(model, train_dataset, val_dataset, args, ui_components):
    """初始化训练器"""
    try:
        ui_components['status'].info("⏳ 正在初始化训练器...")
        append_log("初始化 Trainer...")
        
        from trainer import Trainer
        
        trainer = Trainer(model, train_dataset, val_dataset, args)
        append_log("Trainer 初始化成功.")
        
        ui_components['status'].success("✅ 训练器准备就绪!")
        return trainer
        
    except Exception as e:
        error_msg = f"错误：初始化 Trainer 时发生错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        ui_components['status'].error("❌ Trainer 初始化失败！")
        traceback.print_exc()
        return None

def _initialize_run_result(training_params, device):
    """初始化训练结果记录"""
    return {
        "start_time": st.session_state.training_start_time,
        "start_time_str": datetime.fromtimestamp(st.session_state.training_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "date_created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_name": training_params['model_name'],
        "strategy": training_params['strategy_choice'],
        "parameters": {k: v for k, v in training_params.items() if k != 'strategy_choice'},
        "backbone": training_params['backbone'],
        "anno_dir": training_params['anno_dir'],
        "img_dir": training_params['img_dir'],
        "status": "ongoing",
        "total_epochs": training_params['epochs'],
        "completed_epochs": 0,
        "best_val_loss": float('inf'),
        "best_epoch": None,
        "best_model_path": None,
        "diagnostic_summary": None,
        "functional_test_result": "未执行",
        "end_time": None,
        "end_time_str": None,
        "duration": None,
        "duration_str": None,
    }

def _execute_training(trainer, device, gpu_index, ui_components, current_run_result, results_file):
    """执行训练循环"""
    append_log("\n==================== 开始训练 ====================")
    ui_components['status'].info(f"🚀 模型训练中... 设备: {device}")
    
    best_val_loss = float('inf')
    best_model_file_path = None
    training_interrupted = False
    history_df = None
    
    try:
        for epoch in range(trainer.epochs):
            epoch_start_time = time.time()
            
            # 检查是否请求停止
            if st.session_state.get('stop_requested', False):
                append_log("训练被用户中断。")
                ui_components['status'].warning("⚠️ 训练已停止。")
                training_interrupted = True
                # 新增：中断时写入状态和时间
                current_run_result["status"] = "failed"
                if not current_run_result.get("date_created"):
                    current_run_result["date_created"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                all_results = load_results(results_file)
                all_results = [r for r in all_results if r.get('model_name') != current_run_result['model_name']]
                all_results.append(current_run_result)
                save_results(all_results, results_file)
                append_log(f"训练被中断，已保存记录到 {results_file}")
                break
            
            ui_components['status'].info(f"Epoch {epoch+1}/{trainer.epochs}: 正在训练...")
            append_log(f"\n--- 开始训练 Epoch {epoch+1}/{trainer.epochs} ---")
            
            # 更新GPU信息
            if gpu_index is not None:
                update_gpu_info(gpu_index, ui_components['gpu_info'], ui_components['gpu_charts'])
            
            # 训练阶段
            train_metrics = _train_epoch(trainer, epoch, ui_components)
            
            # 如果训练被中断，则跳出循环
            if st.session_state.get('stop_requested', False):
                append_log("训练在批次处理中被中断...")
                training_interrupted = True
                # 新增：中断时写入状态和时间
                current_run_result["status"] = "failed"
                if not current_run_result.get("date_created"):
                    current_run_result["date_created"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                all_results = load_results(results_file)
                all_results = [r for r in all_results if r.get('model_name') != current_run_result['model_name']]
                all_results.append(current_run_result)
                save_results(all_results, results_file)
                append_log(f"训练被中断，已保存记录到 {results_file}")
                break
            
            # 验证阶段
            val_metrics = _validate_epoch(trainer, ui_components)
            
            # 计算本轮耗时
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            
            # 更新界面
            from components.training_panel import update_training_ui
            update_training_ui(
                epoch, trainer.epochs, 
                train_metrics, val_metrics, 
                epoch_time, ui_components
            )
            
            # 更新GPU信息
            if gpu_index is not None:
                update_gpu_info(gpu_index, ui_components['gpu_info'], ui_components['gpu_charts'])
            
            # 保存最佳模型
            if math.isfinite(val_metrics['loss']) and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if trainer.model_save_path:
                    # 删除旧的最佳模型
                    _clean_old_model_files(trainer.model_save_path, current_run_result['model_name'])
                    
                    # 保存新的最佳模型
                    best_model_file_path = _save_best_model(
                        trainer, current_run_result['model_name'], epoch, val_metrics['loss']
                    )
                    
                    # 更新当前运行记录
                    current_run_result["best_val_loss"] = val_metrics['loss']
                    current_run_result["best_epoch"] = epoch + 1
                    current_run_result["best_model_path"] = best_model_file_path
            
            # 更新完成轮数
            current_run_result["completed_epochs"] = epoch + 1
            
            # 每轮结束后保存训练记录
            all_results = load_results(results_file)
            # 移除之前的相同模型记录（如果存在）
            all_results = [r for r in all_results if r.get('model_name') != current_run_result['model_name']]
            # 添加当前记录
            all_results.append(current_run_result)
            save_results(all_results, results_file)
            append_log(f"已保存当前训练进度到 {results_file}")
        
        # 训练完成，设置状态
        training_success = not training_interrupted
        
        if training_success:
            current_run_result["status"] = "completed"
            append_log("训练成功完成！")
        else:
            current_run_result["status"] = "failed"
            append_log("训练被中断！")
        
        # 构建历史DataFrame
        history_df = pd.DataFrame(st.session_state.history_df_list).dropna(subset=['epoch'])
        
        return training_success, best_val_loss, history_df
        
    except Exception as e:
        error_msg = f"错误：训练过程中发生严重错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        ui_components['status'].error("❌ 训练失败！")
        traceback.print_exc()
        current_run_result["status"] = "failed"
        if not current_run_result.get("date_created"):
            current_run_result["date_created"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 即使发生错误也保存训练记录
        all_results = load_results(results_file)
        all_results = [r for r in all_results if r.get('model_name') != current_run_result['model_name']]
        all_results.append(current_run_result)
        save_results(all_results, results_file)
        append_log(f"已保存训练错误记录到 {results_file}")
        return False, float('inf'), None

def _train_epoch(trainer, epoch, ui_components):
    """执行单轮训练"""
    trainer.model.train()
    train_loss, train_correct_cats, train_total_samples = 0.0, 0, 0
    num_batches = len(trainer.train_loader)
    
    for i, batch in enumerate(trainer.train_loader):
        # 每10个批次检查是否请求停止
        if i % 10 == 0 and st.session_state.get('stop_requested', False):
            return {
                'loss': float('inf'),
                'accuracy': 0.0
            }
        
        try:
            images = batch['image'].to(trainer.device, non_blocking=True)
            cat_labels = batch['category'].to(trainer.device, non_blocking=True)
            attr_labels = batch['attributes'].to(trainer.device, non_blocking=True)
            
            trainer.optimizer.zero_grad()
            cat_logits, attr_logits = trainer.model(images)
            
            valid_cat_mask = cat_labels != -1
            if valid_cat_mask.sum() == 0:
                loss_cat = torch.tensor(0.0).to(trainer.device)
            else:
                loss_cat = trainer.criterion_category(cat_logits[valid_cat_mask], cat_labels[valid_cat_mask])
                
            loss_attr = trainer.criterion_attribute(attr_logits, attr_labels)
            
            if not (torch.isfinite(loss_cat) and torch.isfinite(loss_attr)):
                append_log(f"警告: Epoch {epoch+1}, Batch {i+1}, 无效损失 (Cat: {loss_cat.item():.4f}, Attr: {loss_attr.item():.4f})，跳过.")
                continue
                
            loss = loss_cat + trainer.attribute_loss_weight * loss_attr
            loss.backward()
            trainer.optimizer.step()
            
            batch_size_actual = images.size(0)
            train_loss += loss.item() * batch_size_actual
            
            if valid_cat_mask.sum() > 0:
                _, predicted_cats = torch.max(cat_logits.data, 1)
                train_correct_cats += (predicted_cats[valid_cat_mask] == cat_labels[valid_cat_mask]).sum().item()
                train_total_samples += valid_cat_mask.sum().item()
            
            # 每50个批次更新状态
            if (i + 1) % 50 == 0 or (i + 1) == num_batches:
                current_avg_loss = train_loss / ((i + 1) * batch_size_actual) if batch_size_actual > 0 else 0
                ui_components['status'].info(f"Epoch {epoch+1}/{trainer.epochs}: 训练中... Batch {i+1}/{num_batches} ({(i + 1)*100/num_batches:.0f}%) | Batch Loss: {loss.item():.4f}")
                
        except Exception as batch_e:
            append_log(f"错误: Epoch {epoch+1}, Batch {i+1} 处理失败: {batch_e}")
            traceback.print_exc()
            continue
    
    # 计算平均损失和准确率
    avg_train_loss = train_loss / train_total_samples if train_total_samples > 0 else float('inf')
    avg_train_cat_acc = 100.0 * train_correct_cats / train_total_samples if train_total_samples > 0 else 0.0
    
    append_log(f"--- Epoch {epoch+1} 训练完成 (Avg Loss: {avg_train_loss:.4f}, Cat Acc: {avg_train_cat_acc:.2f}%) ---")
    
    return {
        'loss': avg_train_loss,
        'accuracy': avg_train_cat_acc
    }

def _validate_epoch(trainer, ui_components):
    """执行单轮验证"""
    if not trainer.val_loader or len(trainer.val_loader.dataset) == 0:
        append_log(f"--- 无验证集或验证集为空，跳过验证 ---")
        return {
            'loss': float('nan'),
            'accuracy': float('nan')
        }
    
    trainer.model.eval()
    val_loss, val_correct_cats, val_total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for i, batch in enumerate(trainer.val_loader):
            try:
                images = batch['image'].to(trainer.device, non_blocking=True)
                cat_labels = batch['category'].to(trainer.device, non_blocking=True)
                attr_labels = batch['attributes'].to(trainer.device, non_blocking=True)
                
                cat_logits, attr_logits = trainer.model(images)
                
                loss_cat = trainer.criterion_category(cat_logits, cat_labels)
                loss_attr = trainer.criterion_attribute(attr_logits, attr_labels)
                
                if not (torch.isfinite(loss_cat) and torch.isfinite(loss_attr)):
                    append_log(f"警告: 验证 Batch {i+1}, 无效损失，跳过.")
                    continue
                    
                loss = loss_cat + trainer.attribute_loss_weight * loss_attr
                
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                
                _, predicted_cats = torch.max(cat_logits.data, 1)
                val_correct_cats += (predicted_cats == cat_labels).sum().item()
                val_total_samples += batch_size
                
            except Exception as batch_e:
                append_log(f"错误: 验证 Batch {i+1} 处理失败: {batch_e}")
                continue
    
    # 计算平均损失和准确率
    avg_val_loss = val_loss / val_total_samples if val_total_samples > 0 else float('inf')
    avg_val_cat_acc = 100.0 * val_correct_cats / val_total_samples if val_total_samples > 0 else 0.0
    
    append_log(f"--- 验证完成 (Avg Loss: {avg_val_loss:.4f}, Cat Acc: {avg_val_cat_acc:.2f}%) ---")
    
    return {
        'loss': avg_val_loss,
        'accuracy': avg_val_cat_acc
    }

def _clean_old_model_files(model_save_path, model_name):
    """清理旧的最佳模型文件"""
    # 提取模型名称的基础部分，以便更灵活地匹配文件
    # 如果有时间戳格式 (_MMDD_HHMM)，去除它用于匹配
    base_model_name = re.sub(r'_\d{4}_\d{4}$', '', model_name)
    
    # 创建更强大的正则表达式模式进行匹配
    # 这允许匹配完整模型名或不带时间戳的基础名称
    pattern = rf"best_model_(({re.escape(model_name)})|({re.escape(base_model_name)}))(_\d+)?_epoch\d+\.pth"
    
    deleted_count = 0
    for old_file in os.listdir(model_save_path):
        if re.match(pattern, old_file):
            try:
                os.remove(os.path.join(model_save_path, old_file))
                append_log(f"已删除旧的最佳模型: {old_file}")
                deleted_count += 1
            except OSError as e:
                append_log(f"无法删除旧模型 {old_file}: {e}")
    
    append_log(f"清理完成，共删除 {deleted_count} 个旧模型文件")

def _save_best_model(trainer, model_name, epoch, val_loss):
    """保存最佳模型"""
    # 首先检查保存路径是否有效
    if not trainer.model_save_path:
        append_log("错误：模型保存路径无效，无法保存模型")
        return None
        
    # 为避免模型名称重复，添加当前时间（MMDD_HHMM格式）
    now = datetime.now()
    time_suffix = ""
    
    # 检查模型名称是否已包含时间戳，如果没有，添加时间戳
    if not re.search(r'_\d{4}_\d{4}$', model_name):
        time_suffix = f"_{now.strftime('%m%d_%H%M')}"
    
    # 构建完整的文件名，包括模型名和时间戳(如果需要)
    filename = f"best_model_{model_name}{time_suffix}_epoch{epoch+1}.pth"
    save_filename = os.path.join(trainer.model_save_path, filename)
    
    try:
        torch.save(trainer.model.state_dict(), save_filename)
        append_log(f"** 新的最佳模型已保存到: {save_filename} (Val Loss: {val_loss:.4f}) **")
        
        # 确认文件是否已成功创建
        if os.path.exists(save_filename):
            return save_filename
        else:
            append_log("警告：模型文件似乎未成功创建，路径可能无效")
            return None
    except Exception as e:
        append_log(f"错误：保存模型时出错: {e}")
        return None

def _finalize_training(training_success, history_df, best_val_loss, model_save_dir, 
                      training_params, current_run_result, ui_components, device, results_file):
    """完成训练后的收尾工作"""
    # 计算总训练时间
    end_time = time.time()
    total_time = end_time - current_run_result["start_time"]
    
    # 更新结果记录
    current_run_result["end_time"] = end_time
    current_run_result["end_time_str"] = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    current_run_result["duration"] = total_time
    current_run_result["duration_str"] = format_time_delta(total_time)
    
    # 记录最终日志
    append_log(f"总训练时间: {total_time:.2f} 秒")
    formatted_best_loss = f"{best_val_loss:.4f}" if math.isfinite(best_val_loss) else "N/A"
    append_log(f"最佳验证损失: {formatted_best_loss}")
    
    # 保存训练数据
    current_run_result = _save_training_data(current_run_result)
    
    # 显示最终时间信息
    _show_final_time_info(total_time, ui_components['time_info'])
    
    # 确保最后一次更新日志
    ui_components['log'].code("\n".join(st.session_state.log_messages), language='log')
    
    # 生成诊断报告
    if history_df is not None and not history_df.empty:
        from components.report_generator import generate_diagnostic_report
        diagnostic_report, evaluation_data = generate_diagnostic_report(history_df, best_val_loss, training_params['epochs'])
        ui_components['diagnostic'].markdown(diagnostic_report)
        append_log("\n--- 诊断报告已生成 ---")
        
        # 保存诊断报告文本和结构化评估数据
        current_run_result["diagnostic_summary"] = diagnostic_report
        current_run_result["evaluation"] = evaluation_data  # 添加结构化评估数据
    else:
        current_run_result["diagnostic_summary"] = "无训练历史数据"
        current_run_result["evaluation"] = {
            "overfitting_risk": "未知",
            "loss_diff": float('nan'),
            "accuracy_diff": float('nan'),
            "convergence_status": "未知"
        }
    
    # 确保训练状态为成功，无论功能测试是否通过
    if training_success:
        current_run_result["status"] = "completed"
        append_log("训练成功完成！")
    else:
        current_run_result["status"] = "failed"
        append_log("训练过程中出现错误。")
    
    # 执行功能测试，但不影响训练成功状态
    test_success = False
    if current_run_result.get("best_model_path") and os.path.exists(current_run_result["best_model_path"]):
        # 先确认元数据文件是否存在
        model_name = training_params['model_name']
        metadata_file = os.path.join(model_save_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_file):
            ui_components['functional_test'].error(f"元数据文件不存在: {metadata_file}")
            append_log(f"元数据文件不存在: {metadata_file}，功能测试将失败")
            ui_components['functional_test'].warning("请手动创建元数据文件后再进行功能测试，或使用fix_metadata.py工具")
            
            # 设置功能测试结果为失败，并提供详细原因
            current_run_result["functional_test_result"] = "failed"
            current_run_result["functional_test_error"] = {
                "error_type": "missing_metadata",
                "message": f"元数据文件不存在: {metadata_file}",
                "solution": "请手动创建元数据文件或使用fix_metadata.py工具"
            }
            return
        
        # 执行功能测试
        from components.report_generator import run_functional_test
        model_config = {
            'num_categories': 50,  # 假设类别数为50
            'backbone': training_params['backbone']
        }
        test_report, test_success, error_details = run_functional_test(
            model_save_dir, training_params['model_name'], model_config, device
        )
        ui_components['functional_test'].markdown(test_report)
        
        if not test_success and error_details:
            # 添加错误详情到测试报告中
            error_solution = error_details.get("solution", "请尝试重新训练模型或联系技术支持。")
            error_message = error_details.get("message", "未知错误")
            error_type = error_details.get("error_type", "unknown")
            
            # 构建详细错误信息
            error_info = f"""
            ### ❌ 功能测试失败详情
            - **错误类型**: {error_type}
            - **错误信息**: {error_message}
            - **建议解决方案**: {error_solution}
            """
            ui_components['functional_test'].error(error_info)
            
        log_summary = "功能测试成功" if test_success else f"功能测试失败: {error_details.get('error_type', '未知错误')}"
        append_log(f"\n--- 功能模拟测试 --- \n{log_summary}")
        current_run_result["functional_test_result"] = "success" if test_success else "failed"
        # 添加错误详情到训练结果
        if not test_success:
            current_run_result["functional_test_error"] = error_details
    else:
        if not current_run_result.get("best_model_path"):
            ui_components['functional_test'].warning("未找到有效的最佳模型路径，跳过功能测试。")
            append_log("未找到有效的最佳模型路径，跳过功能测试。")
            current_run_result["functional_test_result"] = "skipped (no model path)"
        elif not os.path.exists(current_run_result["best_model_path"]):
            ui_components['functional_test'].warning("模型文件不存在，跳过功能测试。")
            append_log(f"模型文件不存在: {current_run_result['best_model_path']}，跳过功能测试。")
            current_run_result["functional_test_result"] = "skipped (model file not found)"
    
    # 保存当前运行结果
    all_results = load_results(results_file)
    all_results.append(current_run_result)
    save_results(all_results, results_file)
    append_log(f"当前训练结果已追加到 {results_file}")
    
    from components.history_viewer import display_history
    display_history()
    
    # 清理GPU监控
    from components.gpu_monitor import shutdown_gpu
    shutdown_gpu()

def _save_training_data(current_run_result):
    """保存训练过程中的数据"""
    # 保存训练历史
    if 'history_df_list' in st.session_state:
        current_run_result['training_history'] = st.session_state.history_df_list
    
    # 保存GPU监控数据
    if 'gpu_metrics_history' in st.session_state:
        current_run_result['gpu_metrics_history'] = st.session_state.gpu_metrics_history
    
    # 保存训练日志
    if 'log_messages' in st.session_state:
        current_run_result['log_messages'] = st.session_state.log_messages
    
    return current_run_result

def _show_final_time_info(total_time, time_info_placeholder):
    """显示最终时间统计信息"""
    if len(st.session_state.epoch_durations) > 0:
        final_avg_epoch_time = sum(st.session_state.epoch_durations) / len(st.session_state.epoch_durations)
        final_time_info_str = (
            f"⏱️ **最终统计:**  "
            f"总耗时: {format_time_delta(total_time)} | "
            f"平均每轮: {final_avg_epoch_time:.2f} 秒 (共 {len(st.session_state.epoch_durations)} 轮)"
        )
        time_info_placeholder.markdown(final_time_info_str) 