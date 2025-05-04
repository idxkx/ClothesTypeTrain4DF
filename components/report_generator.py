import streamlit as st
import torch
import math
import os
import json
import traceback
import pandas as pd
from datetime import datetime
import re
from utils.file_utils import load_results
from utils.state_manager import append_log

def generate_diagnostic_report(history_df, best_val_loss, total_epochs):
    """根据训练历史生成诊断报告和结构化评估结果"""
    report = []
    report.append("### 🩺 训练诊断报告")
    
    # 创建一个结构化结果字典
    result_data = {
        "overfitting_risk": "未知",
        "loss_diff": float('nan'),
        "accuracy_diff": float('nan'),
        "convergence_status": "未知",
        "best_epoch": None,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "final_val_loss": None,
        "final_val_accuracy": None
    }

    if history_df is None or history_df.empty:
        report.append("- ❌ 无法生成报告：缺少训练历史数据。")
        return "\n".join(report), result_data

    final_epoch_data = history_df.iloc[-1]
    best_epoch_data = history_df.loc[history_df['Validation Loss'].idxmin()] if 'Validation Loss' in history_df.columns and history_df['Validation Loss'].notna().any() else None

    # 1. 整体表现
    report.append(f"- **训练轮数:** {len(history_df)} / {total_epochs}")
    if best_epoch_data is not None:
        best_epoch = int(best_epoch_data['epoch'])
        report.append(f"- **最佳验证损失:** {best_epoch_data['Validation Loss']:.4f} (出现在 Epoch {best_epoch})")
        result_data["best_epoch"] = best_epoch
    else:
        report.append("- **最佳验证损失:** 未记录或无效。")
    
    result_data["final_val_loss"] = final_epoch_data['Validation Loss']
    result_data["final_val_accuracy"] = final_epoch_data['Validation Accuracy (%)']
    
    report.append(f"- **最终验证损失:** {final_epoch_data['Validation Loss']:.4f}")
    report.append(f"- **最终验证准确率:** {final_epoch_data['Validation Accuracy (%)']:.2f}%")

    # 2. 收敛性分析
    convergence_status = "未知"
    if len(history_df) >= 5:
        last_5_val_loss = history_df['Validation Loss'].tail(5)
        if last_5_val_loss.is_monotonic_decreasing:
            convergence_status = "持续改善"
            report.append("- **收敛性:** ✅ 验证损失在最后5轮持续下降，可能仍有提升空间。")
        elif last_5_val_loss.iloc[-1] < last_5_val_loss.iloc[0]:
            convergence_status = "波动下降"
            report.append("- **收敛性:** ⚠️ 验证损失在最后5轮有所波动，但整体仍在下降。")
        else:
            convergence_status = "已收敛或停滞"
            report.append("- **收敛性:** ❌ 验证损失在最后5轮未能持续下降，可能已收敛或遇到瓶颈。")
    else:
        convergence_status = "轮数不足"
        report.append("- **收敛性:** ⚠️ 训练轮数较少，难以判断收敛趋势。")
    
    result_data["convergence_status"] = convergence_status

    # 3. 过拟合风险
    train_loss_final = final_epoch_data.get('Train Loss', float('nan'))
    val_loss_final = final_epoch_data.get('Validation Loss', float('nan'))
    train_acc_final = final_epoch_data.get('Train Accuracy (%)', float('nan'))
    val_acc_final = final_epoch_data.get('Validation Accuracy (%)', float('nan'))

    loss_diff = abs(train_loss_final - val_loss_final) if math.isfinite(train_loss_final) and math.isfinite(val_loss_final) else float('inf')
    acc_diff = abs(train_acc_final - val_acc_final) if math.isfinite(train_acc_final) and math.isfinite(val_acc_final) else float('inf')
    
    result_data["loss_diff"] = loss_diff
    result_data["accuracy_diff"] = acc_diff

    # 设定过拟合风险阈值
    overfitting_risk = "低"
    risk_details = []
    
    if loss_diff > 0.5:
        risk_details.append(f"损失差异({loss_diff:.2f})大于0.5")
        overfitting_risk = "高"
    elif loss_diff > 0.2:
        risk_details.append(f"损失差异({loss_diff:.2f})大于0.2")
        overfitting_risk = "中"
        
    if acc_diff > 15:
        risk_details.append(f"准确率差异({acc_diff:.1f}%)大于15%")
        overfitting_risk = "高"
    elif acc_diff > 8:
        risk_details.append(f"准确率差异({acc_diff:.1f}%)大于8%")
        if overfitting_risk != "高":
            overfitting_risk = "中"
    
    result_data["overfitting_risk"] = overfitting_risk
    
    # 格式化风险详情
    risk_detail_text = "，".join(risk_details) if risk_details else "训练集和验证集表现相近"
    
    report.append(f"- **过拟合风险:** {overfitting_risk} ")
    report.append(f"  - **损失差异:** {loss_diff:.2f} | **准确率差异:** {acc_diff:.1f}%") 
    report.append(f"  - **原因:** {risk_detail_text}")
    
    if overfitting_risk != "低":
        recommendation = ""
        if overfitting_risk == "高":
            recommendation = "建议: 增强正则化(增加dropout或权重衰减)、增加数据增强或缩小模型规模"
        else:  # 中风险
            recommendation = "建议: 考虑适当增加正则化或提前停止训练"
        report.append(f"  - _{recommendation}_")

    return "\n".join(report), result_data

def run_functional_test(model_save_dir, model_name, model_config, device):
    """尝试加载最佳模型并进行一次模拟推理，同时检查元数据文件"""
    report = ["### ⚙️ 功能模拟测试"]
    best_model_file = None
    metadata_file = None
    error_details = {}
    
    try:
        # 检查参数有效性
        if not model_save_dir:
            report.append("- ❌ 模型保存目录为空")
            error_details["error_type"] = "empty_save_dir"
            error_details["message"] = "模型保存目录为空，无法进行功能测试"
            error_details["solution"] = "请确保已指定有效的模型保存目录"
            return "\n".join(report), False, error_details
            
        if not os.path.exists(model_save_dir):
            report.append(f"- ❌ 模型保存目录不存在: {model_save_dir}")
            error_details["error_type"] = "dir_not_found"
            error_details["message"] = f"指定的模型保存目录不存在: {model_save_dir}"
            error_details["solution"] = "请检查模型保存路径是否正确，可能需要重新训练模型"
            return "\n".join(report), False, error_details
            
        if not model_name:
            report.append("- ❌ 模型名称为空")
            error_details["error_type"] = "empty_model_name"
            error_details["message"] = "模型名称为空，无法定位模型文件"
            error_details["solution"] = "请确保模型名称已正确设置" 
            return "\n".join(report), False, error_details
            
        if not model_config:
            report.append("- ❌ 模型配置为空")
            error_details["error_type"] = "empty_config"
            error_details["message"] = "模型配置为空，无法初始化模型"
            error_details["solution"] = "请提供有效的模型配置" 
            return "\n".join(report), False, error_details
        
        # 提取模型名称的基础部分，忽略可能的时间戳
        # 例如：从 "MD_RESNET18_5_64_50E04_0503_1137" 提取 "MD_RESNET18_5_64_50E04"
        base_model_name = model_name
        # 如果有时间戳格式 (_MMDD_HHMM)，去除它
        time_stamp_pattern = r'_\d{4}_\d{4}$'
        base_model_name = re.sub(time_stamp_pattern, '', base_model_name)
        
        # 查找模型文件的方式更灵活
        try:
            all_files = os.listdir(model_save_dir)
        except Exception as e:
            report.append(f"- ❌ 无法读取模型目录: {e}")
            error_details["error_type"] = "directory_read_error"
            error_details["message"] = f"读取模型目录时出错: {e}"
            error_details["solution"] = "请检查模型目录权限或是否可访问"
            return "\n".join(report), False, error_details
            
        # 1. 首先尝试完全匹配
        possible_files = [f for f in all_files if f.startswith(f"best_model_{model_name}") and f.endswith(".pth")]
        
        # 2. 如果找不到，尝试使用基础名称部分匹配
        if not possible_files:
            possible_files = [f for f in all_files if f.startswith(f"best_model_{base_model_name}") and f.endswith(".pth")]
            
        # 3. 如果还找不到，尝试更宽松的匹配（适用于名称被截断的情况）
        if not possible_files and len(base_model_name) > 15:
            short_name = base_model_name[:15]  # 取前15个字符
            possible_files = [f for f in all_files if f.startswith(f"best_model_{short_name}") and f.endswith(".pth")]
        
        # 如果所有方法都找不到文件
        if not possible_files:
            report.append("- ❌ 未找到保存的最佳模型文件。")
            error_details["error_type"] = "missing_model_file"
            error_details["message"] = f"无法在 {model_save_dir} 中找到与 '{model_name}' 相关的模型文件"
            error_details["solution"] = "请检查训练是否成功完成，或者模型文件是否被意外删除。可以尝试重新训练模型。"
            return "\n".join(report), False, error_details
            
        # 如果有多个匹配文件，选择最新的一个（按epoch排序）
        try:
            # 首先尝试提取epoch编号进行排序
            possible_files.sort(key=lambda x: int(x.split('_epoch')[-1].split('.')[0]), reverse=True)
        except (IndexError, ValueError):
            # 如果无法提取epoch编号，则按文件修改时间排序
            possible_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_save_dir, x)), reverse=True)
        
        best_model_file = os.path.join(model_save_dir, possible_files[0])
        report.append(f"- 找到最佳模型文件: `{best_model_file}`")
        
        # 同样使用更灵活的方式查找元数据文件
        possible_metadata_files = [
            os.path.join(model_save_dir, f"{model_name}_metadata.json"),
            os.path.join(model_save_dir, f"{base_model_name}_metadata.json")
        ]
        
        metadata_file = None
        for meta_path in possible_metadata_files:
            if meta_path and os.path.exists(meta_path):  # 添加对meta_path非None的检查
                metadata_file = meta_path
                break

        # 读取并显示基本元数据信息
        if metadata_file and os.path.exists(metadata_file):  # 添加对metadata_file非None的检查
            report.append(f"- ✅ 找到元数据文件: `{metadata_file}`")
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                report.append("- 元数据概要:")
                report.append(f"  - 模型名称: {metadata.get('model_name', '未定义')}")
                report.append(f"  - 版本: {metadata.get('version', '未定义')}")
                report.append(f"  - 架构: {metadata.get('architecture', '未定义')}")
                report.append(f"  - 输入形状: {metadata.get('input_shape', '未定义')}")
                report.append(f"  - 支持类别数: {len(metadata.get('class_names', []))} 类")
                report.append(f"  - 支持特征数: {len(metadata.get('feature_names', []))} 项")
                report.append(f"  - 创建日期: {metadata.get('date_created', '未定义')}")
            except Exception as e:
                report.append(f"- ⚠️ 读取元数据文件出错: {e}")
                report.append(f"  - 可能原因: 元数据文件格式不正确或损坏")
                report.append(f"  - 建议解决方案: 重新生成元数据文件")
                error_details["error_type"] = "metadata_read_error"
                error_details["message"] = f"读取元数据文件时出错: {e}"
                error_details["solution"] = "元数据文件可能已损坏，请尝试使用'批量生成缺失的元数据'功能重新生成。"
        else:
            report.append("- ⚠️ 未找到元数据文件，这可能会影响模型的可用性。")
            report.append("  - 解决方案: 点击'批量生成缺失的元数据'按钮或使用模型管理界面生成元数据")
            error_details["missing_metadata"] = True

        # 加载模型
        report.append("- 尝试加载模型...")
        try:
            from model import ClothesModel
            model = ClothesModel(num_categories=model_config['num_categories'], backbone=model_config['backbone'])
        except Exception as model_init_error:
            report.append(f"- ❌ 模型初始化失败: {model_init_error}")
            report.append("  - 可能原因: 模型类定义已更改或参数不匹配")
            report.append("  - 建议解决方案: 请检查模型配置，确保backbone参数正确")
            error_details["error_type"] = "model_init_error"
            error_details["message"] = f"模型初始化失败: {model_init_error}"
            error_details["solution"] = "请检查模型配置，尤其是backbone参数是否正确。可能需要重新训练模型。"
            return "\n".join(report), False, error_details
            
        try:
            model.load_state_dict(torch.load(best_model_file, map_location=device))
        except Exception as load_error:
            report.append(f"- ❌ 模型权重加载失败: {load_error}")
            report.append("  - 可能原因: 模型文件损坏或模型结构与保存时不一致")
            report.append("  - 建议解决方案: 检查模型文件是否完整，可能需要重新训练模型")
            error_details["error_type"] = "model_load_error"
            error_details["message"] = f"模型权重加载失败: {load_error}"
            error_details["solution"] = "模型文件可能已损坏或与当前代码不兼容。请检查模型文件完整性，必要时重新训练模型。"
            return "\n".join(report), False, error_details
            
        model.to(device)
        model.eval()
        report.append("- ✅ 模型加载成功。")

        # 模拟推理
        report.append("- 尝试模拟推理...")
        # 创建虚拟输入 (batch_size=1, channels=3, height=224, width=224)
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                cat_logits, attr_logits = model(dummy_input)
        except RuntimeError as rt_error:
            # CUDA内存不足
            if "CUDA out of memory" in str(rt_error):
                report.append(f"- ❌ 推理失败: GPU内存不足")
                report.append("  - 原因: 当前GPU内存不足以运行此模型")
                report.append("  - 建议解决方案: 关闭其他占用GPU内存的应用，或使用更小的模型")
                error_details["error_type"] = "cuda_oom_error"
                error_details["message"] = f"GPU内存不足: {rt_error}"
                error_details["solution"] = "请尝试关闭其他占用GPU内存的应用程序，或者使用内存占用更小的backbone。"
            else:
                report.append(f"- ❌ 推理过程中出现运行时错误: {rt_error}")
                report.append("  - 可能原因: 模型结构与训练时不一致或输入格式问题")
                report.append("  - 建议解决方案: 检查模型配置，确保backbone参数正确")
                error_details["error_type"] = "inference_runtime_error"
                error_details["message"] = f"推理过程中出现运行时错误: {rt_error}"
                error_details["solution"] = "请检查模型配置与训练时是否一致，尤其是backbone参数。可能需要重新训练模型。"
            return "\n".join(report), False, error_details
        except Exception as infer_error:
            report.append(f"- ❌ 推理失败: {infer_error}")
            report.append("  - 可能原因: 模型结构问题或内部错误")
            report.append("  - 建议解决方案: 检查模型代码，可能需要重新训练模型")
            error_details["error_type"] = "inference_error"
            error_details["message"] = f"推理失败: {infer_error}"
            error_details["solution"] = "模型内部出现错误，请检查模型代码或尝试重新训练。"
            return "\n".join(report), False, error_details
        
        # 检查输出
        report.append(f"- 模型输出 (类别 Logits): {cat_logits.shape}")
        report.append(f"- 模型输出 (属性 Logits): {attr_logits.shape}")
        
        # 验证输出形状
        expected_categories = model_config['num_categories']
        expected_attributes = 26  # 固定26个属性
        
        shape_errors = []
        if cat_logits.shape[0] != 1:
            shape_errors.append(f"类别输出批次维度错误: 期望1，实际{cat_logits.shape[0]}")
        if cat_logits.shape[1] != expected_categories:
            shape_errors.append(f"类别数量错误: 期望{expected_categories}，实际{cat_logits.shape[1]}")
        if attr_logits.shape[0] != 1:
            shape_errors.append(f"属性输出批次维度错误: 期望1，实际{attr_logits.shape[0]}")
        if attr_logits.shape[1] != expected_attributes:
            shape_errors.append(f"属性数量错误: 期望{expected_attributes}，实际{attr_logits.shape[1]}")
            
        if shape_errors:
            report.append("- ❌ 模拟推理完成，但输出形状不符合预期！")
            for err in shape_errors:
                report.append(f"  - {err}")
            report.append("  - 可能原因: 模型训练时的类别或属性数量配置与当前不一致")
            report.append("  - 建议解决方案: 检查训练配置，确保类别数量和属性数量正确设置")
            error_details["error_type"] = "output_shape_error"
            error_details["message"] = "模型输出形状不符合预期"
            error_details["details"] = shape_errors
            error_details["solution"] = "模型输出尺寸与预期不符，可能是训练配置与当前不匹配。请检查模型配置，尤其是类别数量设置。"
            return "\n".join(report), False, error_details
        else:
            report.append("- ✅ 模拟推理成功，输出形状符合预期。")
             
        # 检查是否有元数据和模型，全部成功才返回成功
        if os.path.exists(metadata_file):
            report.append("- 🎉 功能测试全部通过！模型可以正常加载和推理，元数据文件齐全。")
            return "\n".join(report), True, {}
        else:
            report.append("- ⚠️ 功能测试部分成功，但缺少元数据文件。")
            report.append("  - 建议: 点击'批量生成缺失的元数据'按钮生成缺失的元数据文件")
            error_details["error_type"] = "missing_metadata"
            error_details["message"] = "模型可以正常加载和使用，但缺少元数据文件"
            error_details["solution"] = "请使用'批量生成缺失的元数据'功能生成元数据文件。"
            return "\n".join(report), False, error_details

    except Exception as e:
        error_trace = traceback.format_exc()
        report.append(f"- ❌ 功能测试失败: {e}")
        report.append(f"- 错误类型: {type(e).__name__}")
        report.append("- 可能原因:")
        
        if "No such file or directory" in str(e):
            report.append("  - 文件或目录不存在，模型文件或目录可能已被删除")
            report.append("  - 建议解决方案: 检查模型文件和目录是否存在，可能需要重新训练模型")
            error_details["error_type"] = "file_not_found"
        elif "CUDA out of memory" in str(e):
            report.append("  - GPU内存不足，无法加载或运行模型")
            report.append("  - 建议解决方案: 关闭其他占用GPU内存的应用，或使用CPU模式")
            error_details["error_type"] = "cuda_oom"
        elif "ModuleNotFoundError" in error_trace:
            report.append("  - 缺少必要的Python模块")
            report.append("  - 建议解决方案: 安装缺失的依赖包")
            error_details["error_type"] = "missing_module"
        elif "KeyError" in error_trace:
            report.append("  - 模型参数键错误，模型结构可能与保存时不一致")
            report.append("  - 建议解决方案: 确保使用相同版本的代码加载模型，或重新训练模型")
            error_details["error_type"] = "model_key_error"
        else:
            report.append("  - 未知错误，请查看详细错误信息")
            report.append("  - 建议解决方案: 检查环境配置，模型文件，可能需要重新训练模型")
            error_details["error_type"] = "unknown_error"
            
        error_details["message"] = str(e)
        error_details["solution"] = "请根据错误详情采取对应解决方案，必要时可尝试重新训练模型。"
        append_log(f"功能测试失败: {error_trace}")
        return "\n".join(report), False, error_details

def generate_metadata_for_model(model_result):
    """为单个模型生成元数据文件"""
    try:
        model_name = model_result.get("model_name", "")
        model_path = model_result.get("best_model_path", "")
        
        if not model_name:
            return False, "模型名称为空"
            
        if not model_path:
            return False, "模型路径为空"
            
        if not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}"
        
        # 提取模型目录
        model_dir = os.path.dirname(model_path)
        if not model_dir:
            return False, "无法提取模型目录"
            
        # 元数据文件路径
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        # 检查是否已存在
        if metadata_file and os.path.exists(metadata_file):
            return True, "元数据已存在"
        
        # 从训练结果中提取基本信息
        backbone = model_result.get("backbone", "unknown")
        num_categories = model_result.get("num_categories", 13)
        input_size = model_result.get("image_size", 224)
        learning_rate = model_result.get("learning_rate", 0.0001)
        batch_size = model_result.get("batch_size", 32)
        total_epochs = model_result.get("total_epochs", 50)
        
        # 完成的训练轮数
        completed_epochs = model_result.get("completed_epochs", 0)
        
        # 查找性能指标
        best_val_loss = model_result.get("best_val_loss", 0.0)
        best_val_acc = model_result.get("best_val_accuracy", 0.0)
        best_epoch = model_result.get("best_epoch", 0)
        
        # 默认的类别和特征
        # 这些可以从其他地方导入，但为了简单起见，我们在这里硬编码
        from utils.state_manager import get_default_categories, get_default_features
        class_names = get_default_categories()
        feature_names = get_default_features()
        
        # 创建元数据字典
        metadata = {
            "model_name": model_name,
            "version": "1.0",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": f"ClothesModel({backbone})",
            "input_shape": [3, input_size, input_size],
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize": input_size
            },
            "class_names": class_names,
            "feature_names": feature_names,
            "training_info": {
                "backbone": backbone,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": completed_epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc
            },
            "export_format": "PyTorch",
            "usage_examples": {
                "python": "# 请参考示例代码使用此模型"
            }
        }
        
        # 保存元数据文件
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True, "元数据创建成功"
        except Exception as e:
            return False, f"保存元数据文件时出错: {e}"
            
    except Exception as e:
        traceback_info = traceback.format_exc()
        append_log(f"生成元数据时出错: {traceback_info}")
        return False, f"生成元数据时出错: {e}"

def batch_generate_metadata():
    """批量生成缺失的元数据"""
    results = []
    
    # 加载所有训练记录
    all_results = load_results()
    
    for model_data in all_results:
        model_name = model_data.get("model_name", "")
        model_path = model_data.get("best_model_path", "")
        
        if not model_name or not model_path:
            results.append((model_name or "未命名模型", False, "模型信息不完整"))
            continue
            
        # 检查模型文件是否存在
        if model_path and not os.path.exists(model_path):
            results.append((model_name, False, f"模型文件不存在: {model_path}"))
            continue
        
        # 检查元数据文件是否已存在
        model_dir = os.path.dirname(model_path) if model_path else ""
        if not model_dir:
            results.append((model_name, False, "模型路径不完整"))
            continue
            
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        # 如果元数据已存在，跳过
        if metadata_file and os.path.exists(metadata_file):
            results.append((model_name, True, "元数据已存在"))
            continue
        
        # 生成元数据
        try:
            success, message = generate_metadata_for_model(model_data)
            results.append((model_name, success, message))
        except Exception as e:
            results.append((model_name, False, f"生成元数据时出错: {str(e)}"))
    
    return results

def create_metadata_file(model_name, model_data, metadata_input):
    """根据用户输入创建元数据文件"""
    try:
        if not model_name:
            return False, "模型名称为空"
            
        model_path = model_data.get("best_model_path", "")
        if not model_path:
            return False, "模型路径为空"
            
        if not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}"
        
        # 提取模型目录
        model_dir = os.path.dirname(model_path)
        if not model_dir:
            return False, "无法提取模型目录"
            
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        # 检查是否已存在
        if metadata_file and os.path.exists(metadata_file):
            # 进行覆盖确认
            st.warning(f"元数据文件 '{metadata_file}' 已存在，将被覆盖。")
        
        # 准备元数据结构
        metadata = {
            "model_name": model_name,
            "version": metadata_input.get("version", "1.0"),
            "description": metadata_input.get("description", "服装分类与属性识别模型"),
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": f"ClothesModel({model_data.get('backbone', 'unknown')})",
            "input_shape": [3, model_data.get("image_size", 224), model_data.get("image_size", 224)],
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize": model_data.get("image_size", 224)
            },
            # 使用提供的类别和特征名称
            "class_names": metadata_input.get("class_names", []),
            "feature_names": metadata_input.get("feature_names", []),
            "training_info": {
                "backbone": model_data.get("backbone", "unknown"),
                "learning_rate": model_data.get("learning_rate", 0.0001),
                "batch_size": model_data.get("batch_size", 32),
                "epochs": model_data.get("completed_epochs", 0),
                "best_epoch": model_data.get("best_epoch", 0),
                "best_val_loss": model_data.get("best_val_loss", 0.0),
                "best_val_accuracy": model_data.get("best_val_accuracy", 0.0),
                "status": model_data.get("status", "unknown")
            },
            "author": metadata_input.get("author", ""),
            "contact": metadata_input.get("contact", ""),
            "license": metadata_input.get("license", ""),
            "export_format": "PyTorch",
            "usage_notes": metadata_input.get("usage_notes", "")
        }
        
        # 保存元数据
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return True, "元数据创建成功"
    except Exception as e:
        traceback_info = traceback.format_exc()
        append_log(f"创建元数据文件时出错: {traceback_info}")
        return False, f"创建元数据文件时出错: {e}" 