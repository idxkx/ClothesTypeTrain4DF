import streamlit as st
import torch
import math
import os
import json
import traceback
import pandas as pd
from datetime import datetime
import re

from utils.state_manager import append_log

def generate_diagnostic_report(history_df, best_val_loss, total_epochs):
    """根据训练历史生成诊断报告"""
    report = []
    report.append("### 🩺 训练诊断报告")

    if history_df is None or history_df.empty:
        report.append("- ❌ 无法生成报告：缺少训练历史数据。")
        return "\n".join(report)

    final_epoch_data = history_df.iloc[-1]
    best_epoch_data = history_df.loc[history_df['Validation Loss'].idxmin()] if 'Validation Loss' in history_df.columns and history_df['Validation Loss'].notna().any() else None

    # 1. 整体表现
    report.append(f"- **训练轮数:** {len(history_df)} / {total_epochs}")
    if best_epoch_data is not None:
        report.append(f"- **最佳验证损失:** {best_epoch_data['Validation Loss']:.4f} (出现在 Epoch {int(best_epoch_data['epoch'])})")
    else:
        report.append("- **最佳验证损失:** 未记录或无效。")
    report.append(f"- **最终验证损失:** {final_epoch_data['Validation Loss']:.4f}")
    report.append(f"- **最终验证准确率:** {final_epoch_data['Validation Accuracy (%)']:.2f}%")

    # 2. 收敛性分析
    if len(history_df) >= 5:
        last_5_val_loss = history_df['Validation Loss'].tail(5)
        if last_5_val_loss.is_monotonic_decreasing:
            report.append("- **收敛性:** ✅ 验证损失在最后5轮持续下降，可能仍有提升空间。")
        elif last_5_val_loss.iloc[-1] < last_5_val_loss.iloc[0]:
            report.append("- **收敛性:** ⚠️ 验证损失在最后5轮有所波动，但整体仍在下降。")
        else:
             report.append("- **收敛性:** ❌ 验证损失在最后5轮未能持续下降，可能已收敛或遇到瓶颈。")
    else:
        report.append("- **收敛性:** ⚠️ 训练轮数较少，难以判断收敛趋势。")

    # 3. 过拟合风险
    train_loss_final = final_epoch_data.get('Train Loss', float('nan'))
    val_loss_final = final_epoch_data.get('Validation Loss', float('nan'))
    train_acc_final = final_epoch_data.get('Train Accuracy (%)', float('nan'))
    val_acc_final = final_epoch_data.get('Validation Accuracy (%)', float('nan'))

    loss_diff = abs(train_loss_final - val_loss_final) if math.isfinite(train_loss_final) and math.isfinite(val_loss_final) else float('inf')
    acc_diff = abs(train_acc_final - val_acc_final) if math.isfinite(train_acc_final) and math.isfinite(val_acc_final) else float('inf')

    # 设定一些简单的阈值
    overfitting_risk = "低"
    if loss_diff > 0.5 or acc_diff > 15:
        overfitting_risk = "高"
    elif loss_diff > 0.2 or acc_diff > 8:
        overfitting_risk = "中"

    report.append(f"- **过拟合风险:** {overfitting_risk} (基于最终损失差异 {loss_diff:.2f} 和准确率差异 {acc_diff:.1f}%) ")
    if overfitting_risk != "低":
        report.append("  - _建议: 可尝试增加正则化、数据增强或提前停止。_")

    return "\n".join(report)

def run_functional_test(model_save_dir, model_name, model_config, device):
    """尝试加载最佳模型并进行一次模拟推理，同时检查元数据文件"""
    report = ["### ⚙️ 功能模拟测试"]
    best_model_file = None
    metadata_file = None
    error_details = {}
    
    try:
        # 提取模型名称的基础部分，忽略可能的时间戳
        # 例如：从 "MD_RESNET18_5_64_50E04_0503_1137" 提取 "MD_RESNET18_5_64_50E04"
        base_model_name = model_name
        # 如果有时间戳格式 (_MMDD_HHMM)，去除它
        time_stamp_pattern = r'_\d{4}_\d{4}$'
        base_model_name = re.sub(time_stamp_pattern, '', base_model_name)
        
        # 查找模型文件的方式更灵活
        all_files = os.listdir(model_save_dir)
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
            if os.path.exists(meta_path):
                metadata_file = meta_path
                break

        # 读取并显示基本元数据信息
        if metadata_file:
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
    """为单个模型生成元数据"""
    try:
        model_name = model_result.get("model_name", "")
        model_path = model_result.get("best_model_path", "")
        if not model_path or not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}"

        # 构建元数据
        metadata = {
            "model_name": model_name,
            "version": "1.0.0",
            "description": f"基于{model_result.get('backbone', 'unknown')}的服装分类模型",
            "architecture": model_result.get('backbone', 'unknown'),
            "input_shape": [3, 224, 224],  # 标准输入尺寸
            "framework": "PyTorch",
            "date_created": model_result.get("start_time_str", "").split()[0],  # 只取日期部分
            "trained_by": "喵搭服装识别训练平台",
            "training_params": {
                "epochs": model_result.get("total_epochs"),
                "completed_epochs": model_result.get("completed_epochs"),
                "best_val_loss": model_result.get("best_val_loss"),
                "best_epoch": model_result.get("best_epoch"),
                "strategy": model_result.get("strategy"),
                "status": model_result.get("status"),
            },
            # 添加默认的类别和特征名称
            "class_names": [
                "T恤", "衬衫", "卫衣", "毛衣", "西装", "夹克", "羽绒服", "风衣",
                "牛仔裤", "休闲裤", "西裤", "短裤", "运动裤", "连衣裙", "半身裙",
                "旗袍", "礼服", "运动鞋", "皮鞋", "高跟鞋", "靴子", "凉鞋", "拖鞋",
                "帽子", "围巾", "领带", "手套", "袜子", "腰带", "眼镜", "手表",
                "项链", "手链", "耳环", "戒指", "包包", "背包", "手提包", "钱包", "行李箱"
            ],
            "feature_names": [
                "颜色", "材质", "样式", "花纹", "季节", "正式度", "领型", "袖长",
                "长度", "裤型", "鞋型", "高度", "闭合方式"
            ]
        }

        # 保存元数据文件
        model_dir = os.path.dirname(model_path)
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        
        return True, metadata_file
    except Exception as e:
        return False, f"生成元数据时出错: {e}"

def batch_generate_metadata():
    """批量为所有缺失元数据的模型生成元数据文件"""
    from utils.file_utils import load_results
    
    results = []
    all_models = load_results()
    
    for model_result in all_models:
        model_name = model_result.get("model_name", "未命名模型")
        model_path = model_result.get("best_model_path", "")
        
        if not model_path or not os.path.exists(model_path):
            results.append((model_name, False, "模型文件不存在"))
            continue
            
        # 检查是否已有元数据
        model_dir = os.path.dirname(model_path)
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        # 如果有时间戳命名的模型，也尝试检查不带时间戳的元数据文件
        if not os.path.exists(metadata_file):
            # 尝试移除时间戳部分并检查
            base_model_name = re.sub(r'_\d{4}_\d{4}$', '', model_name)
            alt_metadata_file = os.path.join(model_dir, f"{base_model_name}_metadata.json")
            if os.path.exists(alt_metadata_file):
                results.append((model_name, True, f"找到使用基础名称的元数据: {alt_metadata_file}"))
                continue
        else:
            results.append((model_name, True, "元数据已存在"))
            continue
            
        # 生成元数据
        success, message = generate_metadata_for_model(model_result)
        results.append((model_name, success, message))
    
    return results

def create_metadata_file(model_name, model_data, metadata_input):
    """根据用户输入创建元数据文件"""
    try:
        model_path = model_data.get("best_model_path", "")
        if not model_path or not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}"
            
        # 解析输入参数
        version = metadata_input.get("version", "1.0.0")
        description = metadata_input.get("description", f"基于{model_data.get('backbone', 'unknown')}的服装分类模型")
        trained_by = metadata_input.get("trained_by", "喵搭服装识别训练平台")
        date_created = metadata_input.get("date_created", datetime.now().strftime("%Y-%m-%d"))
        
        # 解析输入形状
        input_shape_str = metadata_input.get("input_shape", "3,224,224")
        input_shape = [int(x.strip()) for x in input_shape_str.split(",")]
        
        # 解析类别和特征名称
        class_names_text = metadata_input.get("class_names", "")
        class_names = [name.strip() for name in class_names_text.split("\n") if name.strip()]
        
        feature_names_text = metadata_input.get("feature_names", "")
        feature_names = [name.strip() for name in feature_names_text.split("\n") if name.strip()]
        
        # 构建元数据
        metadata = {
            "model_name": model_name,
            "version": version,
            "description": description,
            "architecture": model_data.get('backbone', 'unknown'),
            "input_shape": input_shape,
            "framework": "PyTorch",
            "date_created": date_created,
            "trained_by": trained_by,
            "training_params": {
                "epochs": model_data.get("total_epochs"),
                "completed_epochs": model_data.get("completed_epochs"),
                "best_val_loss": model_data.get("best_val_loss"),
                "best_epoch": model_data.get("best_epoch"),
                "strategy": model_data.get("strategy"),
                "status": model_data.get("status"),
            },
            "class_names": class_names,
            "feature_names": feature_names
        }
        
        # 保存元数据文件
        model_dir = os.path.dirname(model_path)
        metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        return True, f"已成功创建元数据文件: {metadata_file}"
    except Exception as e:
        return False, f"创建元数据时出错: {e}" 