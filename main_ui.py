# Placeholder for Streamlit UI code 

import streamlit as st
import torch
import os
import time
import pandas as pd
import math
import traceback
from torchvision import transforms # 确保导入 transforms
from datetime import datetime, timedelta 
# --- 新增：导入 json --- 
import json 

# --- GPU 监控依赖 ---
nvml_available = False
pynvml = None
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_available = True
except ImportError:
    # st.sidebar.info("提示：未找到 pynvml 库，无法进行 GPU 监控。可尝试 `pip install nvidia-ml-py` 安装。")
    pass # 不在导入时显示，避免干扰
except pynvml.NVMLError as e:
    # st.sidebar.warning(f"NVML 初始化失败: {e}. 无法进行 GPU 监控。")
    pass # 不在导入时显示
# --------------------->

# 尝试导入我们自己写的模块
# 添加路径到 sys.path 可能更健壮，但这里先用 try-except
try:
    from dataset import DeepFashionDataset, ANNOTATION_DIR_NAME, IMAGE_DIR_NAME
    from model import ClothesModel
    from trainer import Trainer
except ImportError as e:
    st.error(f"错误：无法导入必要的模块或常量。请确保 dataset.py, model.py, trainer.py 与 main_ui.py 在同一目录下。错误: {e}")
    # 如果导入失败，停止执行，避免后续错误
    st.stop()

# --- 页面配置 ---
st.set_page_config(page_title="服装识别训练场", layout="wide")
st.title("👕👗 服装类别与属性识别 - AI 训练场")
st.markdown("--- ")

# --- Session State 初始化 ---
# 使用 st.session_state 来存储跨运行的状态
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'history_df_list' not in st.session_state:
    st.session_state.history_df_list = []
if 'selected_strategy' not in st.session_state:
    # 默认选择均衡策略，或者设为 None 表示手动
    st.session_state.selected_strategy = "均衡推荐 (Balanced)" 
# --- 新增：GPU 指标历史记录 --- 
if 'gpu_metrics_history' not in st.session_state:
    st.session_state.gpu_metrics_history = []
if 'gpu_poll_step' not in st.session_state:
    st.session_state.gpu_poll_step = 0
# --- 新增：时间相关状态 --- 
if 'training_start_time' not in st.session_state:
    st.session_state.training_start_time = None
if 'epoch_durations' not in st.session_state:
    st.session_state.epoch_durations = []
# ------------------------->

# --- 新增：结果文件常量 --- 
RESULTS_FILE = "training_results.json"
# ------------------------>

# --- 函数定义提前 ---

# 日志追加函数
def append_log(message):
    """将带时间戳的消息追加到 session_state 的日志列表中"""
    st.session_state.log_messages.append(f"[{time.strftime('%H:%M:%S')}] {message}")

# GPU 信息更新函数
def update_gpu_info(gpu_index, placeholder, chart_placeholders):
    """更新指定的 placeholder 区域的 GPU 监控信息和图表"""
    gpu_util_chart_placeholder, gpu_mem_chart_placeholder = chart_placeholders
    if nvml_available and pynvml and gpu_index is not None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # 瓦特
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0 # 瓦特

            # 使用全局变量 gpu_names 获取名称
            gpu_name = gpu_names.get(gpu_index, f"GPU {gpu_index}") 
            gpu_info_str = (
                f"**{gpu_name}:**\n"
                f"- **温度:** {temp}°C\n"
                f"- **使用率:** {util.gpu}%\n"
                f"- **显存:** {mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB ({mem.used * 100 / mem.total:.1f}%)\n"
                f"- **功耗:** {power_usage:.1f} W / {power_limit:.1f} W"
            )
            placeholder.markdown(gpu_info_str) # 更新传入的 placeholder

            # --- 新增：记录和绘制 GPU 指标历史 --- 
            current_step = st.session_state.gpu_poll_step
            memory_util_percent = mem.used * 100 / mem.total if mem.total > 0 else 0
            st.session_state.gpu_metrics_history.append({
                'step': current_step,
                'GPU Utilization (%)': util.gpu,
                'Memory Utilization (%)': memory_util_percent
            })
            st.session_state.gpu_poll_step += 1
            
            # 创建 DataFrame 并绘图 (只保留最近 N 个点避免过长？暂时全画)
            gpu_history_df = pd.DataFrame(st.session_state.gpu_metrics_history)
            if len(gpu_history_df) > 1: # 需要至少两个点才能画线
                try:
                    gpu_util_chart_placeholder.line_chart(gpu_history_df.set_index('step')['GPU Utilization (%)'])
                    gpu_mem_chart_placeholder.line_chart(gpu_history_df.set_index('step')['Memory Utilization (%)'])
                except Exception as chart_e:
                    # 防止绘图失败导致整个监控中断
                    if 'chart_error_logged' not in st.session_state or not st.session_state.chart_error_logged:
                         st.sidebar.warning(f"绘制 GPU 图表时出错: {chart_e}")
                         st.session_state.chart_error_logged = True
            # --- 新增结束 ---

        except pynvml.NVMLError as e:
            if 'gpu_error_logged' not in st.session_state or not st.session_state.gpu_error_logged:
                placeholder.warning(f"获取 GPU {gpu_index} 信息失败: {e}")
                st.session_state.gpu_error_logged = True
        except Exception as e:
             if 'gpu_error_logged' not in st.session_state or not st.session_state.gpu_error_logged:
                placeholder.error(f"更新 GPU 信息时发生未知错误: {e}")
                st.session_state.gpu_error_logged = True

# --- 新增：时间格式化函数 ---
def format_time_delta(seconds):
    """将秒数格式化为 HH:MM:SS"""
    if seconds is None or not math.isfinite(seconds):
        return "N/A"
    delta = timedelta(seconds=int(seconds))
    return str(delta)

# --- 新增：诊断报告生成函数 ---
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

    # 设定一些简单的阈值 (可以根据经验调整)
    overfitting_risk = "低"
    if loss_diff > 0.5 or acc_diff > 15:
        overfitting_risk = "高"
    elif loss_diff > 0.2 or acc_diff > 8:
        overfitting_risk = "中"

    report.append(f"- **过拟合风险:** {overfitting_risk} (基于最终损失差异 {loss_diff:.2f} 和准确率差异 {acc_diff:.1f}%) ")
    if overfitting_risk != "低":
        report.append("  - _建议: 可尝试增加正则化、数据增强或提前停止。_")

    return "\n".join(report)

# --- 新增：功能模拟测试函数 ---
def run_functional_test(model_save_dir, model_name, model_config, device):
    """尝试加载最佳模型并进行一次模拟推理"""
    report = ["### ⚙️ 功能模拟测试"] 
    best_model_file = None
    try:
        # 查找最佳模型文件
        possible_files = [f for f in os.listdir(model_save_dir) if f.startswith(f"best_model_{model_name}") and f.endswith(".pth")]
        if not possible_files:
            report.append("- ❌ 未找到保存的最佳模型文件。")
            return "\n".join(report), False
        # 如果有多个，理论上不该发生，但可以取最新的（按文件名中的epoch）
        possible_files.sort(key=lambda x: int(x.split('_epoch')[-1].split('.')[0]), reverse=True)
        best_model_file = os.path.join(model_save_dir, possible_files[0])
        report.append(f"- 找到最佳模型文件: `{best_model_file}`")

        # 加载模型
        report.append("- 尝试加载模型...")
        # 需要确保 ClothesModel 在当前作用域可见
        from model import ClothesModel # 或者确保它已在顶部导入
        model = ClothesModel(num_categories=model_config['num_categories'], backbone=model_config['backbone'])
        model.load_state_dict(torch.load(best_model_file, map_location=device))
        model.to(device)
        model.eval()
        report.append("- ✅ 模型加载成功。")

        # 模拟推理
        report.append("- 尝试模拟推理...")
        # 创建虚拟输入 (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            cat_logits, attr_logits = model(dummy_input)
        
        # 检查输出
        report.append(f"- 模型输出 (类别 Logits): {cat_logits.shape}")
        report.append(f"- 模型输出 (属性 Logits): {attr_logits.shape}")
        if cat_logits.shape[0] == 1 and cat_logits.shape[1] == model_config['num_categories'] and \
           attr_logits.shape[0] == 1 and attr_logits.shape[1] == 26: # 检查属性维度是26
             report.append("- ✅ 模拟推理成功，输出形状符合预期。")
             return "\n".join(report), True
        else:
             report.append("- ❌ 模拟推理完成，但输出形状不符合预期！")
             return "\n".join(report), False

    except Exception as e:
        report.append(f"- ❌ 功能测试失败: {e}")
        append_log(f"功能测试失败: {traceback.format_exc()}") # 记录详细错误到日志
        return "\n".join(report), False

# --- 新增：结果加载/保存函数 ---
def load_results():
    """加载历史训练结果"""
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # 确保返回的是列表
        return results if isinstance(results, list) else []
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        # 文件损坏或为空，返回空列表
        return []

def save_results(results):
    """保存训练结果列表到 JSON 文件"""
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    except IOError as e:
        st.error(f"无法保存训练结果到 {RESULTS_FILE}: {e}")
# ------------------------------>

# --- 策略定义 ---
STRATEGIES = {
    "快速体验 (Fast Trial)": {
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 5e-4,
        "backbone": 'resnet18',
        "attribute_loss_weight": 1.0
    },
    "均衡推荐 (Balanced)": {
        "epochs": 15,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "backbone": 'efficientnet_b3',
        "attribute_loss_weight": 1.0
    },
    "最高精度 (High Accuracy)": {
        "epochs": 30,
        "batch_size": 32, # 保持 32，避免显存问题
        "learning_rate": 5e-5,
        "backbone": 'efficientnet_b4',
        "attribute_loss_weight": 1.0
    },
    # 可以添加一个"手动设置"选项，或者通过选择其他策略后修改来进入手动状态
    "手动设置 (Manual)": {}
}

# --- 侧边栏：控制与参数设置 ---
st.sidebar.header("⚙️ 训练控制中心")

# --- 新增：timm 安装提示 --- 
st.sidebar.info("提示：部分骨干网络依赖 `timm` 库。若选择新网络无效，请尝试运行 `pip install timm` 安装。")

model_name = st.sidebar.text_input("为你的模型起个名字", "my_clothes_model")

# --- 数据集设置 ---
st.sidebar.subheader("💾 数据集路径")
# 移除旧的 root_dir 输入
# data_root = st.sidebar.text_input("数据集根目录", ...)

# 添加新的输入框，使用你提供的路径作为默认值
anno_dir_input = st.sidebar.text_input(
    "Anno_fine 目录绝对路径", 
    r"E:\AIModels\DeepFashion\DeepFashion\Category and Attribute Prediction Benchmark\Anno_fine",
    help="包含 train.txt, train_cate.txt 等标注文件的 Anno_fine 文件夹的完整路径。"
)
img_dir_input = st.sidebar.text_input(
    "高分辨率图片目录绝对路径", 
    r"E:\AIModels\DeepFashion\DeepFashion\Category and Attribute Prediction Benchmark\Img\img_highres",
    help="包含 MEN, WOMEN 等子目录的高分辨率图片文件夹 (如 img_highres) 的完整路径。"
)

# 简单检查路径是否存在
if not os.path.isdir(anno_dir_input):
    st.sidebar.warning(f"警告：Anno_fine 路径 '{anno_dir_input}' 不存在或不是目录。")
if not os.path.isdir(img_dir_input):
    st.sidebar.warning(f"警告：图片目录路径 '{img_dir_input}' 不存在或不是目录。")

# --- 新增：训练策略选择 --- 
st.sidebar.subheader("🎯 训练策略")
def get_strategy_index():
    # 获取当前 session_state 中策略的索引
    strategy_names = list(STRATEGIES.keys())
    current_strategy = st.session_state.get("selected_strategy", "均衡推荐 (Balanced)")
    try:
        return strategy_names.index(current_strategy)
    except ValueError:
        return 1 # 默认返回均衡推荐的索引

strategy_choice = st.sidebar.radio(
    "选择一个预设策略或手动设置:",
    list(STRATEGIES.keys()),
    index=get_strategy_index(), # 根据 session state 设置初始选项
    key="selected_strategy", # 使用 key 绑定到 session state
    help="选择预设策略会自动填充下方参数。选择后仍可手动修改。手动设置表示使用下方填写的参数。"
)

# 根据选择的策略获取默认值
strategy_defaults = STRATEGIES.get(strategy_choice, {})
# 如果是"手动设置"，则不使用策略默认值，允许用户输入或保留上次的值
is_manual_mode = (strategy_choice == "手动设置 (Manual)")

# --- 模型设置 ---
st.sidebar.subheader("🧠 模型架构")
# --- 修改：添加更多骨干网络选项 --- 
backbone_options = (
    'resnet18', 
    'resnet50', 
    'efficientnet_b0', 
    'efficientnet_b3', 
    'efficientnet_b4', 
    'swin_tiny_patch4_window7_224' # Swin Transformer Tiny
)
default_backbone = strategy_defaults.get('backbone', 'efficientnet_b3') if not is_manual_mode else st.session_state.get('backbone_input', 'efficientnet_b3')
# 确保默认值在选项列表中，如果不在，则使用列表中的第一个或一个安全的默认值
if default_backbone not in backbone_options:
    default_backbone_index = 3 # 默认 efficientnet_b3 的索引
else:
    default_backbone_index = backbone_options.index(default_backbone)

backbone = st.sidebar.selectbox(
    "选择骨干网络", 
    backbone_options,
    index=default_backbone_index,
    key='backbone_input',
    help="选择用于提取图像特征的基础网络结构。EfficientNet 通常效率更高。Swin Transformer 是较新的架构。"
)
# pretrained = st.sidebar.checkbox("使用预训练权重?", True, help="推荐勾选，使用在 ImageNet 上预训练的权重可以加快训练并提高效果。")
# 暂时强制使用预训练权重，简化选项
pretrained = True 

# --- 训练参数 ---
st.sidebar.subheader("⏱️ 训练参数")
# 为每个参数设置默认值，优先使用策略值，否则使用通用默认值或 session_state 中已有的值
default_epochs = strategy_defaults.get('epochs', 10) if not is_manual_mode else st.session_state.get('epochs_input', 10)
epochs = st.sidebar.number_input("训练轮次 (Epochs)", 
                               min_value=1, max_value=100, 
                               value=default_epochs, 
                               key='epochs_input', # 添加 key
                               help="模型完整学习一遍所有训练数据的次数。")

default_batch_size = strategy_defaults.get('batch_size', 32) if not is_manual_mode else st.session_state.get('batch_size_input', 32)
batch_size = st.sidebar.number_input("批次大小 (Batch Size)", 
                                   min_value=1, max_value=256, 
                                   value=default_batch_size, 
                                   key='batch_size_input', # 添加 key
                                   help="模型一次处理的图片数量。根据显存大小调整，越大通常越稳定，但更占显存。")

default_lr = strategy_defaults.get('learning_rate', 1e-4) if not is_manual_mode else st.session_state.get('learning_rate_input', 1e-4)
learning_rate = st.sidebar.number_input(
    "学习率 (Learning Rate)", 
    min_value=1e-6, max_value=1e-2, 
    value=default_lr, 
    format="%.1e", 
    key='learning_rate_input', # 添加 key
    help="模型学习的速度。太大会导致不稳定，太小会训练过慢。通常从 1e-4 或 1e-3 开始尝试。"
)

default_attr_weight = strategy_defaults.get('attribute_loss_weight', 1.0) if not is_manual_mode else st.session_state.get('attribute_loss_weight_input', 1.0)
attribute_loss_weight = st.sidebar.slider(
    "属性损失权重", 
    min_value=0.1, max_value=2.0, 
    value=default_attr_weight, 
    step=0.1, 
    key='attribute_loss_weight_input', # 添加 key
    help="调整类别任务和属性任务的重要性。增加此值会让模型更关注属性识别。"
)

# --- 设备与执行 ---
st.sidebar.subheader("💻 运行设备")
# --- 修改：移除 'cpu' 选项，除非没有 GPU 可选 --- 
device_options = ['auto'] # 'auto' 总是可用
device_indices = [] 
gpu_names = {}

if nvml_available and pynvml:
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                     name = pynvml.nvmlDeviceGetName(handle)
                except Exception as e:
                    name = f"GPU {i}" 
                device_options.append(f'cuda:{i} ({name})')
                device_indices.append(i)
                gpu_names[i] = name
        # else: # 如果没有 GPU，则只保留 'auto'，后续会处理
        #    pass 
    except pynvml.NVMLError as e:
        st.sidebar.warning(f"获取 GPU 信息失败: {e}")
# --- 如果没有 GPU 可选，则 'auto' 也无法工作，后面会处理 --- 
# 移除添加 'cpu' 的行
# device_options.append('cpu')

default_device_index = 0 # 默认选择 'auto'
device_choice_display = st.sidebar.selectbox(
    "选择运行设备 (需要 GPU)", 
    device_options, 
    index=default_device_index, 
    help="'auto' 会自动尝试使用第一个可用的 GPU。必须选择 GPU 进行训练。"
)

# 解析选择的设备
selected_device = None # 初始化为 None
selected_gpu_index = None

if not device_indices and device_choice_display == 'auto':
     # 如果选了 auto 但实际上没有 GPU
     st.sidebar.error("错误：未检测到可用的 NVIDIA GPU。无法使用 'auto' 或进行训练。")
     # selected_device 保持 None
elif device_choice_display == 'auto' and device_indices:
     # Auto 且有 GPU，选择第一个
     selected_device = f'cuda:{device_indices[0]}' 
     selected_gpu_index = device_indices[0]
elif device_choice_display != 'auto':
     # 明确选择了某个 GPU
     try:
         selected_device = device_choice_display.split()[0] # e.g., "cuda:0"
         selected_gpu_index = int(selected_device.split(':')[-1])
         if selected_gpu_index not in device_indices:
             st.sidebar.error(f"选择的 GPU cuda:{selected_gpu_index} 不可用。")
             selected_device = None # 标记为不可用
             selected_gpu_index = None
     except (IndexError, ValueError):
         st.sidebar.error("解析 GPU 设备选择失败。")
         selected_device = None
         selected_gpu_index = None

st.sidebar.markdown("--- ")

# --- 控制按钮 --- 
col_btn_1, col_btn_2 = st.sidebar.columns(2)
start_training = col_btn_1.button("🚀 开始训练！")
stop_training = col_btn_2.button("⏹️ 停止训练")

# 处理停止按钮点击
if stop_training:
    st.session_state.stop_requested = True
    st.sidebar.warning("收到停止请求...将在当前轮次结束后尝试停止。")

# --- GPU 实时监控占位符 ---
st.sidebar.markdown("--- ")
st.sidebar.subheader("📈 GPU 监控")
gpu_info_placeholder = st.sidebar.empty()
# --- 新增：GPU 图表占位符 --- 
gpu_util_chart_placeholder = st.sidebar.empty()
gpu_mem_chart_placeholder = st.sidebar.empty()

# 更新 GPU 监控的显示逻辑
if not nvml_available:
    gpu_info_placeholder.info("GPU 监控不可用 (未找到 pynvml)。")
elif not device_indices:
    gpu_info_placeholder.warning("未检测到 NVIDIA GPU，无法进行监控。")
elif selected_device is None:
    gpu_info_placeholder.error("未选择或无法确定有效的 GPU 设备。")
else:
    # --- 修改：传入图表占位符 --- 
    gpu_chart_placeholders = (gpu_util_chart_placeholder, gpu_mem_chart_placeholder)
    update_gpu_info(selected_gpu_index, gpu_info_placeholder, gpu_chart_placeholders) 
# --------------------------->

# --- 主区域：状态显示与结果 ---
col_main_1, col_main_2 = st.columns([2, 1]) # 状态/图表区 vs 日志区

with col_main_1:
    st.subheader("📊 训练状态与指标")
    status_placeholder = st.empty() # 用于显示状态信息
    overall_progress_bar = st.progress(0.0) # 整体 Epoch 进度条
    epoch_info_placeholder = st.empty() # 显示每轮的信息
    # --- 新增：时间信息占位符 --- 
    time_info_placeholder = st.empty()
    # --- 新增：报告和测试占位符 ---
    diagnostic_report_placeholder = st.empty()
    functional_test_placeholder = st.empty()
    # --- -------------------- ---
    loss_chart_placeholder = st.empty()
    acc_chart_placeholder = st.empty()

with col_main_2:
    st.subheader("📜 训练日志")
    # --- 修改：使用 st.code 代替 st.text_area --- 
    log_placeholder = st.empty() 
    # 初始化时显示空的代码块
    log_placeholder.code("", language='log')

# --- 新增：历史记录对比区域 --- 
st.markdown("--- ") # 分隔线
st.subheader("📜 历史训练对比")
history_results_placeholder = st.empty()

# 尝试加载并显示历史记录
def display_history():
    all_results = load_results()
    if not all_results:
        history_results_placeholder.info("尚未有训练记录。")
        return

    # 转换数据以便更好地显示
    display_data = []
    for r in reversed(all_results): # 显示最新的在前面
        best_epoch_info = f"{r.get('best_val_loss', 'N/A'):.4f} @ E{r.get('best_epoch', 'N/A')}" if isinstance(r.get('best_val_loss'), (int, float)) else "N/A"
        display_data.append({
            "完成时间": r.get("end_time_str", "N/A"),
            "模型名称": r.get("model_name", "N/A"),
            "策略": r.get("strategy", "N/A"),
            "骨干网络": r.get("backbone", "N/A"),
            "轮数": f"{r.get('completed_epochs', 'N/A')}/{r.get('total_epochs', 'N/A')}",
            "最佳验证损失 (轮)": best_epoch_info,
            # "最佳验证准确率": f"{r.get('best_val_acc', 'N/A'):.2f}%" if isinstance(r.get('best_val_acc'), (int, float)) else "N/A", # 暂时省略准确率
            "状态": r.get("status", "N/A").split('.')[0], # 取第一句
            "总耗时": r.get("duration_str", "N/A"),
            "模型路径": r.get("best_model_path", "N/A"),
            "功能测试": r.get("functional_test_result", "未执行"),
        })

    results_df = pd.DataFrame(display_data)
    history_results_placeholder.dataframe(results_df, use_container_width=True)

# 在应用加载时就显示历史记录
display_history()
# --- 历史记录对比区域结束 ---

# --- 训练逻辑触发 ---
if start_training:
    # --- 新增：启动前检查设备 --- 
    if selected_device is None or not selected_device.startswith('cuda'):
        error_msg = "错误：必须选择一个有效的 GPU 设备才能开始训练。请检查侧边栏的设备选择。"
        st.error(error_msg)
        append_log(error_msg)
        status_placeholder.error("❌ 训练无法开始：未选择有效 GPU。")
        st.stop() # 阻止后续代码执行
    # --- 设备检查结束 ---
    
    # --- 重置状态 --- 
    st.session_state.log_messages = [] 
    st.session_state.history_df_list = []
    st.session_state.stop_requested = False
    st.session_state.gpu_error_logged = False
    st.session_state.gpu_metrics_history = []
    st.session_state.gpu_poll_step = 0
    st.session_state.chart_error_logged = False
    # --- 新增：重置时间状态 --- 
    st.session_state.training_start_time = time.time() # 记录开始时间
    st.session_state.epoch_durations = []
    # --- 新增：清空报告占位符 --- 
    diagnostic_report_placeholder.empty()
    functional_test_placeholder.empty()
    time_info_placeholder.empty()
    gpu_util_chart_placeholder.empty()
    gpu_mem_chart_placeholder.empty()
    
    append_log("训练请求已接收，开始准备...")
    status_placeholder.info("⏳ 正在准备训练环境...")
    overall_progress_bar.progress(0.0)
    loss_chart_placeholder.empty()
    acc_chart_placeholder.empty()
    epoch_info_placeholder.empty()
    # --- 修改：初始化日志显示 --- 
    log_placeholder.code("\n".join(st.session_state.log_messages), language='log')
    
    if selected_gpu_index is not None:
        update_gpu_info(selected_gpu_index, gpu_info_placeholder, gpu_chart_placeholders)
    # --------------->
    
    # --- 收集参数 --- 
    model_save_dir = os.path.join('.', 'models', model_name)
    os.makedirs(model_save_dir, exist_ok=True) # 确保模型保存目录存在
    
    args = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': selected_device, # 现在确认是 'cuda:X' 了
        'model_save_path': model_save_dir, 
        'attribute_loss_weight': attribute_loss_weight,
        'num_workers': 0 # Windows 下多进程可能有问题，先用 0
    }
    append_log(f"使用的训练参数: {args}") # 日志记录实际使用的参数
    append_log(f"使用的骨干网络: {backbone}")
    append_log(f"Anno Dir Path: {anno_dir_input}")
    append_log(f"Image Dir Path: {img_dir_input}")
    # --------------->

    # --- 定义图像转换 --- 
    # 对于预训练模型，通常使用 ImageNet 的均值和标准差
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # 定义训练和验证的转换流程
    # TODO: 以后可以把图像大小 (224) 也作为参数
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

    # --- 1. 初始化数据集 ---
    try:
        status_placeholder.info("⏳ 正在加载数据集...")
        append_log("初始化训练数据集...")
        # !! 使用新的初始化方式，传入绝对路径 !!
        train_dataset = DeepFashionDataset(
            anno_dir_path=anno_dir_input, 
            image_dir_path=img_dir_input, 
            partition='train', 
            transform=train_transform
        )
        append_log(f"训练集加载成功，样本数: {len(train_dataset)}")
        
        append_log("初始化验证数据集...")
        # !! 使用新的初始化方式 !!
        val_dataset = DeepFashionDataset(
            anno_dir_path=anno_dir_input, 
            image_dir_path=img_dir_input, 
            partition='val', 
            transform=val_transform
        )
        append_log(f"验证集加载成功，样本数: {len(val_dataset)}")
        status_placeholder.success("✅ 数据集加载完成!")
    except FileNotFoundError as e:
        # 更新错误消息以反映新的路径输入
        error_msg = f"错误：找不到必要的数据文件或目录！请检查你输入的 Anno_fine 目录 '{anno_dir_input}' 和图片目录 '{img_dir_input}' 是否正确且存在。详细错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        status_placeholder.error("❌ 数据集加载失败！")
        st.stop()
    except ValueError as e: 
        # 更新错误消息
        error_msg = f"错误：加载或解析分区文件时出错。请检查 '{anno_dir_input}' 目录下的分区文件 (如 train.txt, train_cate.txt) 是否存在且格式正确。详细错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        status_placeholder.error("❌ 数据集加载失败！")
        st.stop()
    except Exception as e:
        error_msg = f"错误：加载数据集时发生未知错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        status_placeholder.error("❌ 数据集加载失败！")
        traceback.print_exc() # 打印详细堆栈到控制台
        st.stop()

    # --- 2. 初始化模型 ---
    model = None
    try:
        status_placeholder.info("⏳ 正在初始化模型...")
        append_log(f"初始化模型 (Backbone: {backbone})...") # 日志更新
        num_categories = 50 
        # 使用从下拉框获取的 backbone 初始化模型
        model = ClothesModel(
            num_categories=num_categories, 
            backbone=backbone # 确保这里传递的是最新的 backbone 值
        )
        append_log(f"模型初始化成功。将模型移动到设备: {selected_device}")
        model.to(selected_device)
        status_placeholder.success("✅ 模型初始化完成!")
    except Exception as e:
        error_msg = f"错误：初始化模型 '{backbone}' 时发生错误: {e}" # 包含 backbone 名称
        st.error(error_msg)
        append_log(error_msg)
        status_placeholder.error("❌ 模型初始化失败！请检查选择的骨干网络是否可用或已安装 `timm` 库。")
        traceback.print_exc()
        st.stop()
    # ------------------->

    # --- 3. 初始化 Trainer ---
    try:
        status_placeholder.info("⏳ 正在初始化训练器...")
        append_log("初始化 Trainer...")
        trainer = Trainer(model, train_dataset, val_dataset, args)
        append_log("Trainer 初始化成功.")
        status_placeholder.success("✅ 训练器准备就绪!")
    except Exception as e:
        error_msg = f"错误：初始化 Trainer 时发生错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        status_placeholder.error("❌ Trainer 初始化失败！")
        st.stop()
        
    # --- 4. 开始训练 (阻塞式) ---
    append_log("\n==================== 开始训练 ====================")
    status_placeholder.info(f"🚀 模型训练中... 设备: {selected_device}")
    start_time = time.time()
    best_val_loss = float('inf')
    training_interrupted = False
    history_df = None # 初始化 history_df
    training_successful = False # 标记训练是否正常完成

    # --- 准备存储结果的字典 --- 
    current_run_result = {
        "start_time": st.session_state.training_start_time,
        "start_time_str": datetime.fromtimestamp(st.session_state.training_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "model_name": model_name,
        "strategy": strategy_choice,
        "parameters": args, # 保存训练参数
        "backbone": backbone,
        "anno_dir": anno_dir_input, # 明确保存Anno_fine目录路径
        "status": "进行中",
        "total_epochs": epochs,
        "completed_epochs": 0,
        "best_val_loss": float('inf'),
        "best_epoch": None,
        "best_model_path": None,
        "diagnostic_summary": None, # 可以存报告的关键点
        "functional_test_result": "未执行",
        "end_time": None,
        "end_time_str": None,
        "duration": None,
        "duration_str": None,
    }
    # -------------------------->

    try: 
        for epoch in range(trainer.epochs):
            epoch_start_time = time.time()
            
            # --- 检查是否请求停止 --- 
            if st.session_state.get('stop_requested', False):
                append_log("训练被用户中断。")
                status_placeholder.warning("⚠️ 训练已停止。")
                break # 跳出 epoch 循环
            # ----------------------->

            status_placeholder.info(f"Epoch {epoch+1}/{trainer.epochs}: 正在训练...")
            append_log(f"\n--- 开始训练 Epoch {epoch+1}/{trainer.epochs} ---")
            if selected_gpu_index is not None:
                update_gpu_info(selected_gpu_index, gpu_info_placeholder, gpu_chart_placeholders)
            
            # --- 训练阶段 --- 
            trainer.model.train()
            train_loss, train_correct_cats, train_total_samples = 0.0, 0, 0
            num_batches = len(trainer.train_loader)
            
            for i, batch in enumerate(trainer.train_loader):
                if i % 10 == 0 and st.session_state.get('stop_requested', False):
                     append_log("训练在批次处理中被中断...")
                     training_interrupted = True
                     break
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
                    
                    if (i + 1) % 50 == 0 or (i + 1) == num_batches: # 每 50 个 batch 或最后一个 batch 更新
                         current_avg_loss = (train_loss / (i + 1)) / batch_size if batch_size > 0 else 0 
                         status_placeholder.info(f"Epoch {epoch+1}/{trainer.epochs}: 训练中... Batch {i+1}/{num_batches} ({(i + 1)*100/num_batches:.0f}%) | Batch Loss: {loss.item():.4f}")
                         # 更新 GPU 监控
                         if selected_gpu_index is not None:
                             update_gpu_info(selected_gpu_index, gpu_info_placeholder, gpu_chart_placeholders)
                except Exception as batch_e:
                    append_log(f"错误: Epoch {epoch+1}, Batch {i+1} 处理失败: {batch_e}")
                    traceback.print_exc()
                    continue 

            if training_interrupted: break 
            # --- 训练阶段结束 --->
            
            # --- 验证阶段 --- 
            avg_val_loss = float('nan') 
            avg_val_cat_acc = float('nan')
            if trainer.val_loader and len(trainer.val_loader.dataset) > 0:
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
                                append_log(f"警告: Epoch {epoch+1}, 验证 Batch {i+1}, 无效损失，跳过.")
                                continue
                            loss = loss_cat + trainer.attribute_loss_weight * loss_attr
                            
                            batch_size = images.size(0)
                            val_loss += loss.item() * batch_size
                            _, predicted_cats = torch.max(cat_logits.data, 1)
                            val_correct_cats += (predicted_cats == cat_labels).sum().item()
                            val_total_samples += batch_size
                        except Exception as batch_e:
                             append_log(f"错误: Epoch {epoch+1}, 验证 Batch {i+1} 处理失败: {batch_e}")
                             continue # 跳过当前批次

                avg_val_loss = val_loss / val_total_samples if val_total_samples else float('inf')
                avg_val_cat_acc = 100.0 * val_correct_cats / val_total_samples if val_total_samples else 0.0
            else:
                 append_log(f"--- Epoch {epoch+1} 无验证集或验证集为空，跳过验证 ---")
            epoch_time = time.time() - epoch_start_time
            append_log(f"--- Epoch {epoch+1} 验证完成 (Avg Loss: {avg_val_loss:.4f}, Cat Acc: {avg_val_cat_acc:.2f}%) --- Time: {epoch_time:.2f}s")
            # --- 验证阶段结束 --->

            # --- Epoch 结束处理 --- 
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            st.session_state.epoch_durations.append(epoch_duration)
            
            # --- 更新 Epoch 摘要、日志、图表、进度条、时间 --- 
            avg_train_loss = train_loss / len(trainer.train_loader.dataset) if len(trainer.train_loader.dataset) > 0 else float('inf')
            avg_train_cat_acc = 100.0 * train_correct_cats / train_total_samples if train_total_samples > 0 else 0.0
            append_log(f"--- Epoch {epoch+1} 训练完成 (Avg Loss: {avg_train_loss:.4f}, Cat Acc: {avg_train_cat_acc:.2f}%) ---")
            epoch_summary = (
                f"**Epoch {epoch+1}/{trainer.epochs}** | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_cat_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_cat_acc:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )
            epoch_info_placeholder.markdown(epoch_summary)
            current_epoch_history = {
                'epoch': epoch + 1,
                'Train Loss': avg_train_loss if math.isfinite(avg_train_loss) else None, # 处理 Inf
                'Validation Loss': avg_val_loss if math.isfinite(avg_val_loss) else None,
                'Train Accuracy (%)': avg_train_cat_acc,
                'Validation Accuracy (%)': avg_val_cat_acc
            }
            st.session_state.history_df_list.append(current_epoch_history)
            history_df = pd.DataFrame(st.session_state.history_df_list).dropna(subset=['epoch'])
            if not history_df.empty:
                plot_df_loss = history_df[['epoch', 'Train Loss', 'Validation Loss']].set_index('epoch')
                plot_df_acc = history_df[['epoch', 'Train Accuracy (%)', 'Validation Accuracy (%)']].set_index('epoch')
                with loss_chart_placeholder.container():
                     st.line_chart(plot_df_loss)
                with acc_chart_placeholder.container():
                     st.line_chart(plot_df_acc)
            overall_progress_bar.progress((epoch + 1) / trainer.epochs)
            
            # --- 新增：更新时间信息 --- 
            elapsed_time = epoch_end_time - st.session_state.training_start_time
            avg_epoch_time = sum(st.session_state.epoch_durations) / len(st.session_state.epoch_durations) if st.session_state.epoch_durations else None
            remaining_epochs = trainer.epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs if avg_epoch_time is not None and remaining_epochs > 0 else None
            estimated_completion_time = datetime.now() + timedelta(seconds=int(eta_seconds)) if eta_seconds is not None else None
            
            time_info_str = (
                f"⏱️ **时间统计:**  "
                f"已运行: {format_time_delta(elapsed_time)} | "
                f"平均每轮: {avg_epoch_time:.2f} 秒 | "
                f"预计剩余: {format_time_delta(eta_seconds)} | "
                f"预计完成时间: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S') if estimated_completion_time else 'N/A'}"
            )
            time_info_placeholder.markdown(time_info_str)
            # --- 时间信息更新结束 ---
            
            log_placeholder.code("\n".join(st.session_state.log_messages), language='log')
            
            if selected_gpu_index is not None:
                update_gpu_info(selected_gpu_index, gpu_info_placeholder, gpu_chart_placeholders)
            
            # --- 保存最佳模型 --- 
            if math.isfinite(avg_val_loss) and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if trainer.model_save_path:
                    # 清理旧的最佳模型文件，避免过多文件积累
                    for old_file in os.listdir(trainer.model_save_path):
                        if old_file.startswith(f"best_model_{model_name}") and old_file.endswith(".pth"):
                            try:
                                os.remove(os.path.join(trainer.model_save_path, old_file))
                                append_log(f"已删除旧的最佳模型: {old_file}")
                            except OSError as e:
                                append_log(f"无法删除旧模型 {old_file}: {e}")
                    # 保存新的最佳模型
                    save_filename = os.path.join(trainer.model_save_path, f"best_model_{model_name}_epoch{epoch+1}.pth")
                    try:
                        torch.save(trainer.model.state_dict(), save_filename)
                        append_log(f"** 新的最佳模型已保存到: {save_filename} (Val Loss: {avg_val_loss:.4f}) **")
                        best_model_file_path = save_filename # 记录路径
                        current_run_result["best_val_loss"] = avg_val_loss # 更新记录中的最佳损失
                        current_run_result["best_epoch"] = epoch + 1 # 更新记录中的最佳轮次
                        current_run_result["best_model_path"] = best_model_file_path # 更新记录中的路径
                    except Exception as e:
                        append_log(f"错误：保存模型时出错: {e}")
                else:
                    append_log("模型保存路径未设置，跳过保存最佳模型")

            # --- 更新 Epoch 摘要、日志、图表、进度条、时间 --- 
            current_run_result["completed_epochs"] = epoch + 1 # 更新完成的轮数
            
            # --- 新增：更新时间信息 --- 
            elapsed_time = epoch_end_time - st.session_state.training_start_time
            avg_epoch_time = sum(st.session_state.epoch_durations) / len(st.session_state.epoch_durations) if st.session_state.epoch_durations else None
            remaining_epochs = trainer.epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs if avg_epoch_time is not None and remaining_epochs > 0 else None
            estimated_completion_time = datetime.now() + timedelta(seconds=int(eta_seconds)) if eta_seconds is not None else None
            
            time_info_str = (
                f"⏱️ **时间统计:**  "
                f"已运行: {format_time_delta(elapsed_time)} | "
                f"平均每轮: {avg_epoch_time:.2f} 秒 | "
                f"预计剩余: {format_time_delta(eta_seconds)} | "
                f"预计完成时间: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S') if estimated_completion_time else 'N/A'}"
            )
            time_info_placeholder.markdown(time_info_str)
            # --- 时间信息更新结束 ---
            
            log_placeholder.code("\n".join(st.session_state.log_messages), language='log')
            
            if selected_gpu_index is not None:
                update_gpu_info(selected_gpu_index, gpu_info_placeholder, gpu_chart_placeholders)
            
        # --- Epoch 循环结束 --->
        training_successful = not training_interrupted # 如果没中断就算成功
        
        # 如果训练成功完成，更新状态为"已完成"
        if training_successful:
            current_run_result["status"] = "已完成"
        
    except Exception as e:
        error_msg = f"错误：训练过程中发生严重错误: {e}"
        st.error(error_msg)
        append_log(error_msg)
        status_placeholder.error("❌ 训练失败！")
        traceback.print_exc()
        current_run_result["status"] = "错误"
    finally:
        # --- 训练结束处理 --- 
        end_time = time.time()
        total_time = end_time - current_run_result["start_time"]
        current_run_result["end_time"] = end_time
        current_run_result["end_time_str"] = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        current_run_result["duration"] = total_time
        current_run_result["duration_str"] = format_time_delta(total_time)
        
        append_log(f"总训练时间: {total_time:.2f} 秒")
        formatted_best_loss = f"{best_val_loss:.4f}" if math.isfinite(best_val_loss) else "N/A"
        append_log(f"最佳验证损失: {formatted_best_loss}")
        
        # --- 最终时间信息更新 --- 
        final_avg_epoch_time = sum(st.session_state.epoch_durations) / len(st.session_state.epoch_durations) if st.session_state.epoch_durations else None
        final_time_info_str = (
                f"⏱️ **最终统计:**  "
                f"总耗时: {format_time_delta(total_time)} | "
                f"平均每轮: {final_avg_epoch_time:.2f} 秒 (共 {len(st.session_state.epoch_durations)} 轮)"
            )
        time_info_placeholder.markdown(final_time_info_str) 
        
        # --- 修改：确保最后一次更新日志 --- 
        log_placeholder.code("\n".join(st.session_state.log_messages), language='log')

        # --- 生成报告和执行测试 --- 
        history_df = pd.DataFrame(st.session_state.history_df_list).dropna(subset=['epoch'])
        diagnostic_report = ""
        # 1. 生成诊断报告
        if not history_df.empty:
            diagnostic_report = generate_diagnostic_report(history_df, best_val_loss, trainer.epochs)
            diagnostic_report_placeholder.markdown(diagnostic_report)
            append_log("\n--- 诊断报告已生成 ---") # 简化日志
            # --- 修改：存储报告摘要 --- 
            current_run_result["diagnostic_summary"] = diagnostic_report # 存完整报告，也可以只存关键点
        else:
            current_run_result["diagnostic_summary"] = "无训练历史数据"

        # 2. 执行功能模拟测试
        test_success = False # 初始化
        if training_successful and current_run_result["best_model_path"] is not None: # 使用记录中的路径
             model_config = {
                 'num_categories': num_categories, 
                 'backbone': backbone 
             }
             # --- 修改：使用记录的最佳模型路径 --- 
             test_report, test_success = run_functional_test(model_save_dir, model_name, model_config, selected_device)
             functional_test_placeholder.markdown(test_report)
             log_summary = "功能测试成功" if test_success else "功能测试失败"
             append_log(f"\n--- 功能模拟测试 --- \n{log_summary}")
             current_run_result["functional_test_result"] = "成功" if test_success else "失败"
        elif not training_successful:
             functional_test_placeholder.warning("训练未成功完成，跳过功能测试。")
             append_log("训练未成功完成，跳过功能测试。")
             current_run_result["functional_test_result"] = "跳过 (训练失败)"
        else: 
             functional_test_placeholder.warning("未找到有效的最佳模型文件，跳过功能测试。")
             append_log("未找到有效的最佳模型文件，跳过功能测试。")
             current_run_result["functional_test_result"] = "跳过 (无模型)"

        # --- 保存当前运行结果 --- 
        all_results = load_results()
        all_results.append(current_run_result)
        save_results(all_results)
        append_log(f"当前训练结果已追加到 {RESULTS_FILE}")
        # --- 刷新历史记录显示 --- 
        display_history()
        # ------------------------>

        # 清理 NVML
        if nvml_available and pynvml:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                append_log(f"关闭 NVML 时出错: {e}")

# --- 训练逻辑触发结束 ---

# 只有在按钮没有被点击时，才显示初始提示
if not start_training:
    if not st.session_state.get('log_messages'):
        status_placeholder.info("请在左侧设置参数并点击 '开始训练！' 按钮。")
        # --- 修改：初始化日志显示 --- 
        log_placeholder.code("尚未开始训练...", language='log')
    else:
        # 保留上次训练的最终状态 (日志和图表应该还在)
        pass 

# --- UI 代码结束 --- 