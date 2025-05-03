import streamlit as st
import os
import platform
import subprocess
import json
from datetime import datetime

def select_folder():
    """使用系统命令选择文件夹，支持Windows和Linux"""
    try:
        system = platform.system().lower()
        if system == 'windows':
            # Windows下使用PowerShell的文件夹选择对话框
            command = '''powershell -command "& {
                Add-Type -AssemblyName System.Windows.Forms
                $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
                $folderBrowser.Description = '选择文件夹'
                $folderBrowser.RootFolder = 'MyComputer'
                if ($folderBrowser.ShowDialog() -eq 'OK') {
                    Write-Host $folderBrowser.SelectedPath
                }
            }"'''
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            selected_path = result.stdout.strip()
        else:
            # Linux下检查是否安装了zenity
            try:
                subprocess.run(['which', 'zenity'], check=True, capture_output=True)
                # 使用zenity的文件选择对话框
                command = ['zenity', '--file-selection', '--directory', '--title=选择文件夹']
                result = subprocess.run(command, capture_output=True, text=True)
                selected_path = result.stdout.strip()
            except subprocess.CalledProcessError:
                # 如果没有安装zenity，使用命令行的文件浏览器
                command = ['dialog', '--title', '选择文件夹', '--dselect', '/', 0, 0]
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    selected_path = result.stdout.strip()
                except subprocess.CalledProcessError:
                    st.warning("请手动输入路径。要使用图形化选择，请安装 zenity：`sudo apt-get install zenity`")
                    return None

        return selected_path if selected_path else None
    except Exception as e:
        st.warning(f"文件夹选择失败，请手动输入路径。错误: {e}")
        return None

def create_config_panel(ANNO_DIR, IMG_DIR, CONFIG_PATH, nvml_available, gpu_names):
    """创建训练参数配置面板"""
    st.sidebar.header("⚙️ 训练控制中心")
    
    # --- 预设训练策略 ---
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
            "batch_size": 32,
            "learning_rate": 5e-5,
            "backbone": 'efficientnet_b4',
            "attribute_loss_weight": 1.0
        },
        "手动设置 (Manual)": {}
    }
    
    # --- timm 安装提示 ---
    st.sidebar.info("提示：部分骨干网络依赖 `timm` 库。若选择新网络无效，请尝试运行 `pip install timm` 安装。")
    
    # --- 数据集路径设置 ---
    st.sidebar.subheader("💾 数据集路径")
    
    # Anno_fine 目录选择
    anno_col1, anno_col2 = st.sidebar.columns([3, 1])
    with anno_col1:
        anno_dir_input = st.text_input(
            "标注文件目录 (Anno_fine):",
            value=ANNO_DIR if ANNO_DIR else "",
            key="anno_dir_input"
        )
    with anno_col2:
        if st.button("浏览", key="browse_anno_btn", help="点击选择Anno_fine目录。如果按钮不可用，请手动输入路径。"):
            selected_path = select_folder()
            if selected_path:
                st.session_state.anno_dir_input = selected_path
                # 更新配置文件
                try:
                    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    config['anno_dir'] = selected_path
                    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    st.success("✅ Anno_fine 路径已更新")
                except Exception as e:
                    st.error(f"❌ 更新配置文件失败: {e}")
    
    # 图片目录选择
    img_col1, img_col2 = st.sidebar.columns([3, 1])
    with img_col1:
        img_dir_input = st.text_input(
            "高分辨率图片目录 (img_highres):",
            value=IMG_DIR if IMG_DIR else "",
            key="img_dir_input"
        )
    with img_col2:
        if st.button("浏览", key="browse_img_btn", help="点击选择图片目录。如果按钮不可用，请手动输入路径。"):
            selected_path = select_folder()
            if selected_path:
                st.session_state.img_dir_input = selected_path
                # 更新配置文件
                try:
                    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    config['img_dir'] = selected_path
                    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    st.success("✅ 图片目录路径已更新")
                except Exception as e:
                    st.error(f"❌ 更新配置文件失败: {e}")
    
    # 简单检查路径是否存在
    if not os.path.isdir(anno_dir_input):
        st.sidebar.warning(f"警告：Anno_fine 路径 '{anno_dir_input}' 不存在或不是目录。")
    if not os.path.isdir(img_dir_input):
        st.sidebar.warning(f"警告：图片目录路径 '{img_dir_input}' 不存在或不是目录。")
    
    # 如果用户修改了路径，更新配置文件
    if hasattr(st.session_state, 'last_anno_dir') and hasattr(st.session_state, 'last_img_dir'):
        if anno_dir_input != st.session_state.last_anno_dir or img_dir_input != st.session_state.last_img_dir:
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                config['anno_dir'] = anno_dir_input
                config['img_dir'] = img_dir_input
                with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                st.sidebar.success("✅ 路径已更新到配置文件")
            except Exception as e:
                st.sidebar.error(f"❌ 更新配置文件失败: {e}")
    
    # 更新路径记录
    st.session_state.last_anno_dir = anno_dir_input
    st.session_state.last_img_dir = img_dir_input
    
    # --- 训练策略选择 ---
    st.sidebar.subheader("🎯 训练策略")
    
    # 初始化默认策略（仅在session_state中不存在时）
    if "selected_strategy" not in st.session_state:
        st.session_state.selected_strategy = "均衡推荐 (Balanced)"
    
    strategy_choice = st.sidebar.radio(
        "选择一个预设策略或手动设置:",
        list(STRATEGIES.keys()),
        key="selected_strategy",
        help="选择预设策略会自动填充下方参数。选择后仍可手动修改。手动设置表示使用下方填写的参数。"
    )
    
    # 根据选择的策略获取默认值
    strategy_defaults = STRATEGIES.get(strategy_choice, {})
    # 如果是"手动设置"，则不使用策略默认值，允许用户输入或保留上次的值
    is_manual_mode = (strategy_choice == "手动设置 (Manual)")
    
    # --- 模型架构设置 ---
    st.sidebar.subheader("🧠 模型架构")
    
    backbone_options = (
        'resnet18', 
        'resnet50', 
        'efficientnet_b0', 
        'efficientnet_b3', 
        'efficientnet_b4', 
        'swin_tiny_patch4_window7_224'
    )
    
    default_backbone = strategy_defaults.get('backbone', 'efficientnet_b3') if not is_manual_mode else st.session_state.get('backbone_input', 'efficientnet_b3')
    # 确保默认值在选项列表中
    if default_backbone not in backbone_options:
        default_backbone_index = 3  # 默认 efficientnet_b3 的索引
    else:
        default_backbone_index = backbone_options.index(default_backbone)
    
    backbone = st.sidebar.selectbox(
        "选择骨干网络", 
        backbone_options,
        index=default_backbone_index,
        key='backbone_input',
        help="选择用于提取图像特征的基础网络结构。EfficientNet 通常效率更高。Swin Transformer 是较新的架构。"
    )
    
    # 使用预训练权重(固定为True)
    pretrained = True
    
    # --- 训练参数 ---
    st.sidebar.subheader("⏱️ 训练参数")
    
    # 训练轮数设置
    default_epochs = strategy_defaults.get('epochs', 15) if not is_manual_mode else st.session_state.get('epochs_input', 15)
    epochs = st.sidebar.number_input(
        "训练轮数 (epochs)",
        min_value=1,
        max_value=100,
        value=default_epochs,
        step=1,
        key='epochs_input',
        help="模型训练的总轮数。轮数越多，训练时间越长，效果可能越好。"
    )
    
    # 批次大小设置
    default_batch_size = strategy_defaults.get('batch_size', 32) if not is_manual_mode else st.session_state.get('batch_size_input', 32)
    batch_size = st.sidebar.number_input(
        "批次大小 (batch size)",
        min_value=1,
        max_value=512,
        value=default_batch_size,
        step=1,
        key='batch_size_input',
        help="每次训练处理的样本数量。数值越大训练越快，但需要更多显存。"
    )
    
    # 学习率设置 - 简化为单一直观的控件
    # 获取默认学习率值
    default_lr = strategy_defaults.get('learning_rate', 1e-4) if not is_manual_mode else st.session_state.get('learning_rate_input', 1e-4)
    
    # 使用单个直观的数值输入框
    learning_rate = st.sidebar.number_input(
        "学习率 (Learning Rate)",
        min_value=1e-6, 
        max_value=1e-1, 
        value=default_lr, 
        format="%.1e", 
        key='learning_rate_input',
        help="模型学习的速度。通常建议：快速训练使用1e-3~5e-4，精细训练使用1e-4~5e-5。太大会导致不稳定，太小会训练过慢。"
    )
    
    # 属性损失权重
    default_attr_weight = strategy_defaults.get('attribute_loss_weight', 1.0) if not is_manual_mode else st.session_state.get('attribute_loss_weight_input', 1.0)
    attribute_loss_weight = st.sidebar.slider(
        "属性损失权重", 
        min_value=0.1, max_value=2.0, 
        value=default_attr_weight, 
        step=0.1, 
        key='attribute_loss_weight_input',
        help="调整类别任务和属性任务的重要性。增加此值会让模型更关注属性识别。"
    )
    
    # --- 生成模型名称 ---
    def generate_model_name(backbone, epochs, batch_size, learning_rate):
        """根据训练参数自动生成模型名称，并添加简短时间戳避免重复"""
        # 处理学习率字符串
        lr_str = f"{learning_rate:.1e}"
        parts = lr_str.split('e')
        base = parts[0].replace('.', '')
        exp = parts[1]
        
        if len(base) == 1:
            base = f"{base}0"
        
        lr_formatted = f"{base}E{exp[1:]}".upper()
        backbone_upper = backbone.upper()
        
        # 添加简短时间戳 (MMDD_HHMM 格式)
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        return f"MD_{backbone_upper}_{epochs}_{batch_size}_{lr_formatted}_{timestamp}"
    
    # 更新模型名称
    current_params = (backbone, epochs, batch_size, learning_rate)
    if 'last_params' not in st.session_state:
        st.session_state.last_params = current_params
        
    params_changed = current_params != st.session_state.last_params
    
    if params_changed:
        st.session_state.last_params = current_params
        st.session_state.model_name = generate_model_name(backbone, epochs, batch_size, learning_rate)
    elif 'model_name' not in st.session_state:
        st.session_state.model_name = generate_model_name(backbone, epochs, batch_size, learning_rate)
    
    # 显示模型名称输入框
    model_name = st.sidebar.text_input("为你的模型起个名字", st.session_state.model_name)
    
    # --- 设备选择 ---
    st.sidebar.subheader("💻 运行设备")
    
    device_options = ['auto']
    device_indices = []
    
    if nvml_available:
        try:
            from components.gpu_monitor import get_gpu_count, get_gpu_name
            device_count = get_gpu_count()
            if device_count > 0:
                for i in range(device_count):
                    name = get_gpu_name(i)
                    device_options.append(f'cuda:{i} ({name})')
                    device_indices.append(i)
        except Exception as e:
            st.sidebar.warning(f"获取 GPU 信息失败: {e}")
    
    default_device_index = 0  # 默认选择 'auto'
    device_choice_display = st.sidebar.selectbox(
        "选择运行设备 (需要 GPU)", 
        device_options, 
        index=default_device_index, 
        help="'auto' 会自动尝试使用第一个可用的 GPU。必须选择 GPU 进行训练。"
    )
    
    # 解析选择的设备
    selected_device = None
    selected_gpu_index = None
    
    if not device_indices and device_choice_display == 'auto':
        st.sidebar.error("错误：未检测到可用的 NVIDIA GPU。无法使用 'auto' 或进行训练。")
    elif device_choice_display == 'auto' and device_indices:
        selected_device = f'cuda:{device_indices[0]}'
        selected_gpu_index = device_indices[0]
    elif device_choice_display != 'auto':
        try:
            selected_device = device_choice_display.split()[0]  # e.g., "cuda:0"
            selected_gpu_index = int(selected_device.split(':')[-1])
            if selected_gpu_index not in device_indices:
                st.sidebar.error(f"选择的 GPU cuda:{selected_gpu_index} 不可用。")
                selected_device = None
                selected_gpu_index = None
        except (IndexError, ValueError):
            st.sidebar.error("解析 GPU 设备选择失败。")
            selected_device = None
            selected_gpu_index = None
    
    # 收集所有训练参数
    training_params = {
        'model_name': model_name,
        'backbone': backbone,
        'pretrained': pretrained,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'attribute_loss_weight': attribute_loss_weight,
        'strategy_choice': strategy_choice,
        'anno_dir': anno_dir_input,
        'img_dir': img_dir_input,
        'num_workers': 0  # Windows 下多进程可能有问题，先用 0
    }
    
    return selected_device, selected_gpu_index, training_params 