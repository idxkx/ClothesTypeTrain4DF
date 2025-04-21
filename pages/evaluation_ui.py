# Placeholder for evaluation UI code 

import streamlit as st
import torch
import os
import json
import pandas as pd
from PIL import Image
import traceback
import math
from torchvision import transforms

# --- 常量 ---
RESULTS_FILE = "../training_results.json" # 结果文件相对于 pages 目录
CATEGORY_FILE = "list_category_cloth.txt"
ATTRIBUTE_FILE = "list_attr_cloth.txt"
# --- 新增：映射文件常量 --- 
MAPPING_FILE = "../name_mapping.json" 

# --- 模型导入 ---
# 假设 model.py 在项目根目录
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from model import ClothesModel
except ImportError as e:
    st.error(f"错误：无法导入 ClothesModel。请确保 model.py 在项目根目录下。错误: {e}")
    st.stop()

# --- 辅助函数 ---
# (从 main_ui.py 复制并可能稍作修改)
def load_results():
    """加载历史训练结果"""
    results_path = os.path.join(os.path.dirname(__file__), RESULTS_FILE)
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results if isinstance(results, list) else []
    except FileNotFoundError:
        st.warning(f"未找到训练结果文件: {results_path}。请先在主页面完成至少一次训练。")
        return []
    except json.JSONDecodeError:
        st.error(f"训练结果文件 {results_path} 格式错误。")
        return []

def load_category_names(anno_dir):
    """从 Anno_fine 目录加载类别名称"""
    file_path = os.path.join(anno_dir, CATEGORY_FILE)
    names = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[2:]):
                parts = line.strip().split()
                if len(parts) >= 1:
                    category_name = ' '.join(parts[:-1])
                    names[i + 1] = category_name # Key is ID (1-50)
    except FileNotFoundError:
        st.error(f"错误：找不到类别文件 {file_path}")
        return None
    except Exception as e:
        st.error(f"读取类别文件 {file_path} 时出错: {e}")
        return None
    if len(names) != 50:
        st.warning(f"警告：从 {file_path} 加载了 {len(names)} 个类别名称，预期是 50 个。")
    return names

def load_attribute_names(anno_dir):
    """从 Anno_fine 目录加载属性名称"""
    file_path = os.path.join(anno_dir, ATTRIBUTE_FILE)
    names = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[2:]):
                parts = line.strip().split()
                if len(parts) >= 1:
                    attribute_name = ' '.join(parts[:-1])
                    names[i] = attribute_name # Key is index (0-999)
    except FileNotFoundError:
        st.error(f"错误：找不到属性文件 {file_path}")
        return None
    except Exception as e:
        st.error(f"读取属性文件 {file_path} 时出错: {e}")
        return None
    # --- 修改：更新预期的属性数量 --- 
    expected_attrs = 26
    if len(names) != expected_attrs:
         st.warning(f"警告：从 {file_path} 加载了 {len(names)} 个属性名称，预期是 {expected_attrs} 个。")
    return names

# --- 新增：加载名称映射 --- 
def load_name_mapping():
    mapping_path = os.path.join(os.path.dirname(__file__), MAPPING_FILE)
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        # 提供默认空字典，防止文件不包含某个键
        return mapping.get("categories", {}), mapping.get("attributes", {})
    except FileNotFoundError:
        st.info(f"提示：未找到名称映射文件 {mapping_path}，将仅显示英文名称。")
        return {}, {}
    except json.JSONDecodeError:
        st.error(f"名称映射文件 {mapping_path} 格式错误。")
        return {}, {}

# --- 加载映射 --- 
category_mapping, attribute_mapping = load_name_mapping()

# 图像预处理 (与验证集相同)
# TODO: 使图像大小和归一化参数与训练配置一致 (可以从 results.json 获取?)
img_size = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize,
])

# --- UI 布局 ---
st.set_page_config(page_title="模型效果测试", layout="wide")
st.title("🧪 模型效果测试")
st.markdown("从训练好的模型中选择一个，上传服装图片，查看识别结果。")

# --- 加载和选择模型 ---
all_results = load_results()
# 筛选出成功的训练运行
successful_runs = [
    r for r in all_results
    if r.get("status") == "已完成" and
       r.get("functional_test_result") == "成功" and
       r.get("best_model_path") and
       os.path.exists(os.path.join(os.path.dirname(__file__), '..', r["best_model_path"])) # 检查文件是否存在
]

model_options = {"请选择模型": None}
if successful_runs:
    for run in sorted(successful_runs, key=lambda x: x.get("end_time", 0), reverse=True):
        option_label = f"{run.get('model_name', '未知模型')} (完成于 {run.get('end_time_str', '未知时间')}, Backbone: {run.get('backbone', '未知')})"
        model_options[option_label] = {
            "path": os.path.join(os.path.dirname(__file__), '..', run["best_model_path"]), 
            "backbone": run.get("backbone"),
            # --- 修改：直接存储 anno_dir --- 
            "anno_dir": run.get("parameters", {}).get("anno_dir_input", None) # 从原始参数获取
        }
else:
    st.warning("下拉列表中没有找到符合条件的模型记录。你可以尝试在下方手动指定模型路径。")

st.selectbox(
    "通过下拉列表选择模型 (推荐):",
    list(model_options.keys()),
    key="selected_model_dropdown"
)

selected_model_info = model_options.get(st.session_state.selected_model_dropdown)

# --- 手动指定模型路径 --- 
st.markdown("--- ")
st.markdown("**或者，手动指定模型文件路径进行测试：**")
manual_model_path_input = st.text_input(
    "模型文件绝对路径 (.pth):",
    key="manual_model_path", # Add key for state tracking
    help="输入你想要测试的模型文件的完整路径，例如 C:\path\to\your\model.pth"
)
manual_backbone_input = st.text_input(
    "该模型使用的骨干网络名称:",
    key="manual_backbone", # Add key
    help="输入与上述模型文件匹配的 Backbone 名称，例如 efficientnet_b3"
)

# --- 恢复 Anno_fine 路径输入框，并使其根据上下文显示/填充 --- 
st.markdown("--- ")
anno_dir = None
show_anno_input = True # Default to showing the input

if st.session_state.selected_model_dropdown != "请选择模型" and selected_model_info:
    # If dropdown is used and info is valid
    retrieved_anno_dir = selected_model_info.get("anno_dir")
    if retrieved_anno_dir and os.path.isdir(retrieved_anno_dir):
        anno_dir = retrieved_anno_dir # Use the retrieved path
        show_anno_input = False # Hide the input box if path is valid
    # If retrieved_anno_dir is invalid or missing, show_anno_input remains True

# Only show the input field if necessary
anno_dir_input_value = anno_dir if anno_dir else "" # Default value for input
if show_anno_input:
    st.warning("需要提供 Anno_fine 目录以解释模型输出。")
    anno_dir_input_field = st.text_input(
        "Anno_fine 目录绝对路径:",
        value=anno_dir_input_value,
        key="anno_dir_input", # Add key
        help="包含 list_category_cloth.txt 和 list_attr_cloth.txt 的目录。"
    )
else:
    st.success(f"已自动从训练记录加载 Anno_fine 目录: `{anno_dir}`")
    # Keep the key in session state even if hidden, for consistency
    if "anno_dir_input" not in st.session_state:
        st.session_state.anno_dir_input = anno_dir
    else:
        st.session_state.anno_dir_input = anno_dir # Ensure it's updated
    anno_dir_input_field = anno_dir # Use the automatically found dir

# --- 图片上传 ---
uploaded_file = st.file_uploader(
    "上传一张服装图片:",
    type=["jpg", "jpeg", "png"]
)

# --- 识别按钮和结果显示 ---
col_img, col_results = st.columns(2)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        with col_img:
            st.image(image, caption="上传的图片", use_column_width=True)
    except Exception as e:
        st.error(f"无法加载图片: {e}")
        uploaded_file = None # 阻止后续处理

if st.button("🚀 开始识别！"):
    # --- 确定要使用的模型信息 和 Anno Dir --- 
    model_to_use = None
    backbone_to_use = None
    path_to_use = None
    # --- 修改：从 session_state 或自动获取 anno_dir --- 
    final_anno_dir = None 

    # Determine model path and backbone first
    use_manual_path = bool(st.session_state.get("manual_model_path"))
    if use_manual_path:
        st.write("使用手动指定的模型路径进行识别...")
        manual_path = st.session_state.manual_model_path
        manual_backbone = st.session_state.get("manual_backbone")
        if not os.path.exists(manual_path):
            st.error(f"错误：手动指定的模型路径不存在: {manual_path}")
        elif not manual_path.endswith(".pth"):
            st.error("错误：手动指定的模型路径必须指向一个 .pth 文件。")
        elif not manual_backbone:
            st.error("错误：使用手动路径时，必须同时指定骨干网络名称。")
        else:
            path_to_use = manual_path
            backbone_to_use = manual_backbone
            # For manual path, anno_dir MUST come from the input field
            if show_anno_input: # If the input field was shown
                 final_anno_dir = st.session_state.get("anno_dir_input")
                 if not final_anno_dir or not os.path.isdir(final_anno_dir):
                      st.error(f"错误：使用手动模型路径时，请在上方输入有效的 Anno_fine 目录路径。")
                      path_to_use = None # Prevent proceeding
            else:
                 # This case should theoretically not happen if logic is correct
                 # but as a safeguard, try to use the auto-retrieved one
                 final_anno_dir = anno_dir 
                 if not final_anno_dir or not os.path.isdir(final_anno_dir):
                      st.error(f"错误：无法确定 Anno_fine 目录路径。")
                      path_to_use = None

    elif selected_model_info: # Using dropdown
        st.write("使用下拉列表选择的模型进行识别...")
        path_to_use = selected_model_info.get("path")
        backbone_to_use = selected_model_info.get("backbone")
        retrieved_anno_dir = selected_model_info.get("anno_dir")

        if not path_to_use or not backbone_to_use:
             st.error("错误：从下拉列表选择的模型信息不完整 (路径或Backbone)。")
             path_to_use = None 
        elif retrieved_anno_dir and os.path.isdir(retrieved_anno_dir):
             final_anno_dir = retrieved_anno_dir # Use the valid retrieved path
        else:
             # Dropdown used, but anno_dir was missing or invalid, check input field
             if show_anno_input: # Input field should be visible in this case
                  final_anno_dir = st.session_state.get("anno_dir_input")
                  if not final_anno_dir or not os.path.isdir(final_anno_dir):
                       st.error(f"错误：无法从训练记录获取 Anno_fine 目录，请在上方输入有效路径。")
                       path_to_use = None # Prevent proceeding
             else:
                 # Should not happen if show_anno_input logic is correct
                 st.error("内部错误：无法确定 Anno_fine 目录路径。")
                 path_to_use = None
    else:
        st.error("请先通过下拉列表选择一个模型，或手动指定模型路径。")
    # --- 模型信息和 Anno Dir 确定结束 ---

    # --- 后续逻辑使用 path_to_use, backbone_to_use, final_anno_dir --- 
    if path_to_use and backbone_to_use and final_anno_dir and uploaded_file:
        # --- anno_dir 已确定，不再需要检查 anno_dir_input ---

        model_path = path_to_use
        backbone = backbone_to_use
        num_categories = 50 
        num_attributes = 26 

        # 加载类别和属性名称 (使用确定的 final_anno_dir)
        category_names_en = load_category_names(final_anno_dir)
        attribute_names_en = load_attribute_names(final_anno_dir)

        if category_names_en is None or attribute_names_en is None:
            st.error("无法加载类别或属性名称，无法继续识别。")
        else:
            # --- 开始处理 ---
            with st.spinner("正在加载模型并进行识别..."):
                try:
                    # 1. 加载模型
                    model = ClothesModel(num_categories=num_categories, backbone=backbone)
                    # 尝试自动选择设备
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()

                    # 2. 预处理图片
                    img_tensor = eval_transform(image).unsqueeze(0).to(device) # 添加 batch 维度

                    # 3. 模型推理
                    with torch.no_grad():
                        cat_logits, attr_logits = model(img_tensor)

                    # 4. 解析结果
                    # 类别
                    pred_cat_id = torch.argmax(cat_logits, dim=1).item() + 1 
                    en_cat_name = category_names_en.get(pred_cat_id, f"Unknown Category (ID: {pred_cat_id})")
                    zh_cat_name = category_mapping.get(en_cat_name) # 从映射查找中文名
                    display_cat_name = f"{zh_cat_name} ({en_cat_name})" if zh_cat_name else en_cat_name # 拼接显示

                    # 属性 (使用 Sigmoid + 阈值)
                    attr_probs = torch.sigmoid(attr_logits).squeeze(0) 
                    threshold = 0.5 
                    pred_attr_indices = torch.where(attr_probs > threshold)[0].tolist()
                    pred_attr_names_display = []
                    for idx in pred_attr_indices:
                        en_attr_name = attribute_names_en.get(idx, f"Unknown Attr (Idx: {idx})")
                        zh_attr_name = attribute_mapping.get(en_attr_name) # 尝试查找中文名
                        display_attr_name = f"{zh_attr_name} ({en_attr_name})" if zh_attr_name else en_attr_name
                        pred_attr_names_display.append(display_attr_name)

                    # 5. 显示结果
                    with col_results:
                        st.subheader("识别结果:")
                        st.markdown(f"**预测类别:** {display_cat_name}")
                        st.markdown("**预测属性:**")
                        if pred_attr_names_display:
                            rows = (len(pred_attr_names_display) + 2) // 3 
                            for r in range(rows):
                                cols_attr = st.columns(3)
                                for c in range(3):
                                    idx = r * 3 + c
                                    if idx < len(pred_attr_names_display):
                                        with cols_attr[c]:
                                            st.info(pred_attr_names_display[idx])
                        else:
                            st.write("未检测到显著属性。")

                    st.success("识别完成！")

                except Exception as e:
                    st.error(f"识别过程中发生错误: {e}")
                    st.error(traceback.format_exc()) # 显示详细错误 