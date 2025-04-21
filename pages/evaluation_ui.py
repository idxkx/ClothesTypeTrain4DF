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

# 添加默认Anno_fine路径常量
DEFAULT_ANNO_DIR = r"E:\AIModels\DeepFashion\DeepFashion\Category and Attribute Prediction Benchmark\Anno_fine"

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

# 设置初始状态，默认为下拉列表选择模式
if "using_dropdown_selection" not in st.session_state:
    st.session_state.using_dropdown_selection = True

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
            # 使用默认路径作为备选
            "anno_dir": run.get("anno_dir", DEFAULT_ANNO_DIR) 
        }
else:
    st.warning("没有找到符合条件的训练好的模型记录。请先在主页面完成模型训练。")

# 简化UI，只使用下拉列表选择模型
st.selectbox(
    "选择一个训练好的模型:",
    list(model_options.keys()),
    key="selected_model_dropdown"
)

selected_model_info = model_options.get(st.session_state.selected_model_dropdown)

# 如果选择了模型，显示模型信息
if selected_model_info:
    st.success(f"已选择模型，骨干网络: {selected_model_info.get('backbone', '未知')}")

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
    # 简化逻辑，只处理下拉列表选择的情况
    if not selected_model_info:
        st.error("请先从下拉列表中选择一个模型。")
    elif not uploaded_file:
        st.error("请先上传一张图片。")
    else:
        # 获取模型信息
        path_to_use = selected_model_info.get("path")
        backbone_to_use = selected_model_info.get("backbone")
        final_anno_dir = selected_model_info.get("anno_dir")
        
        # 添加调试信息
        st.write(f"模型路径: {path_to_use}")
        st.write(f"骨干网络: {backbone_to_use}")
        st.write(f"Anno目录: {final_anno_dir}")
        st.write(f"目录存在: {os.path.isdir(final_anno_dir) if final_anno_dir else False}")

        # 使用布尔标志控制流程
        should_proceed = True
        
        # 检查必要信息是否完整
        if not path_to_use or not backbone_to_use:
            st.error("错误：选择的模型信息不完整 (路径或Backbone)。")
            should_proceed = False
        elif not final_anno_dir:
            st.error(f"错误：模型记录中没有Anno_fine目录路径。")
            should_proceed = False
        elif not os.path.isdir(final_anno_dir):
            # 如果目录不存在，尝试使用默认目录
            st.warning(f"指定的Anno_fine目录不存在: {final_anno_dir}")
            if os.path.isdir(DEFAULT_ANNO_DIR):
                st.info(f"使用默认Anno_fine目录: {DEFAULT_ANNO_DIR}")
                final_anno_dir = DEFAULT_ANNO_DIR
            else:
                st.error(f"默认Anno_fine目录也不存在: {DEFAULT_ANNO_DIR}")
                should_proceed = False
        
        # 只有当所有条件都满足时才继续处理
        if should_proceed:
            # 所有信息都完整，开始处理
            model_path = path_to_use
            backbone = backbone_to_use
            num_categories = 50 
            num_attributes = 26 

            # 加载类别和属性名称
            category_names_en = load_category_names(final_anno_dir)
            attribute_names_en = load_attribute_names(final_anno_dir)

            if category_names_en is None or attribute_names_en is None:
                st.error("无法加载类别或属性名称，无法继续识别。")
            else:
                # 开始识别流程
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
                        # 类别 - 获取所有类别的概率分布
                        cat_probs = torch.softmax(cat_logits, dim=1).squeeze(0).cpu().numpy()
                        
                        # 创建类别索引、名称和概率的列表
                        cat_data = []
                        for idx in range(len(cat_probs)):
                            cat_id = idx + 1  # 类别ID从1开始
                            en_cat_name = category_names_en.get(cat_id, f"Unknown (ID: {cat_id})")
                            zh_cat_name = category_mapping.get(en_cat_name)
                            display_cat_name = f"{zh_cat_name} ({en_cat_name})" if zh_cat_name else en_cat_name
                            cat_data.append({
                                'index': cat_id,
                                'name': display_cat_name,
                                'probability': cat_probs[idx]
                            })
                        
                        # 按概率降序排序
                        cat_data.sort(key=lambda x: x['probability'], reverse=True)
                        
                        # 获取概率最高的类别
                        top_category = cat_data[0]
                        
                        # 属性 (使用 Sigmoid 获取所有属性的概率)
                        attr_probs = torch.sigmoid(attr_logits).squeeze(0).cpu().numpy()
                        
                        # 创建属性索引、名称和概率的列表
                        attr_data = []
                        for idx, prob in enumerate(attr_probs):
                            en_attr_name = attribute_names_en.get(idx, f"Unknown Attr (Idx: {idx})")
                            zh_attr_name = attribute_mapping.get(en_attr_name)
                            display_attr_name = f"{zh_attr_name} ({en_attr_name})" if zh_attr_name else en_attr_name
                            attr_data.append({
                                'index': idx,
                                'name': display_attr_name, 
                                'probability': prob
                            })
                        
                        # 按概率降序排序
                        attr_data.sort(key=lambda x: x['probability'], reverse=True)
                        
                        # 5. 显示结果
                        with col_results:
                            st.subheader("识别结果:")
                            
                            # 显示类别预测结果
                            st.markdown("**预测类别及概率:**")
                            # 显示前3个最可能的类别
                            cols_cat = st.columns(3)
                            for i, cat in enumerate(cat_data[:3]):
                                with cols_cat[i]:
                                    if i == 0:  # 最高概率用绿色
                                        st.success(f"{cat['name']} ({cat['probability']*100:.1f}%)")
                                    else:  # 其他候选用蓝色
                                        st.info(f"{cat['name']} ({cat['probability']*100:.1f}%)")
                            
                            # 类别详情折叠面板
                            with st.expander("查看所有类别概率详情"):
                                # 显示前10个最可能的类别
                                st.markdown("##### 前10个最可能的类别:")
                                cat_top10_df = pd.DataFrame(cat_data[:10])
                                cat_top10_df.columns = ["ID", "类别名称", "概率"]
                                cat_top10_df["概率"] = cat_top10_df["概率"].apply(lambda x: f"{x*100:.2f}%")
                                st.dataframe(cat_top10_df)
                                
                                # 显示所有类别的概率分布图
                                st.markdown("##### 类别概率分布:")
                                if len(cat_data) > 10:
                                    fig_data = pd.DataFrame({
                                        '类别': [d['name'].split(' ')[0] for d in cat_data[:10]] + ['其他'],
                                        '概率': [d['probability'] for d in cat_data[:10]] + [sum(d['probability'] for d in cat_data[10:])]
                                    })
                                else:
                                    fig_data = pd.DataFrame({
                                        '类别': [d['name'].split(' ')[0] for d in cat_data],
                                        '概率': [d['probability'] for d in cat_data]
                                    })
                                st.bar_chart(fig_data.set_index('类别'))
                            
                            # 显示属性预测结果
                            st.markdown("**预测属性及概率:**")
                            
                            # 计算显示多少列
                            num_columns = 3  # 默认3列显示
                            
                            # 根据阈值筛选属性（默认0.5，但这里显示所有）
                            # st.slider可以让用户调整筛选阈值
                            threshold = st.slider("属性置信度阈值", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
                            filtered_attrs = [attr for attr in attr_data if attr['probability'] >= threshold]
                            
                            if filtered_attrs:
                                # 按概率分组显示属性
                                # 高概率组 (>0.7)
                                high_prob_attrs = [attr for attr in filtered_attrs if attr['probability'] > 0.7]
                                if high_prob_attrs:
                                    st.markdown("##### 高置信度属性 (>70%)")
                                    rows = (len(high_prob_attrs) + num_columns - 1) // num_columns 
                                    for r in range(rows):
                                        cols_attr = st.columns(num_columns)
                                        for c in range(num_columns):
                                            idx = r * num_columns + c
                                            if idx < len(high_prob_attrs):
                                                attr = high_prob_attrs[idx]
                                                with cols_attr[c]:
                                                    st.success(f"{attr['name']} ({attr['probability']*100:.1f}%)")
                                
                                # 中概率组 (0.5-0.7)
                                medium_prob_attrs = [attr for attr in filtered_attrs if 0.5 <= attr['probability'] <= 0.7]
                                if medium_prob_attrs:
                                    st.markdown("##### 中等置信度属性 (50%-70%)")
                                    rows = (len(medium_prob_attrs) + num_columns - 1) // num_columns 
                                    for r in range(rows):
                                        cols_attr = st.columns(num_columns)
                                        for c in range(num_columns):
                                            idx = r * num_columns + c
                                            if idx < len(medium_prob_attrs):
                                                attr = medium_prob_attrs[idx]
                                                with cols_attr[c]:
                                                    st.info(f"{attr['name']} ({attr['probability']*100:.1f}%)")
                                
                                # 低概率组 (阈值-0.5)
                                low_prob_attrs = [attr for attr in filtered_attrs if attr['probability'] < 0.5]
                                if low_prob_attrs:
                                    st.markdown("##### 低置信度属性 (<50%)")
                                    rows = (len(low_prob_attrs) + num_columns - 1) // num_columns 
                                    for r in range(rows):
                                        cols_attr = st.columns(num_columns)
                                        for c in range(num_columns):
                                            idx = r * num_columns + c
                                            if idx < len(low_prob_attrs):
                                                attr = low_prob_attrs[idx]
                                                with cols_attr[c]:
                                                    st.warning(f"{attr['name']} ({attr['probability']*100:.1f}%)")
                            else:
                                st.write("在当前阈值下未检测到显著属性。")
                            
                            # 显示所有属性的表格视图（可折叠）
                            with st.expander("查看所有属性概率详情"):
                                attr_df = pd.DataFrame(attr_data)
                                attr_df.columns = ["索引", "属性名称", "概率"]
                                attr_df["概率"] = attr_df["概率"].apply(lambda x: f"{x*100:.1f}%")
                                st.dataframe(attr_df)

                        st.success("识别完成！")

                    except Exception as e:
                        st.error(f"识别过程中发生错误: {e}")
                        
                        st.error(traceback.format_exc()) # 显示详细错误 