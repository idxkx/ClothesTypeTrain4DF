import streamlit as st
import os
import json
import uuid
import time
import pandas as pd
from PIL import Image
import shutil
from datetime import datetime
import sys
import traceback
import torch
from torchvision import transforms

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型定义
try:
    from model import ClothesModel
except ImportError as e:
    st.error(f"错误：无法导入 ClothesModel。请确保 model.py 在项目根目录下。错误: {e}")

# 常量定义
LABEL_DATA_DIR = "../labeled_data"
ANNO_FINE_DIR = "E:/AIModels/DeepFashion/DeepFashion/Category and Attribute Prediction Benchmark/Anno_fine"
CATEGORY_FILE = "list_category_cloth.txt"
ATTRIBUTE_FILE = "list_attr_cloth.txt"
CUSTOM_CATEGORIES_FILE = "../custom_categories.json"
CUSTOM_ATTRIBUTES_FILE = "../custom_attributes.json"
RESULTS_FILE = "../training_results.json"
# 新增：映射文件常量
MAPPING_FILE = "../name_mapping.json"

# 页面配置
st.set_page_config(page_title="数据标注工具", layout="wide")
st.title("🏷️ 服装数据标注工具")
st.markdown("上传新图片，标注类别和属性，扩充数据集以提升模型识别能力")

# 确保数据目录存在
os.makedirs(LABEL_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(LABEL_DATA_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(LABEL_DATA_DIR, "annotations"), exist_ok=True)

# 加载原始类别和属性
def load_original_categories():
    """从原始数据集加载所有服装类别"""
    categories = {}
    try:
        with open(os.path.join(ANNO_FINE_DIR, CATEGORY_FILE), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[2:]):  # 跳过前两行
                parts = line.strip().split()
                if len(parts) >= 1:
                    category_id = i + 1  # 类别ID从1开始
                    category_name = ' '.join(parts[:-1])
                    categories[category_id] = category_name
        return categories
    except Exception as e:
        st.error(f"加载原始类别文件失败: {e}")
        return {}

def load_original_attributes():
    """从原始数据集加载所有服装属性"""
    attributes = {}
    try:
        with open(os.path.join(ANNO_FINE_DIR, ATTRIBUTE_FILE), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[2:]):  # 跳过前两行
                parts = line.strip().split()
                if len(parts) >= 1:
                    attribute_name = ' '.join(parts[:-1])
                    attributes[i] = attribute_name
        return attributes
    except Exception as e:
        st.error(f"加载原始属性文件失败: {e}")
        return {}

# 加载自定义类别和属性
def load_custom_categories():
    """加载用户定义的自定义类别"""
    if os.path.exists(CUSTOM_CATEGORIES_FILE):
        try:
            with open(CUSTOM_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"加载自定义类别失败: {e}，将使用空列表")
    return {}

def load_custom_attributes():
    """加载用户定义的自定义属性"""
    if os.path.exists(CUSTOM_ATTRIBUTES_FILE):
        try:
            with open(CUSTOM_ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"加载自定义属性失败: {e}，将使用空列表")
    return {}

def save_custom_categories(categories):
    """保存自定义类别"""
    try:
        with open(CUSTOM_CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存自定义类别失败: {e}")

def save_custom_attributes(attributes):
    """保存自定义属性"""
    try:
        with open(CUSTOM_ATTRIBUTES_FILE, 'w', encoding='utf-8') as f:
            json.dump(attributes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存自定义属性失败: {e}")

# 初始化数据
if 'original_categories' not in st.session_state:
    st.session_state.original_categories = load_original_categories()
if 'original_attributes' not in st.session_state:
    st.session_state.original_attributes = load_original_attributes()
if 'custom_categories' not in st.session_state:
    st.session_state.custom_categories = load_custom_categories()
if 'custom_attributes' not in st.session_state:
    st.session_state.custom_attributes = load_custom_attributes()

# 标注数据管理
def get_labeled_data():
    """获取已标注的数据列表"""
    data = []
    annotations_dir = os.path.join(LABEL_DATA_DIR, "annotations")
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(annotations_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    annotation = json.load(f)
                    data.append(annotation)
            except Exception as e:
                st.warning(f"读取标注文件 {filename} 失败: {e}")
    return data

def save_annotation(annotation):
    """保存标注数据"""
    annotations_dir = os.path.join(LABEL_DATA_DIR, "annotations")
    file_path = os.path.join(annotations_dir, f"{annotation['id']}.json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存标注数据失败: {e}")
        return False

def save_image(image, image_id):
    """保存上传的图片，自动等比例压缩到适合的分辨率"""
    images_dir = os.path.join(LABEL_DATA_DIR, "images")
    file_path = os.path.join(images_dir, f"{image_id}.jpg")
    try:
        # 获取原始尺寸
        width, height = image.size
        
        # 计算目标尺寸（等比例缩放，优先以宽度为准）
        TARGET_WIDTH = 800
        TARGET_HEIGHT = 1200
        
        # 先以宽度为基准计算
        new_width = TARGET_WIDTH
        new_height = int(height * (TARGET_WIDTH / width))
        
        # 如果高度超过了目标高度，则以高度为基准重新计算
        if new_height > TARGET_HEIGHT:
            new_height = TARGET_HEIGHT
            new_width = int(width * (TARGET_HEIGHT / height))
        
        # 检查是否需要缩放（仅当原图大于目标尺寸时）
        if width > TARGET_WIDTH or height > TARGET_HEIGHT:
            # 等比例缩放图片
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # 保存压缩后的图片，保持适当质量
            resized_image.save(file_path, "JPEG", quality=85)
            st.info(f"图片已自动压缩: {width}x{height} → {new_width}x{new_height}")
        else:
            # 原图尺寸已经合适，直接保存
            image.save(file_path, "JPEG", quality=85)
            st.info(f"图片尺寸适中，未压缩: {width}x{height}")
        
        return True
    except Exception as e:
        st.error(f"保存图片失败: {e}")
        st.error(traceback.format_exc())
        return False

# 加载训练结果
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

# 加载名称映射
def load_name_mapping():
    """加载类别和属性的中英文映射"""
    mapping_path = os.path.join(os.path.dirname(__file__), MAPPING_FILE)
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        # 提供默认空字典，防止文件不包含某个键
        return mapping.get("categories", {}), mapping.get("attributes", {})
    except FileNotFoundError:
        return {}, {}
    except json.JSONDecodeError:
        st.error(f"名称映射文件 {mapping_path} 格式错误。")
        return {}, {}

# 图像预处理 (与验证集相同)
img_size = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize,
])

# 页面功能区域
tabs = st.tabs(["标注新图片", "管理自定义类别", "管理已标注数据", "导出数据", "模型测试"])

# 标注新图片选项卡
with tabs[0]:
    st.header("上传并标注新图片")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传服装图片:",
        type=["jpg", "jpeg", "png"],
        key="label_new_image_upload"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="待标注图片", use_column_width=True)
            
            with col2:
                st.subheader("图片标注")
                
                # 合并原始类别和自定义类别
                all_categories = {}
                all_categories.update(st.session_state.original_categories)
                all_categories.update(st.session_state.custom_categories)
                
                # 制作类别选择列表
                category_options = {}
                # 优先显示自定义类别（如旗袍）
                for cat_id, cat_name in st.session_state.custom_categories.items():
                    display_name = f"{cat_name} (自定义)"
                    category_options[display_name] = cat_id
                # 然后显示原始类别
                for cat_id, cat_name in st.session_state.original_categories.items():
                    category_options[cat_name] = cat_id
                
                # 类别选择
                selected_category = st.selectbox(
                    "选择服装类别:",
                    options=list(all_categories.keys()),
                    key="label_category_select"
                )
                
                selected_category_id = category_options.get(selected_category) if selected_category else None
                
                # 合并原始属性和自定义属性
                all_attributes = {}
                all_attributes.update(st.session_state.original_attributes)
                all_attributes.update(st.session_state.custom_attributes)
                
                # 属性选择
                selected_attributes = {}
                
                # 按类型对属性分组显示（这里简化为3组，实际可根据需要细分）
                attribute_groups = {
                    "款式属性": [],
                    "材质属性": [],
                    "其他属性": []
                }
                
                # 简单分组规则，实际项目可采用更精确的分组
                for attr_id, attr_name in all_attributes.items():
                    if any(keyword in attr_name.lower() for keyword in ["sleeve", "collar", "length", "neck"]):
                        attribute_groups["款式属性"].append((attr_id, attr_name))
                    elif any(keyword in attr_name.lower() for keyword in ["cotton", "fabric", "leather", "material"]):
                        attribute_groups["材质属性"].append((attr_id, attr_name))
                    else:
                        attribute_groups["其他属性"].append((attr_id, attr_name))
                
                # 为自定义属性添加标记
                for attr_id in st.session_state.custom_attributes.keys():
                    for group_name in attribute_groups:
                        for i, (a_id, a_name) in enumerate(attribute_groups[group_name]):
                            if a_id == attr_id:
                                attribute_groups[group_name][i] = (a_id, f"{a_name} (自定义)")
                
                # 使用选项卡显示各组属性
                attr_tabs = st.tabs(list(attribute_groups.keys()))
                for i, (group_name, attrs) in enumerate(attribute_groups.items()):
                    with attr_tabs[i]:
                        cols = st.columns(3)
                        for j, (attr_id, attr_name) in enumerate(attrs):
                            with cols[j % 3]:
                                selected_attributes[attr_id] = st.checkbox(
                                    attr_name,
                                    key=f"label_attr_{attr_id}"
                                )
                
                # 标注备注
                notes = st.text_area(
                    "标注备注:",
                    key="label_notes_input"
                )
                
                # 保存按钮
                if st.button("保存标注", key="label_save_btn"):
                    if selected_category_id:
                        # 生成唯一ID
                        image_id = str(uuid.uuid4())
                        
                        # 获取选中的属性ID
                        selected_attr_ids = [attr_id for attr_id, selected in selected_attributes.items() if selected]
                        
                        # 创建标注数据
                        annotation = {
                            "id": image_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "category_id": selected_category_id,
                            "category_name": next((name for name, id in category_options.items() if id == selected_category_id), None),
                            "attributes": selected_attr_ids,
                            "notes": notes
                        }
                        
                        # 保存图片和标注
                        if save_image(image, image_id) and save_annotation(annotation):
                            st.success("标注数据保存成功!")
                            # 清空上传
                            st.session_state.label_new_image_upload = None
                            # 重新加载页面
                            st.rerun()
                    else:
                        st.error("请选择服装类别!")
        except Exception as e:
            st.error(f"处理图片失败: {e}")
            st.error(traceback.format_exc())

# 管理自定义类别选项卡
with tabs[1]:
    st.header("管理自定义类别与属性")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("自定义服装类别")
        
        # 显示现有自定义类别
        if st.session_state.custom_categories:
            st.write("当前自定义类别:")
            custom_cat_df = pd.DataFrame([
                {"ID": cat_id, "类别名称": cat_name} 
                for cat_id, cat_name in st.session_state.custom_categories.items()
            ])
            st.dataframe(custom_cat_df)
        else:
            st.info("暂无自定义类别，使用下方表单添加")
        
        # 添加新类别
        st.write("添加新类别:")
        # 生成一个新的类别ID，避免与原始类别冲突
        next_id = 1001
        while str(next_id) in st.session_state.custom_categories:
            next_id += 1
            
        new_cat_id = st.text_input(
            "类别ID (建议使用1000以上的值):",
            value=str(next_id),
            key="new_category_id_input"
        )
        new_cat_name = st.text_input(
            "类别名称 (例如: Qipao/旗袍):",
            key="new_category_name_input"
        )
        
        if st.button("添加类别", key="add_category_btn"):
            if new_cat_id and new_cat_name:
                # 检查ID是否已存在
                if new_cat_id in st.session_state.original_categories or new_cat_id in st.session_state.custom_categories:
                    st.error(f"类别ID {new_cat_id} 已存在!")
                else:
                    st.session_state.custom_categories[new_cat_id] = new_cat_name
                    save_custom_categories(st.session_state.custom_categories)
                    st.success(f"类别 '{new_cat_name}' 添加成功!")
                    st.rerun()
            else:
                st.error("请输入有效的类别ID和名称!")
    
    with col2:
        st.subheader("自定义服装属性")
        
        # 显示现有自定义属性
        if st.session_state.custom_attributes:
            st.write("当前自定义属性:")
            custom_attr_df = pd.DataFrame([
                {"ID": attr_id, "属性名称": attr_name} 
                for attr_id, attr_name in st.session_state.custom_attributes.items()
            ])
            st.dataframe(custom_attr_df)
        else:
            st.info("暂无自定义属性，使用下方表单添加")
        
        # 添加新属性
        st.write("添加新属性:")
        # 生成一个新的属性ID，避免与原始属性冲突
        next_attr_id = 1001
        while str(next_attr_id) in st.session_state.custom_attributes:
            next_attr_id += 1
            
        new_attr_id = st.text_input(
            "属性ID (建议使用1000以上的值):",
            value=str(next_attr_id),
            key="new_attribute_id_input"
        )
        new_attr_name = st.text_input(
            "属性名称 (例如: mandarin_collar/立领):",
            key="new_attribute_name_input"
        )
        
        if st.button("添加属性", key="add_attribute_btn"):
            if new_attr_id and new_attr_name:
                # 检查ID是否已存在
                if new_attr_id in st.session_state.original_attributes or new_attr_id in st.session_state.custom_attributes:
                    st.error(f"属性ID {new_attr_id} 已存在!")
                else:
                    st.session_state.custom_attributes[new_attr_id] = new_attr_name
                    save_custom_attributes(st.session_state.custom_attributes)
                    st.success(f"属性 '{new_attr_name}' 添加成功!")
                    st.rerun()
            else:
                st.error("请输入有效的属性ID和名称!")

# 管理已标注数据选项卡
with tabs[2]:
    st.header("管理已标注数据")
    
    # 加载已标注数据
    labeled_data = get_labeled_data()
    
    if not labeled_data:
        st.info("暂无已标注数据，请先在'标注新图片'选项卡中添加标注。")
    else:
        st.write(f"已标注 {len(labeled_data)} 张图片")
        
        # 显示标注数据表格
        labeled_df = pd.DataFrame([
            {
                "ID": item["id"],
                "类别": item["category_name"],
                "属性数量": len(item["attributes"]),
                "标注时间": item["timestamp"],
                "备注": item.get("notes", "")
            }
            for item in labeled_data
        ])
        
        st.dataframe(labeled_df)
        
        # 查看和编辑单个标注
        st.subheader("查看/编辑标注")
        selected_id = st.selectbox(
            "选择要查看的标注:",
            options=[item["id"] for item in labeled_data],
            key="view_label_select"
        )
        
        if selected_id:
            selected_item = next((item for item in labeled_data if item["id"] == selected_id), None)
            if selected_item:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # 显示图片
                    image_path = os.path.join(LABEL_DATA_DIR, "images", f"{selected_id}.jpg")
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption=f"图片 {selected_id}", use_column_width=True)
                    else:
                        st.error(f"图片文件不存在: {image_path}")
                
                with col2:
                    # 显示标注详情
                    st.write("标注详情:")
                    st.write(f"类别: {selected_item['category_name']}")
                    
                    # 显示属性
                    all_attributes = {}
                    all_attributes.update(st.session_state.original_attributes)
                    all_attributes.update(st.session_state.custom_attributes)
                    
                    if selected_item["attributes"]:
                        st.write("属性:")
                        for attr_id in selected_item["attributes"]:
                            attr_name = all_attributes.get(str(attr_id), f"未知属性 (ID: {attr_id})")
                            st.write(f"- {attr_name}")
                    else:
                        st.write("无属性标注")
                    
                    st.write(f"标注时间: {selected_item['timestamp']}")
                    st.write(f"备注: {selected_item.get('notes', '')}")
                    
                    # 删除标注按钮
                    if st.button("删除此标注", key="delete_label_btn"):
                        try:
                            # 删除标注文件
                            annotation_file = os.path.join(LABEL_DATA_DIR, "annotations", f"{selected_id}.json")
                            if os.path.exists(annotation_file):
                                os.remove(annotation_file)
                            
                            # 删除图片文件
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                
                            st.success("标注已删除!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除标注失败: {e}")

# 导出数据选项卡
with tabs[3]:
    st.header("导出标注数据")
    
    # 加载已标注数据
    labeled_data = get_labeled_data()
    
    if not labeled_data:
        st.info("暂无已标注数据可导出。")
    else:
        st.write(f"当前共有 {len(labeled_data)} 条标注数据可导出")
        
        st.subheader("导出选项")
        
        export_format = st.radio(
            "选择导出格式:",
            ["DeepFashion格式", "COCO格式", "YOLO格式"],
            key="export_format_select"
        )
        
        if st.button("导出数据", key="export_data_btn"):
            try:
                export_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_dir = os.path.join(LABEL_DATA_DIR, f"export_{export_time}")
                os.makedirs(export_dir, exist_ok=True)
                
                if export_format == "DeepFashion格式":
                    # 创建DeepFashion格式的导出文件
                    # 1. 创建Anno_fine目录结构
                    anno_fine_dir = os.path.join(export_dir, "Anno_fine")
                    os.makedirs(anno_fine_dir, exist_ok=True)
                    
                    # 2. 创建img目录结构
                    img_dir = os.path.join(export_dir, "Img", "img")
                    os.makedirs(img_dir, exist_ok=True)
                    
                    # 3. 复制图片到img目录，按照DeepFashion的组织方式
                    # 简化处理：所有新标注图片都放在一个自定义类别文件夹下
                    custom_category_id = next(iter(st.session_state.custom_categories.keys()), "1001")
                    category_img_dir = os.path.join(img_dir, custom_category_id)
                    os.makedirs(category_img_dir, exist_ok=True)
                    
                    # 4. 复制和创建基础类别/属性定义文件
                    # 4.1 创建list_category_cloth.txt
                    category_file = os.path.join(anno_fine_dir, "list_category_cloth.txt")
                    with open(category_file, 'w', encoding='utf-8') as f:
                        # 写入文件头
                        categories_count = len(st.session_state.original_categories) + len(st.session_state.custom_categories)
                        f.write(f"{categories_count}\n")
                        f.write("category_name category_type\n")
                        
                        # 写入原始类别
                        for cat_id, cat_name in st.session_state.original_categories.items():
                            # 简化处理：所有类别默认类型为1
                            f.write(f"{cat_name} 1\n")
                        
                        # 写入自定义类别
                        for cat_id, cat_name in st.session_state.custom_categories.items():
                            f.write(f"{cat_name} 1\n")
                    
                    # 4.2 创建list_attr_cloth.txt
                    attr_file = os.path.join(anno_fine_dir, "list_attr_cloth.txt")
                    with open(attr_file, 'w', encoding='utf-8') as f:
                        # 写入文件头
                        attrs_count = len(st.session_state.original_attributes) + len(st.session_state.custom_attributes)
                        f.write(f"{attrs_count}\n")
                        f.write("attribute_name attribute_type\n")
                        
                        # 写入原始属性
                        for attr_id, attr_name in st.session_state.original_attributes.items():
                            # 简化处理：所有属性默认类型为1
                            f.write(f"{attr_name} 1\n")
                        
                        # 写入自定义属性
                        for attr_id, attr_name in st.session_state.custom_attributes.items():
                            f.write(f"{attr_name} 1\n")
                    
                    # 5. 创建train.txt, train_cate.txt, train_attr.txt文件
                    # 5.1 train.txt - 包含图像路径
                    with open(os.path.join(anno_fine_dir, "train.txt"), 'w', encoding='utf-8') as f:
                        f.write(f"{len(labeled_data)}\n")
                        for item in labeled_data:
                            # 复制图片到相应目录
                            src_img = os.path.join(LABEL_DATA_DIR, "images", f"{item['id']}.jpg")
                            dst_img = os.path.join(category_img_dir, f"{item['id']}.jpg")
                            if os.path.exists(src_img):
                                shutil.copy2(src_img, dst_img)
                                # 路径格式：img/类别ID/图像ID.jpg
                                f.write(f"img/{custom_category_id}/{item['id']}.jpg\n")
                    
                    # 5.2 train_cate.txt - 图像与类别的映射
                    with open(os.path.join(anno_fine_dir, "train_cate.txt"), 'w', encoding='utf-8') as f:
                        f.write(f"{len(labeled_data)}\n")
                        for item in labeled_data:
                            # 路径格式 + 类别ID
                            f.write(f"img/{custom_category_id}/{item['id']}.jpg {item['category_id']}\n")
                    
                    # 5.3 train_attr.txt - 图像与属性的映射
                    with open(os.path.join(anno_fine_dir, "train_attr.txt"), 'w', encoding='utf-8') as f:
                        f.write(f"{len(labeled_data)}\n")
                        
                        # 获取所有属性ID（包含原始和自定义）
                        all_attr_ids = set()
                        for key in st.session_state.original_attributes.keys():
                            all_attr_ids.add(str(key))
                        for key in st.session_state.custom_attributes.keys():
                            all_attr_ids.add(str(key))
                        
                        # 排序属性ID
                        sorted_attr_ids = sorted(all_attr_ids, key=lambda x: int(x) if x.isdigit() else float('inf'))
                        
                        # 为每个图片写入属性标签
                        for item in labeled_data:
                            item_attrs = [str(attr_id) for attr_id in item["attributes"]]
                            attr_labels = []
                            for attr_id in sorted_attr_ids:
                                # 1表示有此属性，-1表示无此属性
                                label = 1 if attr_id in item_attrs else -1
                                attr_labels.append(str(label))
                            
                            f.write(f"img/{custom_category_id}/{item['id']}.jpg {' '.join(attr_labels)}\n")
                    
                    # 6. 创建验证集和测试集的文件（与训练集相同，实际使用时可分离）
                    # 6.1 复制train.txt到val.txt和test.txt
                    shutil.copy2(os.path.join(anno_fine_dir, "train.txt"), 
                               os.path.join(anno_fine_dir, "val.txt"))
                    shutil.copy2(os.path.join(anno_fine_dir, "train.txt"), 
                               os.path.join(anno_fine_dir, "test.txt"))
                    
                    # 6.2 复制train_cate.txt到val_cate.txt和test_cate.txt
                    shutil.copy2(os.path.join(anno_fine_dir, "train_cate.txt"), 
                               os.path.join(anno_fine_dir, "val_cate.txt"))
                    shutil.copy2(os.path.join(anno_fine_dir, "train_cate.txt"), 
                               os.path.join(anno_fine_dir, "test_cate.txt"))
                    
                    # 6.3 复制train_attr.txt到val_attr.txt和test_attr.txt
                    shutil.copy2(os.path.join(anno_fine_dir, "train_attr.txt"), 
                               os.path.join(anno_fine_dir, "val_attr.txt"))
                    shutil.copy2(os.path.join(anno_fine_dir, "train_attr.txt"), 
                               os.path.join(anno_fine_dir, "test_attr.txt"))
                    
                    # 7. 创建全局映射文件（可选，但为完整性添加）
                    # 7.1 list_category_img.txt - 所有图片的类别映射
                    with open(os.path.join(anno_fine_dir, "list_category_img.txt"), 'w', encoding='utf-8') as f:
                        f.write(f"{len(labeled_data)}\n")
                        f.write("image_name category_label\n")
                        for item in labeled_data:
                            f.write(f"img/{custom_category_id}/{item['id']}.jpg {item['category_id']}\n")
                    
                    # 7.2 list_attr_img.txt - 所有图片的属性映射
                    with open(os.path.join(anno_fine_dir, "list_attr_img.txt"), 'w', encoding='utf-8') as f:
                        f.write(f"{len(labeled_data)}\n")
                        # 属性总数
                        f.write(f"{len(sorted_attr_ids)}\n")
                        
                        for item in labeled_data:
                            item_attrs = [str(attr_id) for attr_id in item["attributes"]]
                            attr_labels = []
                            for attr_id in sorted_attr_ids:
                                # 1表示有此属性，-1表示无此属性
                                label = 1 if attr_id in item_attrs else -1
                                attr_labels.append(str(label))
                            
                            f.write(f"img/{custom_category_id}/{item['id']}.jpg {' '.join(attr_labels)}\n")
                    
                    st.success(f"数据已成功导出为DeepFashion格式，完全符合Anno_fine目录结构，保存在: {export_dir}")
                
                elif export_format == "COCO格式":
                    # 导出为COCO格式
                    export_file = os.path.join(export_dir, "labeled_data.coco")
                    
                    # 创建COCO格式的标注文件
                    coco_data = {
                        "images": [
                            {
                                "id": item["id"],
                                "file_name": f"images/{item['id']}.jpg",
                                "width": image.width,
                                "height": image.height
                            }
                            for item, image in zip(labeled_data, [Image.open(os.path.join(LABEL_DATA_DIR, "images", f"{item['id']}.jpg")) for item in labeled_data])
                        ],
                        "categories": [
                            {
                                "id": cat_id,
                                "name": cat_name,
                                "supercategory": "clothing"
                            }
                            for cat_id, cat_name in st.session_state.custom_categories.items()
                        ],
                        "annotations": [
                            {
                                "id": item["id"],
                                "image_id": item["id"],
                                "category_id": item["category_id"],
                                "bbox": [0, 0, image.width, image.height],
                                "area": image.width * image.height,
                                "iscrowd": 0
                            }
                            for item, image in zip(labeled_data, [Image.open(os.path.join(LABEL_DATA_DIR, "images", f"{item['id']}.jpg")) for item in labeled_data])
                        ]
                    }
                    
                    # 写入COCO标注文件
                    with open(export_file, 'w', encoding='utf-8') as f:
                        json.dump(coco_data, f, ensure_ascii=False, indent=2)
                    
                    # 复制图片
                    for item in labeled_data:
                        src_img = os.path.join(LABEL_DATA_DIR, "images", f"{item['id']}.jpg")
                        dst_img = os.path.join(export_dir, "images", f"{item['id']}.jpg")
                        if os.path.exists(src_img):
                            os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
                            shutil.copy2(src_img, dst_img)
                    
                    st.success(f"数据已成功导出为COCO格式，保存在: {export_file}")
                
                else:  # YOLO格式
                    # 导出为YOLO格式
                    export_file = os.path.join(export_dir, "labeled_data.yolo")
                    
                    # 创建YOLO格式的标注文件
                    yolo_data = []
                    for item, image in zip(labeled_data, [Image.open(os.path.join(LABEL_DATA_DIR, "images", f"{item['id']}.jpg")) for item in labeled_data]):
                        yolo_data.append(f"{item['category_id']} {item['attributes'][0] / image.width} {item['attributes'][1] / image.height} {(item['attributes'][2] - item['attributes'][0]) / image.width} {(item['attributes'][3] - item['attributes'][1]) / image.height}")
                    
                    # 写入YOLO标注文件
                    with open(export_file, 'w', encoding='utf-8') as f:
                        f.write("\n".join(yolo_data))
                    
                    # 复制图片
                    for item in labeled_data:
                        src_img = os.path.join(LABEL_DATA_DIR, "images", f"{item['id']}.jpg")
                        dst_img = os.path.join(export_dir, "images", f"{item['id']}.jpg")
                        if os.path.exists(src_img):
                            os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
                            shutil.copy2(src_img, dst_img)
                    
                    st.success(f"数据已成功导出为YOLO格式，保存在: {export_file}")
                
                # 显示下载链接
                st.write("请手动复制以下路径访问导出文件:")
                st.code(export_dir)
            
            except Exception as e:
                st.error(f"导出数据失败: {e}")
                st.error(traceback.format_exc())
        
        st.subheader("重新训练模型")
        st.write("导出数据后，您可以使用这些数据重新训练模型，提升对新类别（如旗袍）的识别能力。")
        
        if st.button("跳转到训练页面", key="goto_train_page_btn"):
            # 跳转到训练页面的URL
            js = f"""
            <script>
                window.parent.location.href = "/";
            </script>
            """
            st.components.v1.html(js)

# 新增: 模型测试选项卡
with tabs[4]:
    st.header("使用新生成的模型进行测试")
    
    # 模型选择方式
    model_selection_method = st.radio(
        "选择模型方式:",
        ["从下拉列表选择", "自定义模型路径"],
        key="model_selection_method"
    )
    
    # 从训练历史中获取模型
    if model_selection_method == "从下拉列表选择":
        all_results = load_results()
        # 筛选出成功的训练运行
        successful_runs = [
            r for r in all_results
            if r.get("status") == "已完成" and
               r.get("functional_test_result") == "成功" and
               r.get("best_model_path") and
               os.path.exists(os.path.join(os.path.dirname(__file__), '..', r["best_model_path"]))
        ]

        model_options = {"请选择模型": None}
        if successful_runs:
            for run in sorted(successful_runs, key=lambda x: x.get("end_time", 0), reverse=True):
                option_label = f"{run.get('model_name', '未知模型')} (完成于 {run.get('end_time_str', '未知时间')}, Backbone: {run.get('backbone', '未知')})"
                model_options[option_label] = {
                    "path": os.path.join(os.path.dirname(__file__), '..', run["best_model_path"]), 
                    "backbone": run.get("backbone"),
                    # 使用默认路径作为备选
                    "anno_dir": run.get("anno_dir", ANNO_FINE_DIR) 
                }
        else:
            st.warning("没有找到符合条件的训练好的模型记录。请先在主页面完成至少一次训练或选择自定义模型路径。")

        selected_model = st.selectbox(
            "选择训练好的模型:",
            list(model_options.keys()),
            key="model_selection_dropdown"
        )
        
        selected_model_info = model_options.get(selected_model)
        
        # 如果选择了模型，显示模型信息
        if selected_model_info:
            st.success(f"已选择模型，骨干网络: {selected_model_info.get('backbone', '未知')}")
            model_path = selected_model_info.get("path")
            backbone = selected_model_info.get("backbone")
            anno_dir = selected_model_info.get("anno_dir")
    
    # 自定义模型路径
    else:
        model_dir = st.text_input(
            "输入模型目录路径 (例如：models/MD_RESNET18_5_64_50E04_DEMO):",
            key="custom_model_dir"
        )
        
        if model_dir:
            full_model_dir = os.path.join(os.path.dirname(__file__), '..', model_dir)
            
            if os.path.isdir(full_model_dir):
                # 列出目录中的.pth文件
                model_files = [f for f in os.listdir(full_model_dir) if f.endswith('.pth')]
                
                if model_files:
                    selected_model_file = st.selectbox(
                        "选择模型文件:",
                        model_files,
                        key="custom_model_file"
                    )
                    
                    if selected_model_file:
                        model_path = os.path.join(full_model_dir, selected_model_file)
                        
                        # 选择骨干网络
                        backbone = st.selectbox(
                            "选择骨干网络:",
                            ["resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b3"],
                            key="custom_model_backbone"
                        )
                        
                        # 使用默认的Anno_fine目录
                        anno_dir = ANNO_FINE_DIR
                        
                        st.success(f"已选择模型: {model_path}")
                else:
                    st.error(f"在目录 {full_model_dir} 中没有找到 .pth 模型文件")
                    model_path = None
                    backbone = None
                    anno_dir = None
            else:
                st.error(f"目录 {full_model_dir} 不存在")
                model_path = None
                backbone = None
                anno_dir = None
        else:
            model_path = None
            backbone = None
            anno_dir = None
    
    # 图片上传区域
    uploaded_file = st.file_uploader(
        "上传一张服装图片进行测试:",
        type=["jpg", "jpeg", "png"],
        key="test_image_upload"
    )
    
    # 结果显示区域
    col_img, col_results = st.columns(2)
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            with col_img:
                st.image(image, caption="上传的测试图片", use_column_width=True)
        except Exception as e:
            st.error(f"无法加载图片: {e}")
            uploaded_file = None  # 阻止后续处理
    
    # 测试按钮
    if st.button("🚀 开始测试！", key="start_testing_btn"):
        if not model_path:
            st.error("请先选择一个模型。")
        elif not uploaded_file:
            st.error("请先上传一张图片。")
        else:
            should_proceed = True
            
            # 验证Anno目录
            if not os.path.isdir(anno_dir):
                st.warning(f"指定的Anno_fine目录不存在: {anno_dir}")
                should_proceed = False
            
            if should_proceed:
                # 加载类别和属性映射
                category_mapping, attribute_mapping = load_name_mapping()
                
                # 开始识别流程
                with st.spinner("正在加载模型并进行测试..."):
                    try:
                        # 1. 加载类别和属性名称
                        category_names = {}
                        try:
                            with open(os.path.join(anno_dir, CATEGORY_FILE), 'r') as f:
                                lines = f.readlines()
                                for i, line in enumerate(lines[2:]):
                                    parts = line.strip().split()
                                    if len(parts) >= 1:
                                        category_name = ' '.join(parts[:-1])
                                        category_names[i + 1] = category_name
                        except Exception as e:
                            st.error(f"读取类别文件失败: {e}")
                            category_names = {}
                        
                        attribute_names = {}
                        try:
                            with open(os.path.join(anno_dir, ATTRIBUTE_FILE), 'r') as f:
                                lines = f.readlines()
                                for i, line in enumerate(lines[2:]):
                                    parts = line.strip().split()
                                    if len(parts) >= 1:
                                        attribute_name = ' '.join(parts[:-1])
                                        attribute_names[i] = attribute_name
                        except Exception as e:
                            st.error(f"读取属性文件失败: {e}")
                            attribute_names = {}
                        
                        # 2. 加载模型
                        model = ClothesModel(num_categories=50, backbone=backbone)
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.load_state_dict(torch.load(model_path, map_location=device))
                        model.to(device)
                        model.eval()
                        
                        # 3. 预处理图片
                        img_tensor = eval_transform(image).unsqueeze(0).to(device)
                        
                        # 4. 模型推理
                        with torch.no_grad():
                            cat_logits, attr_logits = model(img_tensor)
                        
                        # 5. 解析结果
                        # 类别 - 获取所有类别的概率分布
                        cat_probs = torch.softmax(cat_logits, dim=1).squeeze(0).cpu().numpy()
                        
                        # 创建类别索引、名称和概率的列表
                        cat_data = []
                        for idx in range(len(cat_probs)):
                            cat_id = idx + 1  # 类别ID从1开始
                            en_cat_name = category_names.get(cat_id, f"Unknown (ID: {cat_id})")
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
                            en_attr_name = attribute_names.get(idx, f"Unknown Attr (Idx: {idx})")
                            zh_attr_name = attribute_mapping.get(en_attr_name)
                            display_attr_name = f"{zh_attr_name} ({en_attr_name})" if zh_attr_name else en_attr_name
                            attr_data.append({
                                'index': idx,
                                'name': display_attr_name, 
                                'probability': prob
                            })
                        
                        # 按概率降序排序
                        attr_data.sort(key=lambda x: x['probability'], reverse=True)
                        
                        # 6. 显示结果
                        with col_results:
                            st.subheader("测试结果:")
                            
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
                            
                            # 显示属性预测结果
                            st.markdown("**预测属性及概率:**")
                            
                            # 阈值选择
                            threshold = st.slider(
                                "属性置信度阈值",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.3,
                                step=0.05,
                                key="test_confidence_threshold"
                            )
                            
                            # 筛选属性
                            filtered_attrs = [attr for attr in attr_data if attr['probability'] >= threshold]
                            
                            if filtered_attrs:
                                # 分组显示属性
                                # 高概率组 (>0.7)
                                high_prob_attrs = [attr for attr in filtered_attrs if attr['probability'] > 0.7]
                                if high_prob_attrs:
                                    st.markdown("##### 高置信度属性 (>70%)")
                                    cols_attr = st.columns(3)
                                    for i, attr in enumerate(high_prob_attrs):
                                        with cols_attr[i % 3]:
                                            st.success(f"{attr['name']} ({attr['probability']*100:.1f}%)")
                                
                                # 中概率组 (0.5-0.7)
                                medium_prob_attrs = [attr for attr in filtered_attrs if 0.5 <= attr['probability'] <= 0.7]
                                if medium_prob_attrs:
                                    st.markdown("##### 中等置信度属性 (50%-70%)")
                                    cols_attr = st.columns(3)
                                    for i, attr in enumerate(medium_prob_attrs):
                                        with cols_attr[i % 3]:
                                            st.info(f"{attr['name']} ({attr['probability']*100:.1f}%)")
                                
                                # 低概率组 (阈值-0.5)
                                low_prob_attrs = [attr for attr in filtered_attrs if attr['probability'] < 0.5]
                                if low_prob_attrs:
                                    st.markdown("##### 低置信度属性 (<50%)")
                                    cols_attr = st.columns(3)
                                    for i, attr in enumerate(low_prob_attrs):
                                        with cols_attr[i % 3]:
                                            st.warning(f"{attr['name']} ({attr['probability']*100:.1f}%)")
                            else:
                                st.write("在当前阈值下未检测到显著属性。")
                            
                            st.success("测试完成！")
                    
                    except Exception as e:
                        st.error(f"测试过程中发生错误: {e}")
                        st.error(traceback.format_exc()) 