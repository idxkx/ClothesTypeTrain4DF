import streamlit as st
import os
import json
from datetime import datetime

from components.report_generator import generate_metadata_for_model, batch_generate_metadata

def create_metadata_form(model_name, model_data):
    """为模型创建元数据表单"""
    backbone = model_data.get("backbone", "unknown")
    
    st.subheader("基本信息")
    version = st.text_input("版本", "1.0.0", key="metadata_version")
    description = st.text_input("描述", f"基于{backbone}的服装分类模型", key="metadata_description")
    trained_by = st.text_input("训练者", "喵搭服装识别训练平台", key="metadata_trained_by")
    date_created = st.date_input("创建日期", datetime.now(), key="metadata_date").strftime("%Y-%m-%d")
    
    st.subheader("模型配置")
    st.markdown(f"架构: **{backbone}**")
    input_shape = st.text_input("输入形状 (逗号分隔，如3,224,224)", "3,224,224", key="metadata_input_shape")
    
    st.subheader("类别与特征")
    st.markdown("### 类别名称")
    st.markdown("每行输入一个类别名称，如为空将使用默认名称")
    default_class_names = (
        "T恤\n衬衫\n卫衣\n毛衣\n西装\n夹克\n羽绒服\n风衣\n"
        "牛仔裤\n休闲裤\n西裤\n短裤\n运动裤\n连衣裙\n半身裙\n"
        "旗袍\n礼服\n运动鞋\n皮鞋\n高跟鞋\n靴子\n凉鞋\n拖鞋\n"
        "帽子\n围巾\n领带\n手套\n袜子\n腰带\n眼镜\n手表\n"
        "项链\n手链\n耳环\n戒指\n包包\n背包\n手提包\n钱包\n行李箱"
    )
    class_names_text = st.text_area("类别名称列表", default_class_names, height=200, key="metadata_class_names")
    
    st.markdown("### 特征名称")
    st.markdown("每行输入一个特征名称，如为空将使用默认名称")
    default_feature_names = (
        "颜色\n材质\n样式\n花纹\n季节\n正式度\n领型\n袖长\n"
        "长度\n裤型\n鞋型\n高度\n闭合方式"
    )
    feature_names_text = st.text_area("特征名称列表", default_feature_names, height=150, key="metadata_feature_names")
    
    # 返回元数据输入
    return {
        "version": version,
        "description": description,
        "trained_by": trained_by,
        "date_created": date_created,
        "input_shape": input_shape,
        "class_names": class_names_text,
        "feature_names": feature_names_text
    }

def create_metadata_file(model_name, model_data, metadata_input):
    """创建元数据文件"""
    try:
        # 检查模型文件是否存在
        model_path = model_data.get("best_model_path", "")
        if not model_path or not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}"
        
        # 解析输入形状
        input_shape_str = metadata_input.get("input_shape", "3,224,224")
        try:
            input_shape = [int(x.strip()) for x in input_shape_str.split(",")]
        except ValueError:
            return False, f"输入形状格式错误: {input_shape_str}，应为逗号分隔的整数，如 3,224,224"
        
        # 解析类别和特征名称
        class_names_text = metadata_input.get("class_names", "")
        class_names = [name.strip() for name in class_names_text.split("\n") if name.strip()]
        
        feature_names_text = metadata_input.get("feature_names", "")
        feature_names = [name.strip() for name in feature_names_text.split("\n") if name.strip()]
        
        # 构建元数据
        metadata = {
            "model_name": model_name,
            "version": metadata_input.get("version", "1.0.0"),
            "description": metadata_input.get("description", ""),
            "architecture": model_data.get("backbone", "unknown"),
            "input_shape": input_shape,
            "framework": "PyTorch",
            "date_created": metadata_input.get("date_created", datetime.now().strftime("%Y-%m-%d")),
            "trained_by": metadata_input.get("trained_by", ""),
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
        
        return True, f"元数据已成功保存到: {metadata_file}"
    except Exception as e:
        return False, f"创建元数据时出错: {e}"

def display_metadata_viewer():
    """显示元数据查看界面"""
    from utils.file_utils import load_results
    
    st.markdown("""
    此区域可以查看已训练模型的元数据文件，包含模型架构、类别名称、特征名称等信息。
    """)
    
    # 获取模型列表
    all_results = load_results()
    model_names = [r.get("model_name", "未命名模型") for r in all_results]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "选择要查看元数据的模型",
            options=model_names,
            index=0 if model_names else None,
            key="metadata_model_select",
            help="选择一个模型来查看其元数据内容"
        )
    
    with col2:
        if st.button("查看元数据", key="view_metadata_btn"):
            if not selected_model:
                st.info("请先选择一个模型")
                return
                
            # 查找选定模型的记录
            model_results = [r for r in all_results if r.get("model_name") == selected_model]
            if not model_results:
                st.warning(f"⚠️ 找不到 {selected_model} 的训练记录")
                return
                
            model_result = model_results[0]
            model_path = model_result.get("best_model_path", "")
            
            if not model_path or not os.path.exists(model_path):
                st.warning(f"⚠️ 找不到 {selected_model} 的模型文件")
                return
                
            model_dir = os.path.dirname(model_path)
            metadata_file = os.path.join(model_dir, f"{selected_model}_metadata.json")
            
            if not os.path.exists(metadata_file):
                st.warning(f"⚠️ 未找到 {selected_model} 的元数据文件 ({metadata_file})")
                return
                
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                st.json(metadata)
                st.success(f"✅ 已成功加载 {selected_model} 的元数据")
            except Exception as e:
                st.error(f"读取元数据时发生错误: {e}")

def display_metadata_creator():
    """显示元数据创建界面"""
    from utils.file_utils import load_results
    
    st.markdown("""
    此功能允许为已训练的模型手动创建元数据文件，适用于元数据丢失或未自动生成的情况。
    """)
    
    # 获取模型列表
    all_results = load_results()
    model_names = [r.get("model_name", "未命名模型") for r in all_results]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        metadata_model = st.selectbox(
            "选择要创建元数据的模型",
            options=model_names,
            index=0 if model_names else None,
            key="create_metadata_model_select",
            help="选择一个模型来创建或重新生成其元数据文件"
        )
    
    if not metadata_model:
        st.info("请先选择一个模型")
        return
        
    # 查找选定模型的记录
    model_results = [r for r in all_results if r.get("model_name") == metadata_model]
    if not model_results:
        st.warning(f"找不到模型 {metadata_model} 的训练记录")
        return
        
    model_result = model_results[0]
    
    # 显示元数据表单
    metadata_input = create_metadata_form(metadata_model, model_result)
    
    if st.button("创建元数据文件", key="create_metadata_file_btn"):
        # 调用元数据创建函数
        success, message = create_metadata_file(metadata_model, model_result, metadata_input)
        if success:
            st.success(message)
        else:
            st.error(message)
            
        # 刷新历史记录显示
        from components.history_viewer import display_history 