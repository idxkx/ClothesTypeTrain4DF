import streamlit as st
import pandas as pd

# --- GPU 监控依赖 ---
nvml_available = False
pynvml = None

def initialize_gpu():
    """初始化GPU监控，并返回GPU信息"""
    global nvml_available, pynvml
    
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml_available = True
        
        # 获取GPU名称
        gpu_names = {}
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    gpu_names[i] = name
                except Exception:
                    gpu_names[i] = f"GPU {i}"
        except Exception:
            pass
            
        return nvml_available, pynvml, gpu_names
    except ImportError:
        st.warning("未安装GPU监控依赖。如需监控GPU，请安装: pip install nvidia-ml-py")
        return False, None, {}
    except Exception as e:
        st.warning(f"初始化GPU监控失败: {e}")
        return False, None, {}

def get_gpu_count():
    """获取可用GPU数量"""
    if not nvml_available or not pynvml:
        return 0
    try:
        return pynvml.nvmlDeviceGetCount()
    except Exception:
        return 0

def get_gpu_name(gpu_index):
    """获取指定GPU的名称"""
    if not nvml_available or not pynvml:
        return f"GPU {gpu_index}"
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        return name
    except Exception:
        return f"GPU {gpu_index}"

def update_gpu_info(gpu_index, placeholder, chart_placeholders):
    """更新GPU监控信息并记录指标历史"""
    if not nvml_available or not pynvml or gpu_index is None:
        placeholder.info("GPU监控不可用")
        return
        
    gpu_util_chart_placeholder, gpu_mem_chart_placeholder = chart_placeholders
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        try:
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 瓦特
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0  # 瓦特
            power_info = f"- **功耗:** {power_usage:.1f} W / {power_limit:.1f} W"
        except Exception:
            power_info = "- **功耗:** 不可用"

        # 获取GPU名称
        gpu_name = get_gpu_name(gpu_index)
        
        # 格式化GPU信息
        gpu_info_str = (
            f"**{gpu_name}:**\n"
            f"- **温度:** {temp}°C\n"
            f"- **使用率:** {util.gpu}%\n"
            f"- **显存:** {mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB ({mem.used * 100 / mem.total:.1f}%)\n"
            f"{power_info}"
        )
        placeholder.markdown(gpu_info_str)

        # 记录和绘制GPU指标历史
        if 'gpu_metrics_history' not in st.session_state:
            st.session_state.gpu_metrics_history = []
            
        if 'gpu_poll_step' not in st.session_state:
            st.session_state.gpu_poll_step = 0
            
        current_step = st.session_state.gpu_poll_step
        memory_util_percent = mem.used * 100 / mem.total if mem.total > 0 else 0
        
        st.session_state.gpu_metrics_history.append({
            'step': current_step,
            'GPU Utilization (%)': util.gpu,
            'Memory Utilization (%)': memory_util_percent
        })
        st.session_state.gpu_poll_step += 1
        
        # 创建DataFrame并绘图
        gpu_history_df = pd.DataFrame(st.session_state.gpu_metrics_history)
        if len(gpu_history_df) > 1:  # 需要至少两个点才能画线
            try:
                gpu_util_chart_placeholder.line_chart(gpu_history_df.set_index('step')['GPU Utilization (%)'])
                gpu_mem_chart_placeholder.line_chart(gpu_history_df.set_index('step')['Memory Utilization (%)'])
            except Exception as chart_e:
                if 'chart_error_logged' not in st.session_state or not st.session_state.chart_error_logged:
                    placeholder.warning(f"绘制GPU图表时出错: {chart_e}")
                    st.session_state.chart_error_logged = True

    except Exception as e:
        if 'gpu_error_logged' not in st.session_state or not st.session_state.gpu_error_logged:
            placeholder.warning(f"获取GPU {gpu_index} 信息失败: {e}")
            st.session_state.gpu_error_logged = True

def shutdown_gpu():
    """关闭GPU监控"""
    if nvml_available and pynvml:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass 