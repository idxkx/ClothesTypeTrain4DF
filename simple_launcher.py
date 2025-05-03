#!/usr/bin/env python
"""
简化版启动器 - 解决PyTorch和Streamlit的兼容性问题
"""
import os
import sys
import warnings
import traceback

# 禁用Streamlit的文件监视
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_WATCHDOG_FREQUENCY"] = "86400"

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.__path__.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*instantiate class.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def fix_torch_classes():
    """修复torch.classes.__path__问题"""
    try:
        import torch
        if hasattr(torch, 'classes') and not hasattr(torch.classes, '__path__'):
            class ModulePathPatch:
                def __init__(self):
                    self._path = []
                def __iter__(self):
                    return iter(self._path)
                def __getattr__(self, attr):
                    return []
            
            torch.classes.__path__ = ModulePathPatch()
            print("已应用PyTorch补丁")
            return True
    except ImportError:
        print("警告: 无法导入PyTorch")
    except Exception as e:
        print(f"应用PyTorch补丁时出错: {e}")
    
    return False

def init_asyncio():
    """初始化asyncio事件循环"""
    try:
        import asyncio
        import nest_asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        nest_asyncio.apply()
        print("已初始化asyncio事件循环")
        return True
    except Exception as e:
        print(f"初始化asyncio事件循环时出错: {e}")
    
    return False

def main():
    """主函数"""
    print("正在启动喵搭服装识别训练场...")
    
    # 应用PyTorch补丁
    fix_torch_classes()
    
    # 初始化asyncio事件循环
    init_asyncio()
    
    try:
        import streamlit.cli
        
        # 设置启动参数
        sys.argv = ["streamlit", "run", 
                   os.path.join(current_dir, "main_ui.py"),
                   "--server.headless=true",
                   "--server.port=8502",
                   "--server.fileWatcherType=none"]
        
        # 启动应用
        print(f"正在启动Streamlit应用: {sys.argv[2]}")
        streamlit.cli.main()
    except Exception as e:
        print(f"启动应用失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 