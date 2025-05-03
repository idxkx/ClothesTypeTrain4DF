# 安全启动脚本 - 应用所有补丁并启动应用
import os
import sys
import importlib.util

# 设置环境变量 - 禁用Streamlit文件监视器
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_WATCHDOG_FREQUENCY"] = "86400"  # 24小时
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # 彻底禁用文件监视器
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"
os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

# 屏蔽Streamlit的模块监视
sys.modules['streamlit.watcher.local_sources_watcher'] = type('', (), {})()
setattr(sys.modules['streamlit.watcher.local_sources_watcher'], 'LocalSourcesWatcher', type('', (), {'__init__': lambda *args, **kwargs: None, 'start_watching_local_modules': lambda *args, **kwargs: None}))

# 确保当前目录在sys.path中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # 先加载补丁
    import torch_patch
    
    # 再导入真正的PyTorch
    import torch
    
    # 更深入地修复torch.classes.__path__问题
    if hasattr(torch, 'classes'):
        if not hasattr(torch.classes, '__path__'):
            class ModulePathPatch:
                def __init__(self):
                    self._path = []
                def __iter__(self):
                    return iter(self._path)
                def __getattr__(self, attr):
                    if attr == '_path':
                        return []
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
            
            # 应用补丁
            torch.classes.__path__ = ModulePathPatch()
        
        # 屏蔽torch._classes模块中的问题函数
        if hasattr(torch, '_classes') and hasattr(torch._classes, '__getattr__'):
            # 保存原始__getattr__
            original_getattr = torch._classes.__getattr__
            
            # 定义新的__getattr__
            def safe_getattr(self, attr):
                if attr == '__path__' or attr == '_path':
                    return []
                try:
                    return original_getattr(self, attr)
                except RuntimeError as e:
                    if "Tried to instantiate class '__path__._path'" in str(e):
                        return []
                    raise
            
            # 应用补丁
            torch._classes.__getattr__ = safe_getattr
    
    print("已成功应用PyTorch补丁到已加载的torch模块")
    
    # 处理asyncio事件循环
    import asyncio
    import nest_asyncio
    
    # 初始化asyncio事件循环
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # 应用nest_asyncio补丁
    nest_asyncio.apply()
    
    # 直接导入并运行main_ui.py
    if __name__ == "__main__":
        # 确保main_ui.py的完整路径
        main_ui_path = os.path.join(current_dir, "main_ui.py")
        print(f"直接运行主UI文件: {main_ui_path}")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(main_ui_path):
                raise FileNotFoundError(f"找不到主UI文件: {main_ui_path}")
            
            # 导入main_ui模块并执行
            spec = importlib.util.spec_from_file_location("main_ui", main_ui_path)
            main_ui = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_ui)
            
            # 如果main_ui有main函数，执行它
            if hasattr(main_ui, 'main'):
                main_ui.main()
        except Exception as e:
            print(f"启动失败: {e}")
            import traceback
            traceback.print_exc()
except Exception as e:
    print(f"初始化失败: {e}")
    import traceback
    traceback.print_exc() 