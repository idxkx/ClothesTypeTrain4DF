# 创建一个包装脚本，禁用Streamlit的文件监视系统
import os
import sys
import importlib.util

# 关键设置：禁用Streamlit的文件监视器
os.environ["STREAMLIT_SERVER_WATCHDOG_FREQUENCY"] = "86400"  # 设置为24小时
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 应用PyTorch补丁
try:
    import torch_patch
except ImportError:
    print("警告：找不到torch_patch模块，PyTorch可能会有兼容性问题")

# 导入必要的库并应用补丁
try:
    # 处理asyncio事件循环问题
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

    # 安全导入torch
    try:
        import torch
        if hasattr(torch, 'classes') and not hasattr(torch.classes, '__path__'):
            class ModulePathPatch:
                def __init__(self):
                    self._path = []
                def __iter__(self):
                    return iter(self._path)
                def __getattr__(self, attr):
                    if attr == '_path':
                        return []
                    return []
            torch.classes.__path__ = ModulePathPatch()
        print("已应用运行时torch类补丁")
    except ImportError:
        print("警告：无法导入torch")
except Exception as e:
    print(f"警告：初始化补丁时出错: {e}")

# 运行主应用
if __name__ == "__main__":
    # 清晰显示启动信息
    main_ui_path = os.path.join(current_dir, "main_ui.py")
    print(f"正在启动Streamlit应用: {main_ui_path}")
    
    # 检查文件是否存在
    if not os.path.exists(main_ui_path):
        print(f"错误：找不到主UI文件: {main_ui_path}")
        sys.exit(1)
    
    try:
        # 直接导入main_ui模块并执行main函数
        spec = importlib.util.spec_from_file_location("main_ui", main_ui_path)
        main_ui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_ui)
        
        # 检查模块是否有main函数
        if hasattr(main_ui, 'main'):
            print("正在执行main_ui.main()...")
            main_ui.main()
        else:
            print("错误：main_ui.py中找不到main函数")
            print("正在尝试使用streamlit直接启动...")
            # 作为备选方案，尝试使用streamlit命令
            import streamlit.web.bootstrap as bootstrap
            bootstrap.run(main_ui_path, "", [], {"server.port": 8502, "server.headless": True, "server.fileWatcherType": "none"})
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc() 