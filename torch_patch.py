# PyTorch补丁文件 - 在PyTorch被导入前应用此补丁
import sys
import importlib
import types
import os
import warnings

# 禁用文件监视相关环境变量
os.environ["STREAMLIT_SERVER_WATCHDOG_FREQUENCY"] = "86400"  # 24小时
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # 完全禁用文件监视

class SafePathWrapper:
    """安全的路径包装器，防止Streamlit的监视器访问不存在的属性"""
    def __init__(self):
        self._path = []
        
    def __iter__(self):
        """让路径对象可迭代"""
        return iter(self._path)
    
    def __getattr__(self, attr):
        """捕获所有属性访问"""
        if attr == '_path':
            return []
        return []  # 返回空列表而不是引发异常

def torch_patch():
    """修复PyTorch类属性和Streamlit监视器的兼容性问题"""
    print("应用PyTorch补丁...")
    
    # 创建一个空的torch模块
    torch_module = types.ModuleType('torch')
    
    # 添加一个假的classes模块
    classes_module = types.ModuleType('torch.classes')
    
    # 设置安全的__path__属性
    path_wrapper = SafePathWrapper()
    setattr(classes_module, '__path__', path_wrapper)
    
    # 添加一个假的_classes模块
    _classes_module = types.ModuleType('torch._classes')
    
    # 为_classes模块添加安全的__getattr__方法
    def safe_getattr(self, name):
        if name == '__path__' or name == '_path':
            return []
        return None
    setattr(_classes_module, '__getattr__', safe_getattr)
    
    # 将模块添加到torch
    setattr(torch_module, 'classes', classes_module)
    setattr(torch_module, '_classes', _classes_module)
    
    # 将修补后的模块添加到sys.modules
    sys.modules['torch'] = torch_module
    sys.modules['torch.classes'] = classes_module
    sys.modules['torch._classes'] = _classes_module
    
    print("PyTorch补丁已应用")

# 在加载PyTorch之前应用补丁
torch_patch()

# 然后再导入真正的PyTorch
try:
    del sys.modules['torch']
    del sys.modules['torch.classes']
    del sys.modules['torch._classes']
except KeyError:
    pass

# 屏蔽StreamLit的文件监视器
try:
    sys.modules['streamlit.watcher.local_sources_watcher'] = type('', (), {})()
    class DummyWatcher:
        def __init__(self, *args, **kwargs):
            pass
        def start_watching_local_modules(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    sys.modules['streamlit.watcher.local_sources_watcher'].LocalSourcesWatcher = DummyWatcher
except Exception:
    pass

# 忽略与PyTorch相关的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Tried to instantiate class \"__path__._path\".*")
warnings.filterwarnings("ignore", message=".*torch.classes.__path__.*")
warnings.filterwarnings("ignore", message=".*torch._C.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*'_path'.*")
warnings.filterwarnings("ignore", message=".*extract_paths.*") 