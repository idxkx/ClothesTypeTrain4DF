import math
from datetime import datetime, timedelta

def format_time_delta(seconds):
    """将秒数格式化为 HH:MM:SS"""
    if seconds is None or not math.isfinite(seconds):
        return "N/A"
    delta = timedelta(seconds=int(seconds))
    return str(delta)

def format_datetime(dt=None, format_str='%Y-%m-%d %H:%M:%S'):
    """格式化日期时间"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)

def get_current_datetime():
    """获取当前日期时间"""
    return datetime.now()

def calculate_eta(elapsed_time, completed, total):
    """计算预计完成时间
    
    参数:
        elapsed_time: 已经过时间(秒)
        completed: 已完成的任务数量
        total: 总任务数量
        
    返回:
        (预计剩余时间(秒), 预计完成时间(datetime))
    """
    if completed <= 0:
        return None, None
        
    # 计算每单位任务平均时间
    avg_time_per_unit = elapsed_time / completed
    
    # 计算剩余任务
    remaining = total - completed
    
    # 计算预计剩余时间
    eta_seconds = avg_time_per_unit * remaining
    
    # 计算预计完成时间点
    eta_datetime = datetime.now() + timedelta(seconds=eta_seconds)
    
    return eta_seconds, eta_datetime 