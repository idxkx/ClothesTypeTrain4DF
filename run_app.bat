@echo off
echo 正在启动喵搭服装识别训练场...
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set STREAMLIT_SERVER_WATCHDOG_FREQUENCY=86400
python simple_launcher.py
pause 