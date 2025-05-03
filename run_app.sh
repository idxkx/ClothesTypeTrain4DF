#!/bin/bash
echo "正在启动喵搭服装识别训练场..."
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_WATCHDOG_FREQUENCY=86400
python3 simple_launcher.py 