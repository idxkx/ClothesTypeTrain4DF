# 服装类别与属性识别演示项目 (ClothesTypeDemo)

## 项目目标

本项目旨在利用 DeepFashion 数据集的 "Category and Attribute Prediction Benchmark"，构建一个用户友好的平台，用于训练和测试能够识别服装类别和属性的 AI 模型。项目特别关注为不熟悉 AI 或编程的用户提供直观、易懂的操作体验。

## 项目结构

本项目主要包含两个核心模块：

1.  **模型预训练模块:** 提供可视化的界面来监控和调整模型训练过程。
2.  **模型效果测试模块:** (待后续设计) 用于上传图片，测试已训练模型的识别效果。

## 主要技术栈

*   **语言:** Python
*   **深度学习框架:** PyTorch (或 TensorFlow)
*   **用户界面:** Streamlit

## 目录结构

```
ClothesTypeDemo/
│
├── .cursor/             # Cursor AI 配置目录 (如果使用)
├── data/                # (建议) 存放数据相关文件或链接
├── models/              # (建议) 存放训练好的模型文件
│
├── main_ui.py           # Streamlit 应用主界面入口
├── trainer.py           # 模型训练核心逻辑
├── dataset.py           # 数据集加载与处理
├── model.py             # 神经网络模型定义
├── utils.py             # 辅助工具函数
│
├── README.md            # 项目说明文件 (就是这个文件)
└── requirements.txt     # Python 依赖库列表
```

## 如何开始 (初步)

1.  **环境设置:** 
    *   安装 Python (建议 3.8 或更高版本)。
    *   创建虚拟环境（强烈推荐），例如使用 `venv` 或 `conda`:
        ```bash
        # 使用 venv
        python -m venv .venv
        # Windows
        .venv\Scripts\activate
        # Linux/macOS
        source .venv/bin/activate 
        
        # 或者使用 conda
        conda create -n clothes_demo python=3.10 # 指定 Python 版本
        conda activate clothes_demo
        ```
    *   **安装 PyTorch (重要):** 
        *   **对于 GPU 用户 (尤其是较新的 NVIDIA 显卡):** 你需要安装与你的显卡驱动和 CUDA 版本匹配的 PyTorch。请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你系统的安装命令。
        *   **如果你需要 CUDA 12.8 支持 (例如 RTX 50 系列显卡):** 根据用户反馈，可以使用以下 nightly build 命令安装 (请注意 nightly build 可能不稳定):
            ```bash
            pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
            ```
        *   **对于 CPU 用户或不确定:** 可以尝试安装 CPU 版本：
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ```
    *   **安装其他依赖库:** 
        ```bash
        pip install -r requirements.txt
        ```
2.  **数据准备:**
    *   下载 DeepFashion 的 Category and Attribute Prediction Benchmark 数据集。
    *   确保标注文件位于 `E:\AIModels\DeepFashion\DeepFashion\Category and Attribute Prediction Benchmark\Anno_fine\` (或根据需要修改代码中的路径，推荐将数据放在项目 `data/` 目录下并通过参数指定)。
    *   确保图像文件位于 `E:\AIModels\DeepFashion\DeepFashion\Category and Attribute Prediction Benchmark\Img\img_highres\` (或相应的 `img/` 目录)。
3.  **运行应用:** 
    *   确保你处于激活的虚拟环境中。
    *   在项目根目录下运行 Streamlit 命令: 
        ```bash
        streamlit run main_ui.py
        ```

## 功能详解

### 模型预训练模块 (进行中)

(这里将详细描述训练模块的界面、功能、参数说明等)

### 模型效果测试模块 (待设计)

(这里将详细描述测试模块的功能)

---

**注意:** 本 README 会随着项目进展持续更新。 