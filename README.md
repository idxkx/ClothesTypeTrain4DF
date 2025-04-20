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
├── data/                # (建议) 存放数据相关文件或链接 (此目录已在 .gitignore 中，不会提交到 Git)
├── models/              # (建议) 存放训练好的模型文件 (此目录已在 .gitignore 中，不会提交到 Git)
│
├── main_ui.py           # (计划中) Streamlit 应用主界面入口
├── trainer.py           # (计划中) 模型训练核心逻辑
├── dataset.py           # (计划中) 数据集加载与处理
├── model.py             # (计划中) 神经网络模型定义
├── utils.py             # (计划中) 辅助工具函数
│
├── .gitignore           # Git 忽略配置文件
├── README.md            # 项目说明文件 (就是这个文件)
└── requirements.txt     # Python 基础依赖库列表 (PyTorch 需单独安装)
```
*注意: `.gitignore` 文件配置了忽略 `data/`, `models/`, Python 缓存, 虚拟环境等，确保仓库整洁。*

## 如何开始 (初步)

1.  **环境设置:** 
    *   安装 Python (建议 3.8 或更高版本)。
    *   创建虚拟环境（强烈推荐），例如使用 `venv` 或 `conda`:
        ```bash
        # 使用 venv
        python -m venv .venv
        # Windows
        .venv\\Scripts\\activate
        # Linux/macOS
        source .venv/bin/activate 
        
        # 或者使用 conda
        conda create -n clothes_demo python=3.10 # 指定 Python 版本
        conda activate clothes_demo
        ```
    *   **安装 PyTorch (重要):** 
        *   访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你系统的 **稳定版本** 安装命令。
        *   根据你的硬件选择合适的版本：
            *   **GPU 加速 (NVIDIA):** 选择与你的显卡驱动和 CUDA 版本匹配的选项。
            *   **CPU Only:** 如果没有合适的 GPU 或不确定，选择 CPU 版本。
        *   *(可选，仅当最新稳定版不支持你的硬件时考虑)* 对于非常新的硬件（如需要特定 CUDA 版本如 12.8），可能需要查找预发布版 (Nightly Build)，但这可能不稳定。
    *   **安装其他依赖库:** 
        *   确保 PyTorch 已安装。
        *   运行以下命令安装 `requirements.txt` 中列出的基础库:
            ```bash
            pip install -r requirements.txt
            ```
2.  **数据准备:**
    *   **下载数据:** 从官方渠道获取 DeepFashion 的 "Category and Attribute Prediction Benchmark" 数据集。你需要图像文件 (通常在 `img/` 或 `img_highres/` 目录下) 和标注文件 (通常在 `Anno_fine/` 目录下)。
    *   **组织数据 (重要):** 
        *   **强烈建议:** 将下载的数据集（或其符号链接）放置在项目根目录下的 `data/` 目录中，例如：
            ```
            ClothesTypeDemo/
            ├── data/
            │   ├── Anno_fine/     # 标注文件
            │   └── img_highres/   # 图像文件 (或 img/)
            │   ...
            ├── main_ui.py
            ...
            ```
        *   **原因:** 这样可以方便地在代码中使用**相对路径**引用数据，避免像 `E:\AIModels\...` 这样的**硬编码绝对路径**。硬编码路径会导致项目在不同电脑或不同用户那里无法直接运行。
        *   代码实现时，应设计成可以配置数据路径的方式（例如通过配置文件或命令行参数），而不是写死路径。
    *   *提醒: `data/` 目录及其内容默认不会被提交到 Git 仓库 (已在 `.gitignore` 中配置)。*
3.  **运行应用:** 
    *   确保你处于激活的虚拟环境中。
    *   在项目根目录下运行 Streamlit 命令: 
        ```bash
        streamlit run main_ui.py
        ```

## 功能详解

### 模型预训练模块 (进行中)

此模块旨在提供一个可视化的训练环境，主要功能规划如下：

*   **训练启动与配置:**
    *   选择模型架构 (例如 ResNet, VGG 等)。
    *   调整核心训练参数 (学习率、批次大小、训练轮次)。
    *   配置数据增强策略。
*   **实时训练监控:**
    *   可视化显示损失函数、准确率 (类别/属性) 等关键指标曲线。
    *   显示训练进度、预计剩余时间。
    *   (可选) GPU/CPU 资源使用情况监控。
*   **智能提示与建议:**
    *   根据训练状态 (如过拟合、欠拟合) 提供调整建议。
    *   参数调整的辅助说明和推荐范围。

### 模型效果测试模块 (待设计)

此模块用于评估已训练模型的性能，主要功能规划如下：

*   **模型选择:** 加载和切换不同的已训练模型。
*   **图像输入:** 支持上传本地图片、网络图片 URL 或摄像头拍摄。
*   **结果展示:**
    *   显示预测的服装类别及其置信度。
    *   显示检测到的服装属性。
    *   (可选) 可视化模型关注区域 (例如使用热力图)。
*   **参数调整:** (可选) 允许调整部分推理参数 (如置信度阈值)。

---

**注意:** 本 README 会随着项目进展持续更新。 