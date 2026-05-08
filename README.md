# TEM Easy Calibrator

TEM Easy Calibrator 是一个面向透射电子显微镜（TEM）图像的交互式校准与分析工具。项目基于 Python、Streamlit、OpenCV 和科学计算生态构建，把常见 TEM 校准流程整理成多个可视化页面，适合在本地快速完成图像上传、像素比例换算、FFT/SAED 分析、漂移评估和污染率表征。

> 说明：本项目定位为科研辅助工具。计算结果应结合样品信息、显微镜条件和人工复核共同判断，不建议作为唯一判定依据。

## 功能特性

| 模块 | 主要能力 |
| --- | --- |
| 膜厚校准 | 上传 TEM 图像，进行预处理、像素比例换算、线剖面分析，并估算单层条纹厚度与误差。 |
| 高分辨晶面间距 | 基于 FFT 选择中心与衍射斑点，支持 Friedel 对称过滤、IFFT 重构，并计算晶面间距。 |
| SAED 选区衍射校准 | 自动或手动定位透射斑中心，使用 LoG 检测衍射斑点，计算 d-spacing、角度和距离分组。 |
| 漂移/畸变校准 | 对漂移前后图像进行相位相关配准，输出亚像素漂移向量、漂移距离和漂移率。 |
| 污染率表征 | 对时间序列图像进行轮廓提取，计算目标区域等效直径、面积、圆形度和污染增长率。 |

## 支持格式

常规图像格式：

- `.tif` / `.tiff`
- `.png`
- `.jpg` / `.jpeg`

部分页面预留了 `.dm3` 入口，并会尝试通过 HyperSpy 读取 DM3 元数据中的像素比例。DM3 文件的实际图像读取兼容性会受文件来源和依赖版本影响，建议优先使用 TIF/TIFF 作为开源示例格式。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Jerry-fzy2019/TEM-easy-calibrator.git
cd TEM-easy-calibrator
```

### 2. 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

建议使用 Python 3.10 或更新版本。

### 3. 启动 Web 界面

```bash
streamlit run src/ui_streamlit/app.py
```

启动后在浏览器中打开 Streamlit 给出的本地地址，通常是：

```text
http://localhost:8501
```

### 4. 启动桌面窗口

项目也提供了一个基于 pywebview 的桌面入口：

```bash
python main.py
```

该入口会在后台启动 Streamlit 服务，并在本地桌面窗口中打开应用。

## 基本使用流程

1. 从左侧导航栏选择分析模块。
2. 上传 TEM 图像或图像序列。
3. 通过直接输入 `nm/px` 或在标尺上画线完成像素比例换算。
4. 根据页面提示选择中心点、衍射斑点、线剖面或图像序列。
5. 查看结果表格、图像叠加、曲线图和合格性判断。

## 项目结构

```text
.
├── main.py                         # 桌面入口，使用 pywebview 承载 Streamlit
├── requirements.txt                # Python 依赖
└── src
    ├── core                        # 核心算法与图像处理逻辑
    │   ├── common                  # 通用像素比例换算、视觉工具
    │   ├── contamination           # 污染率表征
    │   ├── drift                   # 漂移/畸变校准
    │   ├── hr_spacing              # 高分辨晶面间距
    │   ├── saed                    # SAED 选区衍射校准
    │   └── thickness               # 膜厚校准与 FFT/线剖面分析
    └── ui_streamlit                # Streamlit UI
        ├── app.py                  # Streamlit 首页
        ├── pages                   # 多页面分析模块
        └── scale_calibration_dialog.py
```

## 开发

当前项目没有单独的自动化测试套件。可以先用 Python 编译检查确认源码语法正常：

```bash
python -m compileall -q .
```

运行 Streamlit 后，建议至少手动检查以下路径：

- 首页是否能正常打开。
- 五个页面是否能从侧边栏进入。
- TIF/PNG/JPG 图像是否能上传。
- 像素比例换算、画线和结果展示是否符合预期。

## 开源前建议

- 添加 `LICENSE` 文件，明确开源许可证。
- 添加 `.gitignore`，忽略 `__pycache__/`、`.pyc`、虚拟环境和本地缓存。
- 补充一组脱敏的示例图片或截图，方便用户快速理解输入和输出。
- 如果计划发布桌面版，可以补充 PyInstaller 打包脚本和发布流程。

## 许可证

当前仓库尚未包含许可证文件。正式开源前请先选择并添加许可证，例如 MIT、BSD-3-Clause、Apache-2.0 或 GPL-3.0。

## 贡献

欢迎通过 Issue 或 Pull Request 反馈问题、改进算法或补充示例数据。提交问题时建议包含：

- 操作系统与 Python 版本
- 运行方式（Streamlit 或桌面入口）
- 输入图像格式
- 复现步骤
- 错误日志或截图
