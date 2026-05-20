# TEM Easy Calibrator

TEM Easy Calibrator 是一个面向透射电子显微镜（TEM）图像的交互式校准与分析工具。项目基于 Python、Streamlit、OpenCV 和科学计算生态构建，把常见 TEM 校准流程整理成多个可视化页面，适合在本地快速完成图像上传、像素比例换算、FFT/SAED 分析、漂移评估和污染率表征。

> 说明：本项目定位为科研辅助工具。计算结果应结合样品信息、显微镜条件和人工复核共同判断，不建议作为唯一判定依据。

## 下载与安装

如果只是使用 TEM Easy Calibrator，推荐优先从 [GitHub Releases](https://github.com/Jerry-fzy2019/TEM-easy-calibrator/releases) 下载已打包的 Windows 版本：

- `TEM-Easy-Calibrator-Setup-版本号.exe`：安装器，可安装到系统程序目录并创建快捷方式。
- `TEM-Easy-Calibrator-windows-portable.zip`：便携版，解压后直接运行。

目前预构建包主要面向 Windows。macOS / Linux 用户可以继续阅读“快速开始”，从源码运行项目。

### 本地打包

Windows PowerShell：

```powershell
.\scripts\build_windows.ps1
```

如果只想生成便携版 zip，不生成安装器：

```powershell
.\scripts\build_windows.ps1 -SkipInstaller
```

如果需要把 DM3/HyperSpy 支持也打进程序：

```powershell
.\scripts\build_windows.ps1 -IncludeDm3
```

> 注意：`-IncludeDm3` 会把 GPLv3 许可证的 HyperSpy 打进发布包。分发该版本时，请同时遵守 HyperSpy 的许可证要求。默认构建不包含 HyperSpy。

### GitHub Actions 自动发布

仓库已包含 `.github/workflows/build-windows.yml`。你可以在 GitHub 页面手动运行 workflow，生成 Windows 便携版和安装器。

如果创建形如 `v0.1.0` 的 tag 并推送到 GitHub，workflow 会自动构建并发布到 GitHub Release：

```bash
git tag v0.1.0
git push origin v0.1.0
```

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

部分页面预留了 `.dm3` 入口。核心功能不依赖 DM3 支持；如果需要从 DM3 元数据中自动读取像素比例，可以额外安装可选依赖：

```bash
pip install -r requirements-dm3.txt
```

DM3 元数据读取会使用 HyperSpy。HyperSpy 采用 GPLv3 许可证，因此它被拆分为可选依赖，未包含在主依赖 `requirements.txt` 中。

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

建议使用 Python 3.10、3.11 或 3.12。由于当前依赖中包含 `numpy<2.0`，不建议使用 Python 3.13 进行安装或打包。

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
├── requirements.txt                # 主依赖
├── requirements-dm3.txt            # 可选 DM3 元数据读取依赖
├── requirements-build.txt          # 打包构建依赖
├── LICENSE                         # Apache-2.0 许可证
├── scripts                         # 本地构建脚本
├── packaging                       # PyInstaller 与安装器配置
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

## 许可证

本项目采用 Apache License 2.0 开源。详见 [LICENSE](LICENSE)。

DM3 元数据读取是可选功能，依赖 HyperSpy。HyperSpy 采用 GPLv3 许可证；如果你安装 `requirements-dm3.txt` 并分发包含 HyperSpy 的版本，请同时遵守 HyperSpy 的许可证要求。

## 贡献

欢迎通过 Issue 或 Pull Request 反馈问题、改进算法或补充示例数据。提交问题时建议包含：

- 操作系统与 Python 版本
- 运行方式（Streamlit 或桌面入口）
- 输入图像格式
- 复现步骤
- 错误日志或截图
