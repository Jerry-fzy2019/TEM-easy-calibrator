"""
图像漂移/畸变校准工作台。

功能要点：
- 支持多文件上传（tif/dm3/png/jpg），执行基于相位相关的漂移估计与校正。
- 可配置亚像素精度倍数以获得更高精度。
- 手动像素比例换算（在漂移前图上画线）。
- 时间输入功能，计算漂移率。
- 结果包含漂移向量、校正后图像预览、漂移曲线与合格性判断。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 项目根目录加入路径，便于导入核心模块
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.drift.drift_correction import DriftAnalyzer  # noqa: E402


st.set_page_config(page_title="漂移校准", layout="wide")
st.title("📐 图像漂移/畸变校准")


def _save_uploaded_files(files: Sequence[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Path]:
    """将上传文件保存到临时目录，返回路径列表。"""
    tmp_dir = Path(tempfile.mkdtemp(prefix="drift_upload_"))
    paths: List[Path] = []
    for idx, f in enumerate(files):
        suffix = Path(f.name).suffix or ".tif"
        out_path = tmp_dir / f"frame_{idx:03d}{suffix}"
        with open(out_path, "wb") as fp:
            fp.write(f.getbuffer())
        paths.append(out_path)
    return paths


def _load_image_bgr(path: Path) -> np.ndarray:
    """读取图像，保持 16-bit/8-bit，返回 BGR 或灰度。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img


def _to_display_image(img: np.ndarray) -> np.ndarray:
    """将 8/16-bit BGR/Gray 转为可显示的 RGB/Gray uint8。"""
    if img is None:
        raise ValueError("图像为空，无法显示")
    arr = img
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    return arr


def _build_drift_dataframe(drift_vectors: List[tuple[float, float]]) -> pd.DataFrame:
    """将漂移向量转换为 DataFrame 便于展示。"""
    return pd.DataFrame(
        {
            "帧序号": list(range(len(drift_vectors))),
            "dy_像素": [v[0] for v in drift_vectors],
            "dx_像素": [v[1] for v in drift_vectors],
        }
    )


# ========== 侧边栏 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 时间输入
    st.subheader("⏱️ 时间设置")
    time_interval = st.number_input(
        "时间间隔 (分钟)",
        min_value=0.01,
        value=1.0,
        step=0.1,
        help="两张图像之间的时间间隔（分钟）",
        key="drift_time_interval"
    )
    
    st.divider()
    
    # 参数设置
    st.subheader("🔧 校准参数")
    upsample_factor = st.slider(
        "亚像素精度倍数",
        min_value=1,
        max_value=1000,
        value=100,
        step=1,
        help="相位相关亚像素精度倍数，值越大精度越高但计算更慢",
    )

# ========== 主工作区 ==========
st.header("📋 操作步骤")
drift_step = st.session_state.get("drift_step", 0)
col_step1, col_step2, col_step3 = st.columns(3)
with col_step1:
    if drift_step >= 1:
        st.success("✅ 步骤1: 上传图像")
    else:
        st.info("步骤1: 上传图像")
with col_step2:
    if drift_step >= 2:
        st.success("✅ 步骤2: 换算像素比例")
    else:
        st.info("步骤2: 换算像素比例")
with col_step3:
    if drift_step >= 3:
        st.success("✅ 步骤3: 查看漂移结果")
    else:
        st.info("步骤3: 查看漂移结果")
st.divider()

st.subheader("📤 步骤1: 上传图像")
col_up1, col_up2 = st.columns(2)
with col_up1:
    calib_file = st.file_uploader(
        "漂移前/标尺图（单张）",
        type=["tif", "tiff", "dm3", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="calib_file",
    )
with col_up2:
    drift_file = st.file_uploader(
        "漂移后图（单张）",
        type=["tif", "tiff", "dm3", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="drift_file",
    )

if not calib_file:
    st.session_state.drift_step = 0
    st.info("请先上传漂移前/标尺图，用于像素比例换算。")
elif not drift_file:
    st.session_state.drift_step = 1
    st.info("请再上传漂移后图以计算漂移。")
else:
    st.session_state.drift_step = 1

# ========== 像素比例换算 ==========
nm_per_pixel: Optional[float] = None

if calib_file:
    # 保存漂移前图片到临时位置用于校准
    suffix = Path(calib_file.name).suffix or ".png"
    calib_file.seek(0)
    file_content = calib_file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        tmp_calib_path = tmp_file.name
    
    try:
        # 加载图像
        calib_img = cv2.imread(tmp_calib_path, cv2.IMREAD_UNCHANGED)
        if calib_img is None:
            st.error("❌ 无法读取漂移前图片")
        else:
            # 转换为灰度（如果是彩色）
            if len(calib_img.shape) == 3:
                calib_img_gray = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
            else:
                calib_img_gray = calib_img
            
            # 确保是 uint8
            if calib_img_gray.dtype != np.uint8:
                if calib_img_gray.max() <= 1.0:
                    calib_img_gray = (calib_img_gray * 255).astype(np.uint8)
                else:
                    calib_img_gray = cv2.normalize(calib_img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 手动像素比例换算（缩放图 + 原图像素距离，避免显示缩放导致比例错误）
            with st.expander("✏️ 手动像素比例换算", expanded=True):
                st.markdown("**在图像上画一条线覆盖标尺，输入物理长度（按原图像素换算）**")
                img_height, img_width = calib_img_gray.shape[:2]
                if img_width == 0 or img_height == 0:
                    st.error("❌ 图像尺寸无效")
                else:
                    max_display_size = 400
                    scale_display = min(1.0, max_display_size / max(img_width, img_height))
                    display_width = int(img_width * scale_display)
                    display_height = int(img_height * scale_display)
                    image_pil = Image.fromarray(calib_img_gray, mode='L').convert('RGB')
                    image_pil = image_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=3,
                        stroke_color="#FF0000",
                        background_image=image_pil,
                        update_streamlit=True,
                        drawing_mode="line",
                        point_display_radius=0,
                        key="drift_scale_canvas",
                        width=display_width,
                        height=display_height,
                    )
                    if canvas_result and canvas_result.json_data:
                        objects = canvas_result.json_data.get("objects", [])
                        if objects:
                            last_line = objects[-1]
                            if last_line.get("type") == "line":
                                cx1 = float(last_line.get("x1", 0))
                                cy1 = float(last_line.get("y1", 0))
                                cx2 = float(last_line.get("x2", 0))
                                cy2 = float(last_line.get("y2", 0))
                                scale_x = img_width / display_width
                                scale_y = img_height / display_height
                                x1_orig = cx1 * scale_x
                                x2_orig = cx2 * scale_x
                                pixel_distance = abs(x2_orig - x1_orig)
                                st.info(f"像素距离（原图）: {pixel_distance:.2f} px")
                                default_length = st.session_state.get("drift_scale_length", 50.0)
                                scale_length_nm = st.number_input(
                                    "标尺物理长度 (nm)",
                                    min_value=0.1,
                                    value=default_length,
                                    step=0.1,
                                    key="drift_scale_length_input"
                                )
                                st.session_state["drift_scale_length"] = scale_length_nm
                                if st.button("✅ 设为像素比例", type="primary", key="confirm_drift_scale"):
                                    if pixel_distance > 1e-6:
                                        nm_per_pixel = scale_length_nm / pixel_distance
                                        st.session_state["drift_nm_per_pixel"] = nm_per_pixel
                                        st.session_state.drift_step = 2
                                        st.toast(f"✅ 像素比例换算成功！{nm_per_pixel:.4f} nm/px", icon="✅")
                                        st.rerun()
                                    else:
                                        st.warning("⚠️ 请先画一条有效线")
                        else:
                            st.info("👆 请在图像上画一条线覆盖标尺")
            
            # 显示已换算的像素比例
            if "drift_nm_per_pixel" in st.session_state:
                nm_per_pixel = st.session_state["drift_nm_per_pixel"]
                st.success(f"✅ 已设置像素比例: {nm_per_pixel:.4f} nm/px")
    
    finally:
        # 清理临时文件
        if os.path.exists(tmp_calib_path):
            try:
                os.remove(tmp_calib_path)
            except OSError:
                pass


def _render_preview(col, title: str, img: np.ndarray) -> None:
    col.markdown(f"**{title}**")
    if img is None:
        col.warning("无法加载此帧图像")
        return
    col.image(_to_display_image(img), clamp=True)




# ========== 执行分析 ==========
if st.button("🚀 开始校准", type="primary", use_container_width=True):
    if not calib_file or not drift_file:
        st.error("请同时上传漂移前（标尺）图和漂移后图。")
    elif "drift_nm_per_pixel" not in st.session_state:
        st.error("请先完成像素比例换算（在漂移前图上画线并点击'设为像素比例'）。")
    else:
        tmp_paths: List[Path] = []
        orig_images: List[np.ndarray] = []
        try:
            with st.spinner("正在执行漂移校准..."):
                # 保存上传文件，保持顺序：先标尺/漂移前，再漂移后
                calib_file.seek(0)
                drift_file.seek(0)
                tmp_paths = _save_uploaded_files([calib_file, drift_file])
                orig_images = [_load_image_bgr(p) for p in tmp_paths]
                if any(img is None for img in orig_images):
                    raise ValueError("存在无法读取的图像文件，请检查上传文件格式")

                analyzer = DriftAnalyzer()
                result = analyzer.analyze_single_image(
                    tmp_paths,
                    {"upsample_factor": upsample_factor},
                )

            if result.get("status") != "success":
                st.error(f"校准失败: {result.get('error_message', '未知错误')}")
                if result.get("traceback"):
                    with st.expander("查看 Traceback"):
                        st.code(result["traceback"])
            else:
                st.session_state.drift_step = 3
                drift_vectors = result["drift_vectors"]
                corrected_images = result["corrected_images"]
                processing_time = result.get("processing_time", 0)
                nm_per_pixel = st.session_state.get("drift_nm_per_pixel", None)

                st.success(f"✅ 校准完成，耗时 {processing_time} 秒")

                # 漂移结果
                df_drift = _build_drift_dataframe(drift_vectors)
                
                # 添加纳米单位（如果有像素比例）
                if nm_per_pixel and nm_per_pixel > 0:
                    df_drift["dy_纳米"] = df_drift["dy_像素"] * nm_per_pixel
                    df_drift["dx_纳米"] = df_drift["dx_像素"] * nm_per_pixel
                
                # 添加时间列
                if len(df_drift) > 1:
                    df_drift["时间_分钟"] = df_drift["帧序号"] * time_interval
                
                # ========== 1. 展示测量数据 ==========
                st.markdown("### 📊 测量数据")
                
                # 显示数据表格
                display_cols = ["帧序号", "dy_像素", "dx_像素"]
                if nm_per_pixel and nm_per_pixel > 0:
                    display_cols.extend(["dy_纳米", "dx_纳米"])
                if len(df_drift) > 1:
                    display_cols.append("时间_分钟")
                
                st.dataframe(df_drift[display_cols], hide_index=True, use_container_width=True)
                
                if nm_per_pixel and nm_per_pixel > 0:
                    st.caption(f"已按像素比例换算：1 px = {nm_per_pixel:.4f} nm")

                # ========== 2. 误差计算公式和计算 ==========
                if nm_per_pixel and nm_per_pixel > 0 and len(df_drift) > 1:
                    st.divider()
                    st.markdown("### 📐 误差计算")
                    
                    # 计算总漂移距离
                    total_drift_nm = np.sqrt(df_drift.iloc[-1]["dx_纳米"]**2 + df_drift.iloc[-1]["dy_纳米"]**2)
                    drift_rate_nm_per_min = total_drift_nm / time_interval if time_interval > 0 else 0
                    
                    # 显示计算公式
                    st.markdown("**计算公式：**")
                    st.latex(r"""
                    \begin{align}
                    \text{总漂移距离} &= \sqrt{(\Delta x)^2 + (\Delta y)^2} \\
                    \text{漂移率} &= \frac{\text{总漂移距离}}{\text{时间间隔}}
                    \end{align}
                    """)
                    
                    # 显示计算过程
                    dx_nm = df_drift.iloc[-1]["dx_纳米"]
                    dy_nm = df_drift.iloc[-1]["dy_纳米"]
                    st.markdown("**计算过程：**")
                    st.markdown(f"""
                    - 漂移向量：Δx = {dx_nm:.3f} nm, Δy = {dy_nm:.3f} nm
                    - 总漂移距离 = √({dx_nm:.3f}² + {dy_nm:.3f}²) = **{total_drift_nm:.3f} nm**
                    - 漂移率 = {total_drift_nm:.3f} nm / {time_interval:.2f} min = **{drift_rate_nm_per_min:.3f} nm/min**
                    """)
                    
                    # ========== 3. 误差值和规范比对并给出检定结果 ==========
                    st.divider()
                    st.markdown("### ✅ 检定结果")
                    
                    # TEM漂移标准：高分辨率TEM漂移率应 < 0.5 nm/min
                    drift_threshold = 0.5  # nm/min
                    is_qualified = drift_rate_nm_per_min < drift_threshold
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总漂移距离", f"{total_drift_nm:.3f} nm")
                    with col2:
                        st.metric("漂移率", f"{drift_rate_nm_per_min:.3f} nm/min")
                    with col3:
                        if is_qualified:
                            st.success(f"✅ 合格\n(标准: < {drift_threshold} nm/min)")
                        else:
                            st.error(f"❌ 不合格\n(标准: < {drift_threshold} nm/min)")

                # ========== 4. 漂移矢量方向示意 ==========
                if len(drift_vectors) > 0:
                    st.divider()
                    st.markdown("### 🧭 漂移矢量方向示意")
                    # 取最后一帧的漂移向量（相对参考帧）
                    dy_pix, dx_pix = drift_vectors[-1]
                    mag_pix = float(np.hypot(dx_pix, dy_pix))
                    nm_per_pixel = st.session_state.get("drift_nm_per_pixel", None)
                    mag_nm = mag_pix * nm_per_pixel if nm_per_pixel and nm_per_pixel > 0 else None
                    base_img = _to_display_image(orig_images[0]).copy()
                    h0, w0 = base_img.shape[:2]
                    cx, cy = w0 // 2, h0 // 2
                    # 为了让方向可见，保证箭头在可见范围内（仅用于显示，不影响数值）
                    if mag_pix < 1e-6:
                        # 几乎没有漂移时，用一个固定长度的水平箭头表示接近零漂移
                        dx_vis, dy_vis = 40.0, 0.0
                    else:
                        # 至少放大到 40 像素长度，便于肉眼观察方向
                        min_len = 40.0
                        vis_scale = max(1.0, min_len / mag_pix)
                        dx_vis = dx_pix * vis_scale
                        dy_vis = dy_pix * vis_scale
                    end_x = int(round(cx + dx_vis))
                    end_y = int(round(cy + dy_vis))
                    cv2.arrowedLine(
                        base_img,
                        (cx, cy),
                        (end_x, end_y),
                        (0, 255, 0),
                        2,
                        tipLength=0.3,
                    )
                    # 在箭头终点附近标注距离（优先显示 nm，其次 px），只使用 ASCII 字符避免显示为 "???"
                    # 用物理量形式 |r| 表示漂移矢量的模长
                    if mag_nm is not None:
                        label = f"|r| = {mag_nm:.3f} nm"
                    else:
                        label = f"|r| = {mag_pix:.2f} px"
                    cv2.putText(
                        base_img,
                        label,
                        (end_x + 5, end_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    st.image(
                        base_img,
                        caption=f"漂移方向示意：绿色箭头为漂移方向，终点文字为本次估算位移（优先 nm）",
                        clamp=True,
                    )


        finally:
            # 清理临时目录
            for p in tmp_paths:
                if p.exists():
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if tmp_paths:
                try:
                    shutil.rmtree(tmp_paths[0].parent, ignore_errors=True)
                except Exception:
                    pass
