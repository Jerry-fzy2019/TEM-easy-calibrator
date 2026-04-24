"""
污染率表征页面 - 基于轮廓的形态测量法（带像素比例换算）。
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.contamination.contamination_eval import ContaminationAnalyzer

st.set_page_config(page_title="污染率", layout="wide")

st.title("💧 TEM 污染率表征")

st.info("""
本模块用于测量随时间变大的不规则斑块（如积碳或孔洞扩张）。
请先上传第一张图片（带标尺）进行像素比例换算，再上传后续图片序列。
建议先运行漂移校准以确保图像序列稳定。
""")

# ========== 侧边栏参数设置 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 时间输入
    st.subheader("⏱️ 时间设置")
    time_interval = st.number_input(
        "时间间隔 (分钟)",
        min_value=0.01,
        value=1.0,
        step=0.1,
        help="相邻图像之间的时间间隔（分钟）",
        key="contamination_time_interval"
    )
    
    st.divider()
    
    # 检测参数
    st.subheader("🔍 检测参数")
    target_type = st.radio(
        "检测目标类型",
        options=["黑斑 (Dark Spot)", "亮孔 (Bright Hole)"],
        index=0,
        help="黑斑：亮背景下的暗色区域（如积碳）；亮孔：暗背景下的亮色区域（如刻蚀孔）"
    )
    
    target_type_value = "dark" if target_type == "黑斑 (Dark Spot)" else "bright"
    
    min_area = st.number_input(
        "最小轮廓面积 (像素)",
        min_value=10,
        value=100,
        step=10,
        help="过滤掉面积小于此值的噪点轮廓"
    )
    
    show_overlay = st.checkbox(
        "显示轮廓叠加",
        value=True,
        help="在代表性图像上叠加绿色轮廓线，用于验证识别准确性"
    )

# ========== 主工作区 ==========
st.header("📋 操作步骤")
contamination_step = st.session_state.get("contamination_step", 0)
col_step1, col_step2, col_step3 = st.columns(3)
with col_step1:
    if contamination_step >= 1:
        st.success("✅ 步骤1: 上传图像")
    else:
        st.info("步骤1: 上传图像")
with col_step2:
    if contamination_step >= 2:
        st.success("✅ 步骤2: 换算像素比例")
    else:
        st.info("步骤2: 换算像素比例")
with col_step3:
    if contamination_step >= 3:
        st.success("✅ 步骤3: 查看污染率结果")
    else:
        st.info("步骤3: 查看污染率结果")
st.divider()

st.subheader("📤 步骤1: 上传图像")
col_up1, col_up2 = st.columns(2)
with col_up1:
    first_file = st.file_uploader(
        "第一张图片（带标尺，用于像素比例换算）",
        type=["tif", "tiff", "dm3", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="first_file",
    )
with col_up2:
    sequence_files = st.file_uploader(
        "后续图片序列（按时间顺序）",
        type=["tif", "tiff", "dm3", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="sequence_files",
    )

if not first_file:
    st.session_state.contamination_step = 0
    st.info("请先上传第一张图片（带标尺）进行像素比例换算。")
elif not sequence_files or len(sequence_files) == 0:
    st.session_state.contamination_step = 1
    st.info("请上传后续图片序列以分析污染区域生长。")
else:
    st.session_state.contamination_step = 1

# ========== 像素比例换算 ==========
nm_per_pixel: Optional[float] = None

if first_file:
    # 保存第一张图片到临时位置用于校准
    suffix = Path(first_file.name).suffix or ".png"
    first_file.seek(0)
    file_content = first_file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        tmp_first_path = tmp_file.name
    
    try:
        # 加载图像
        first_img = cv2.imread(tmp_first_path, cv2.IMREAD_UNCHANGED)
        if first_img is None:
            st.error("❌ 无法读取第一张图片")
        else:
            # 转换为灰度（如果是彩色）
            if len(first_img.shape) == 3:
                first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
            
            # 确保是 uint8
            if first_img.dtype != np.uint8:
                if first_img.max() <= 1.0:
                    first_img = (first_img * 255).astype(np.uint8)
                else:
                    first_img = cv2.normalize(first_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 手动像素比例换算（缩放图 + 原图像素距离，避免显示缩放导致比例错误）
            with st.expander("✏️ 手动像素比例换算", expanded=True):
                st.markdown("**在图像上画一条线覆盖标尺，输入物理长度（按原图像素换算）**")
                img_height, img_width = first_img.shape[:2]
                if img_width == 0 or img_height == 0:
                    st.error("❌ 图像尺寸无效")
                else:
                    max_display_size = 400
                    scale_display = min(1.0, max_display_size / max(img_width, img_height))
                    display_width = int(img_width * scale_display)
                    display_height = int(img_height * scale_display)
                    image_pil = Image.fromarray(first_img, mode='L').convert('RGB')
                    image_pil = image_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=3,
                        stroke_color="#FF0000",
                        background_image=image_pil,
                        update_streamlit=True,
                        drawing_mode="line",
                        point_display_radius=0,
                        key="contamination_scale_canvas",
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
                                scale_length_nm = st.number_input(
                                    "标尺物理长度 (nm)",
                                    min_value=0.1,
                                    value=50.0,
                                    step=0.1,
                                    key="contamination_scale_length"
                                )
                                if st.button("✅ 设为像素比例", type="primary", key="confirm_contamination_scale"):
                                    if pixel_distance > 1e-6:
                                        nm_per_pixel = scale_length_nm / pixel_distance
                                        st.session_state["contamination_nm_per_pixel"] = nm_per_pixel
                                        st.session_state.contamination_step = 2
                                        st.toast(f"✅ 像素比例换算成功！{nm_per_pixel:.4f} nm/px", icon="✅")
                                        st.rerun()
                                    else:
                                        st.warning("⚠️ 请先画一条有效线")
                        else:
                            st.info("👆 请在图像上画一条线覆盖标尺")
            
            # 显示已换算的像素比例
            if "contamination_nm_per_pixel" in st.session_state:
                nm_per_pixel = st.session_state["contamination_nm_per_pixel"]
                st.success(f"✅ 已设置像素比例: {nm_per_pixel:.4f} nm/px")
    
    finally:
        # 清理临时文件
        if os.path.exists(tmp_first_path):
            try:
                os.remove(tmp_first_path)
            except OSError:
                pass

# ========== 执行分析 ==========
if st.button("🚀 开始分析", type="primary", use_container_width=True):
    if not first_file:
        st.error("请先上传第一张图片（带标尺）进行像素比例换算。")
    elif not sequence_files or len(sequence_files) == 0:
        st.error("请上传后续图片序列。")
    elif "contamination_nm_per_pixel" not in st.session_state:
        st.error("请先完成像素比例换算（在第一张图片上画线并点击'设为像素比例'）。")
    else:
        tmp_paths: List[Path] = []
        try:
            with st.spinner("正在分析污染区域生长..."):
                # 保存所有文件到临时目录（第一张 + 序列）
                tmp_dir = Path(tempfile.mkdtemp(prefix="contamination_"))
                tmp_paths = []
                
                # 保存第一张图片
                first_file.seek(0)
                suffix1 = Path(first_file.name).suffix or ".png"
                tmp_path1 = tmp_dir / f"frame_000{suffix1}"
                with open(tmp_path1, "wb") as f:
                    f.write(first_file.getbuffer())
                tmp_paths.append(tmp_path1)
                
                # 保存序列图片
                for idx, uploaded_file in enumerate(sequence_files):
                    suffix = Path(uploaded_file.name).suffix or ".png"
                    tmp_path = tmp_dir / f"frame_{idx+1:03d}{suffix}"
                    uploaded_file.seek(0)
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    tmp_paths.append(tmp_path)
                
                # 按文件名排序
                tmp_paths.sort()
                
                analyzer = ContaminationAnalyzer()
                result = analyzer.analyze_single_image(
                    tmp_paths,
                    {
                        "target_type": target_type_value,
                        "min_area": int(min_area),
                        "show_overlay": show_overlay,
                        "nm_per_pixel": st.session_state.get("contamination_nm_per_pixel", None),
                    },
                )
            
            if result.get("status") != "success":
                st.error(f"分析失败: {result.get('error_message', '未知错误')}")
                if result.get("traceback"):
                    with st.expander("查看 Traceback"):
                        st.code(result["traceback"])
            else:
                st.session_state.contamination_step = 3
                growth_curve = result["growth_curve"]
                overlay_images = result.get("overlay_images", [])
                processing_time = result.get("processing_time", 0)
                
                st.success(f"✅ 分析完成，耗时 {processing_time} 秒")
                
                # 获取像素比例
                nm_per_pixel = st.session_state.get("contamination_nm_per_pixel", None)
                
                # ========== 数据统计 ==========
                if len(growth_curve) > 0:
                    df = pd.DataFrame(growth_curve)
                    
                    # 过滤掉无效数据（面积为0）
                    df_valid = df[df["area_px"] > 0].copy()
                    
                    if len(df_valid) > 0:
                        # 添加时间列
                        df_valid["时间_分钟"] = df_valid["frame"] * time_interval
                        
                        # 计算纳米单位（如果有像素比例）
                        if nm_per_pixel and nm_per_pixel > 0:
                            df_valid["diameter_nm"] = df_valid["diameter_px"] * nm_per_pixel
                            df_valid["area_nm2"] = df_valid["area_px"] * (nm_per_pixel ** 2)
                        
                        initial_diameter_px = df_valid.iloc[0]["diameter_px"]
                        final_diameter_px = df_valid.iloc[-1]["diameter_px"]
                        total_time_min = (len(df_valid) - 1) * time_interval if len(df_valid) > 1 else 0
                        
                        # ========== 1. 展示测量数据 ==========
                        st.markdown("### 📊 测量数据")
                        
                        # 数据表格
                        display_cols = ["frame", "diameter_px", "area_px", "circularity"]
                        if "时间_分钟" in df_valid.columns:
                            display_cols.append("时间_分钟")
                        if nm_per_pixel and nm_per_pixel > 0:
                            display_cols.extend(["diameter_nm", "area_nm2"])
                        
                        # 重命名列以显示中文
                        df_display = df_valid[display_cols].copy()
                        column_mapping = {
                            "frame": "帧序号",
                            "diameter_px": "直径_像素",
                            "area_px": "面积_像素",
                            "circularity": "圆形度",
                            "时间_分钟": "时间_分钟",
                            "diameter_nm": "直径_纳米",
                            "area_nm2": "面积_平方纳米"
                        }
                        df_display.rename(columns=column_mapping, inplace=True)
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                        
                        # ========== 2. 误差计算公式和计算 ==========
                        if nm_per_pixel and nm_per_pixel > 0 and total_time_min > 0:
                            st.divider()
                            st.markdown("### 📐 误差计算")
                            
                            # 计算污染率（nm/min）
                            initial_diameter_nm = initial_diameter_px * nm_per_pixel
                            final_diameter_nm = final_diameter_px * nm_per_pixel
                            contamination_rate_nm_per_min = (final_diameter_nm - initial_diameter_nm) / total_time_min
                            
                            # 显示计算公式
                            st.markdown("**计算公式：**")
                            st.latex(r"""
                            \begin{align}
                            \text{污染率} = \frac{\text{最终直径} - \text{初始直径}}{\text{总时间}}
                            \end{align}
                            """)
                            
                            # 显示计算过程
                            st.markdown("**计算过程：**")
                            st.markdown(f"""
                            - 初始直径 = {initial_diameter_px:.2f} px × {nm_per_pixel:.4f} nm/px = **{initial_diameter_nm:.3f} nm**
                            - 最终直径 = {final_diameter_px:.2f} px × {nm_per_pixel:.4f} nm/px = **{final_diameter_nm:.3f} nm**
                            - 总时间 = {total_time_min:.2f} min
                            - 污染率 = ({final_diameter_nm:.3f} - {initial_diameter_nm:.3f}) nm / {total_time_min:.2f} min = **{contamination_rate_nm_per_min:.3f} nm/min**
                            """)
                            
                            # ========== 3. 误差值和规范比对并给出检定结果 ==========
                            st.divider()
                            st.markdown("### ✅ 检定结果")
                            
                            # TEM污染率标准：对于高分辨率TEM，污染率应 < 0.1 nm/min（理想情况）
                            # 实际应用中，< 0.5 nm/min 可接受
                            contamination_threshold = 0.5  # nm/min
                            is_qualified = contamination_rate_nm_per_min < contamination_threshold
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("初始直径", f"{initial_diameter_nm:.3f} nm", f"{initial_diameter_px:.2f} px")
                            with col2:
                                st.metric("最终直径", f"{final_diameter_nm:.3f} nm", f"{final_diameter_px:.2f} px")
                            with col3:
                                st.metric("污染率", f"{contamination_rate_nm_per_min:.3f} nm/min")
                            
                            col_result = st.columns(1)[0]
                            with col_result:
                                if is_qualified:
                                    st.success(f"✅ 合格（标准: < {contamination_threshold} nm/min）")
                                else:
                                    st.error(f"❌ 不合格（标准: < {contamination_threshold} nm/min）")
                        else:
                            st.info("⚠️ 需要完成像素比例换算和时间设置才能进行误差计算和合格性判断")
                        
                        # ========== 4. 轮廓识别验证（仅用代表性图像，不绘制曲线图表） ==========
                        if show_overlay and len(overlay_images) > 0:
                            st.divider()
                            st.markdown("### 🔍 轮廓识别验证")
                            st.markdown("**绿色轮廓线标识算法识别到的目标区域**")
                            
                            # 展示代表性图像
                            cols = st.columns(len(overlay_images))
                            labels = ["起始帧", "中间帧", "结束帧"]
                            
                            for idx, (col, overlay_img) in enumerate(zip(cols, overlay_images)):
                                with col:
                                    label = labels[idx] if idx < len(labels) else f"帧 {idx}"
                                    st.image(overlay_img, caption=label)
                    else:
                        st.warning("⚠️ 未检测到有效的污染区域，请检查图像或调整参数。")
                else:
                    st.warning("⚠️ 未检测到任何轮廓，请检查图像或调整最小面积阈值。")
        
        finally:
            # 清理临时文件
            for p in tmp_paths:
                if p.exists():
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            # 清理临时目录
            if tmp_paths:
                tmp_dir = tmp_paths[0].parent
                try:
                    tmp_dir.rmdir()
                except OSError:
                    pass
