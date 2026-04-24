"""
SAED 选区衍射校准页面。
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.saed.saed_calibration import SAEDAnalyzer
from src.core.thickness.image_io import load_image
st.set_page_config(page_title="SAED 校准", layout="wide")

st.title("📐 SAED 选区衍射校准")

# ========== 侧边栏参数设置 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 中心检测参数（自动检测模式说明）
    with st.expander("🎯 中心检测", expanded=False):
        st.info("自动检测：选择图像中心最近的亮斑作为透射斑中心")
    
    # 斑点检测参数
    with st.expander("🔬 斑点检测参数", expanded=False):
        mask_radius = st.slider(
            "中心屏蔽半径",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="排除透射斑区域的半径（像素）"
        )
        
        min_sigma = st.slider(
            "最小 Sigma",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="LoG 检测的最小 sigma 值"
        )
        
        max_sigma = st.slider(
            "最大 Sigma",
            min_value=10.0,
            max_value=50.0,
            value=20.0,
            step=5.0,
            help="LoG 检测的最大 sigma 值"
        )
        
        threshold = st.slider(
            "检测阈值",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="LoG 检测的阈值"
        )
        
        distance_group_threshold_pix = st.slider(
            "距离分组阈值 (像素)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="用于将相似半径的斑点归为同一组（a,b,c...），像素差小于该值视为同一长度"
        )

# ========== 主工作区 ==========
st.header("📋 操作步骤")
saed_step = st.session_state.get("saed_step", 0)
col_step1, col_step2, col_step3, col_step4 = st.columns(4)
with col_step1:
    if saed_step >= 1:
        st.success("✅ 步骤1: 上传图像并标定")
    else:
        st.info("步骤1: 上传图像并标定")
with col_step2:
    if saed_step >= 2:
        st.success("✅ 步骤2: 选择中心")
    else:
        st.info("步骤2: 选择中心")
with col_step3:
    if saed_step >= 3:
        st.success("✅ 步骤3: 开始分析")
    else:
        st.info("步骤3: 开始分析")
with col_step4:
    if saed_step >= 4:
        st.success("✅ 步骤4: 查看结果")
    else:
        st.info("步骤4: 查看结果")
st.divider()

st.subheader("📤 步骤1: 上传图像并标定")
uploaded_file = st.file_uploader(
    "上传 SAED 衍射图",
    type=['tif', 'tiff', 'png', 'jpg', 'jpeg', 'dm3'],
    help="支持 TIF, PNG, JPG, DM3 格式",
    key="saed_uploaded_file",
)

if uploaded_file is None:
    st.session_state.saed_step = 0
    st.info("👆 请上传 SAED 衍射图以开始分析...")
    st.markdown("""
        ### 使用说明
        
        1. **上传图像**: 点击上方上传区域，选择您的 SAED 衍射图文件
        2. **设置标定**: 在左侧边栏设置标定系数（如果已标定）
        3. **调整参数**: 根据需要调整斑点检测参数
        4. **开始分析**: 点击"🚀 开始分析"按钮执行分析
        5. **查看结果**: 分析完成后，查看检测到的衍射斑点和 d-spacing 信息
        
        ### 功能特点
        
        - ✅ 自动透射斑中心检测（中心最近亮斑）
        - ✅ LoG 斑点检测
        - ✅ d-spacing 和角度计算
        - ✅ 交互式结果可视化
        """)
else:
        # 保存上传的文件到临时位置
        suffix = Path(uploaded_file.name).suffix
        if not suffix:
            suffix = '.png'
        
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            st.session_state.saed_step = 1
            # 初始化分析器
            if 'saed_analyzer' not in st.session_state:
                st.session_state.saed_analyzer = SAEDAnalyzer()
            
            analyzer = st.session_state.saed_analyzer
            
            # 加载图像
            image = load_image(Path(tmp_path))
            if image is None:
                st.error("❌ 图像加载失败，请检查文件格式")
                st.stop()
            
            # 保存图像和路径到session_state（用于像素比例换算）
            st.session_state.saed_current_image = image
            st.session_state.saed_current_file_path = tmp_path
            
            # 显示原始图像
            st.subheader("🖼️ 原始图像")
            st.image(image, caption=f"图像尺寸: {image.shape[1]}×{image.shape[0]} 像素")

            # SAED 标定（倒易空间标定系数 = nm⁻¹/px，d = 1/(r_pix×系数)），单独一栏
            st.subheader("📏 步骤1: SAED 标定系数 (倒易空间)")
            if 'saed_calibration_factor' not in st.session_state:
                st.session_state.saed_calibration_factor = None
            cal_mode = st.radio(
                "标定方式",
                ["直接输入标定系数 (nm⁻¹/px)", "标尺画线 (nm⁻¹/px)"],
                key="saed_cal_mode",
                help="d-spacing 公式: d = 1/(r_pix × 标定系数)，标定系数单位 nm⁻¹/px",
            )
            current_k = st.session_state.get("saed_calibration_factor")
            current_k = float(current_k) if current_k and current_k > 0 else 0.01
            if cal_mode == "直接输入标定系数 (nm⁻¹/px)":
                k_input = st.number_input(
                    "标定系数 (nm⁻¹/px)",
                    min_value=1e-6,
                    value=current_k,
                    step=0.001,
                    format="%.6f",
                    key="saed_k_input",
                )
                if k_input > 0:
                    st.session_state.saed_calibration_factor = k_input
                    st.session_state.saed_step = 2
                    st.success(f"✅ 标定系数 = {k_input:.6f} nm⁻¹/px")
            else:
                st.caption("在图上画一条线覆盖标尺，输入标尺数值 (nm⁻¹/px)")
                img = st.session_state.get('saed_current_image')
                if img is not None:
                    h, w = img.shape[:2]
                    max_display_size = 400
                    scale_display = min(1.0, max_display_size / max(w, h))
                    display_width = int(w * scale_display)
                    display_height = int(h * scale_display)
                    if len(img.shape) == 2:
                        pil_img = Image.fromarray(img, mode='L').convert('RGB')
                    else:
                        pil_img = Image.fromarray(img, mode='RGB')
                    pil_img = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=2,
                        stroke_color="#FF0000",
                        background_image=pil_img,
                        update_streamlit=True,
                        drawing_mode="line",
                        point_display_radius=0,
                        key="saed_scale_canvas",
                        width=display_width,
                        height=display_height,
                    )
                    if canvas_result and canvas_result.json_data:
                        objs = canvas_result.json_data.get("objects", [])
                        if objs:
                            last_line = objs[-1]
                            if last_line.get("type") == "line":
                                cx1 = float(last_line.get("x1", 0))
                                cy1 = float(last_line.get("y1", 0))
                                cx2 = float(last_line.get("x2", 0))
                                cy2 = float(last_line.get("y2", 0))
                                scale_x = w / display_width
                                scale_y = h / display_height
                                x1_orig = cx1 * scale_x
                                x2_orig = cx2 * scale_x
                                pixel_distance = abs(x2_orig - x1_orig)
                                st.info(f"像素距离（原图）: {pixel_distance:.2f} px")
                                value_1_per_nm = st.number_input(
                                    "标尺 (nm⁻¹)",
                                    min_value=0.001,
                                    value=5.0,
                                    step=0.1,
                                    key="saed_scale_1_per_nm",
                                )
                                if st.button("✅ 设为标定系数", type="primary", key="saed_confirm_k"):
                                    if pixel_distance > 1e-6:
                                        k_cal = value_1_per_nm / pixel_distance
                                        st.session_state.saed_calibration_factor = k_cal
                                        st.session_state.saed_step = 2
                                        st.success(f"✅ 标定系数 = {k_cal:.6f} nm⁻¹/px")
                                        st.rerun()
                                    else:
                                        st.warning("请先画一条有效线")
            
            # ========== 工具栏 ==========
            center_mode = st.radio(
                "中心检测模式",
                ["自动检测（中心最近亮斑）", "手动选择"],
                horizontal=True,
                key="saed_center_mode",
                help="选择透射斑中心检测方式"
            )
            
            # ========== 中心选择（如果选择手动模式） ==========
            if center_mode == "手动选择":
                st.subheader("🎯 手动选择中心")
                # 转换为PIL Image用于canvas
                if len(image.shape) == 2:
                    image_pil = Image.fromarray(image, mode='L').convert('RGB')
                else:
                    image_pil = Image.fromarray(image, mode='RGB')
                
                h, w = image.shape[:2]
                canvas_width = min(800, w)
                canvas_height = int(h * (canvas_width / w))
                
                from streamlit_drawable_canvas import st_canvas
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=image_pil,
                    update_streamlit=True,
                    drawing_mode="point",
                    point_display_radius=5,
                    key="saed_center_canvas",
                    width=canvas_width,
                    height=canvas_height,
                )
                
                # 处理点击事件
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    if objects:
                        last_point = objects[-1]
                        if last_point.get("type") == "circle":
                            canvas_x = last_point.get("x", 0)
                            canvas_y = last_point.get("y", 0)
                            # 转换到原图坐标
                            scale_x = w / canvas_width
                            scale_y = h / canvas_height
                            x = int(canvas_x * scale_x)
                            y = int(canvas_y * scale_y)
                            st.session_state.saed_manual_center_x = x
                            st.session_state.saed_manual_center_y = y
                            st.success(f"✅ 已选择中心位置: ({x}, {y})")
            
            # 像素比例换算在主工作区不再显示，已在侧边栏显示
            # 从session_state读取calibration_factor
            calibration_factor = st.session_state.get("saed_calibration_factor", None)
            if calibration_factor and calibration_factor > 0 and st.session_state.get("saed_step", 0) < 3:
                st.session_state.saed_step = 3

            # 构建配置
            config = {
                'calibration_factor': calibration_factor,
                'mask_radius': mask_radius,
                'min_sigma': min_sigma,
                'max_sigma': max_sigma,
                'threshold': threshold,
                'distance_group_threshold_pix': float(distance_group_threshold_pix),
            }
            
            # 如果选择手动中心，添加手动中心坐标
            if center_mode == "手动选择":
                manual_x = st.session_state.get("saed_manual_center_x", 0)
                manual_y = st.session_state.get("saed_manual_center_y", 0)
                if manual_x > 0 or manual_y > 0:
                    config['manual_center'] = (manual_x, manual_y)
            
            # 执行分析按钮（一次点击立即有反馈）
            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                # 立刻给出提示，避免误以为没有响应
                st.toast("⏳ 已开始分析 SAED 图像，请稍候...", icon="⏳")
                with st.status("🔄 正在分析图像...", expanded=True) as status:
                    try:
                        result = analyzer.analyze_single_image(tmp_path, config=config)
                        status.update(label="✅ 分析完成！", state="complete")
                        
                        # 保存结果到 session_state
                        st.session_state.saed_last_result = result
                        st.session_state.saed_last_image_path = tmp_path
                        if result.get('status') == 'success':
                            st.session_state.saed_step = 4
                        
                        # Toast 消息
                        if result.get('status') == 'success':
                            num_spots = len(result.get('spots', []))
                            st.toast(f"✅ 分析完成！检测到 {num_spots} 个衍射斑点", icon="✅")
                        else:
                            error_msg = result.get('error_message', '未知错误')
                            st.toast(f"❌ 分析失败: {error_msg}", icon="❌")
                    
                    except Exception as e:
                        status.update(label=f"❌ 分析失败: {str(e)}", state="error")
                        st.toast(f"❌ 分析失败: {str(e)}", icon="❌")
                        st.session_state.saed_last_result = None
            
            # 显示结果
            if 'saed_last_result' in st.session_state and st.session_state.saed_last_result is not None:
                result = st.session_state.saed_last_result
                
                if result.get('status') == 'success':
                    # ========== 顶部状态栏 ==========
                    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
                    with col_status1:
                        st.info(f"📄 **当前文件**: {uploaded_file.name}")
                    with col_status2:
                        center = result.get('center', (0, 0))
                        st.info(f"🎯 **中心位置**: ({center[0]:.1f}, {center[1]:.1f})")
                    with col_status3:
                        is_calibrated = result.get('is_calibrated', False)
                        calib_status = "✅ 已标定" if is_calibrated else "⚠️ 未标定"
                        st.info(f"📏 **标定状态**: {calib_status}")
                    with col_status4:
                        processing_time = result.get('processing_time', 0)
                        st.info(f"⏱️ **处理时间**: {processing_time:.2f} 秒")
                    
                    st.divider()
                    
                    # ========== 关键指标 ==========
                    st.subheader("📊 关键指标")
                    spots_list = result.get('spots', [])
                    nearest_ring_spots = result.get('nearest_ring_spots', [])
                    num_spots = len(spots_list)
                    num_ring_spots = len(nearest_ring_spots)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("检测到的斑点", num_spots)
                    with col2:
                        st.metric("最近邻一圈斑点", num_ring_spots)
                    with col3:
                        calibration_factor = result.get('calibration_factor', 0)
                        st.metric("标定系数", f"{calibration_factor:.6f} nm⁻¹/px")
                    with col4:
                        if num_ring_spots > 0:
                            avg_d = np.mean([s['d_spacing_nm'] for s in nearest_ring_spots if s['d_spacing_nm'] > 0])
                            st.metric("平均 d-spacing", f"{avg_d:.4f} nm" if avg_d > 0 else "N/A")
                    
                    st.divider()
                    
                    # ========== 可视化：带标记的图像 ==========
                    st.subheader("🔍 检测结果可视化")
                    
                    # 在图像上绘制中心点和斑点
                    if image is not None:
                        viz_image = image.copy()
                        if len(viz_image.shape) == 2:
                            viz_image = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)
                        
                        center = result.get('center', (0, 0))
                        center_x, center_y = int(center[0]), int(center[1])
                        
                        # 绘制中心点（红色十字）
                        cv2.line(viz_image, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
                        cv2.line(viz_image, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
                        cv2.circle(viz_image, (center_x, center_y), mask_radius, (0, 0, 255), 2)
                        
                        # 绘制最近邻一圈斑点（绿色圆点，文字用分组标识 a1,a2...，字体加粗放大）
                        for idx, spot in enumerate(nearest_ring_spots):
                            x, y = int(spot['x']), int(spot['y'])
                            cv2.circle(viz_image, (x, y), 5, (0, 255, 0), -1)
                            cv2.circle(viz_image, (x, y), 8, (0, 255, 0), 1)
                            
                            # 文本标注使用分组标识（如 a1,a2,b1），若无分组则退回序号
                            group_label = spot.get('group_label', '')
                            member_idx = spot.get('group_member_index', None)
                            if group_label and member_idx is not None:
                                text_label = f"{group_label}{member_idx}"
                            elif group_label:
                                text_label = group_label
                            else:
                                text_label = str(idx + 1)
                            
                            # 绘制到中心的连线
                            cv2.line(viz_image, (center_x, center_y), (x, y), (255, 255, 0), 1)
                            
                            # 计算文字尺寸，先画深色背景块，再画亮色文字，使 a,b,c 更显眼
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.9
                            thickness = 2
                            (text_w, text_h), baseline = cv2.getTextSize(text_label, font, font_scale, thickness)
                            text_x = x + 10
                            text_y = y - 10
                            # 背景矩形
                            cv2.rectangle(
                                viz_image,
                                (text_x - 4, text_y - text_h - 4),
                                (text_x + text_w + 4, text_y + baseline + 4),
                                (0, 0, 0),
                                -1,
                            )
                            # 前景文字（亮绿色）
                            cv2.putText(
                                viz_image,
                                text_label,
                                (text_x, text_y),
                                font,
                                font_scale,
                                (0, 255, 0),
                                thickness,
                                lineType=cv2.LINE_AA,
                            )
                        
                        st.image(viz_image, caption="检测结果：红色十字=中心，绿色圆点=最近邻一圈斑点（按 a,b,c 分组）")
                    
                    st.divider()
                    
                    # ========== 最近邻一圈斑点数据 ==========
                    st.subheader("📋 最近邻一圈斑点数据")
                    
                    distance_groups = result.get('distance_groups', [])
                    
                    if nearest_ring_spots:
                        # 先显示距离分组汇总表（a,b,c...）
                        if distance_groups:
                            st.markdown("**距离分组汇总**")
                            group_rows = []
                            for g in distance_groups:
                                label = g.get('label', '')
                                mean_r = g.get('mean_r_pix', 0.0)
                                mean_d = g.get('mean_d_spacing_nm', 0.0)
                                mean_angle = g.get('mean_angle_deg', 0.0)
                                n_members = len(g.get('spots', []))
                                group_rows.append({
                                    '分组': label,
                                    '成员数量': n_members,
                                    '平均距离 (px)': mean_r,
                                    '平均 d-spacing (nm)': mean_d if mean_d > 0 else 'N/A',
                                    '平均角度 (度)': mean_angle,
                                })
                            
                            group_df = pd.DataFrame(group_rows)
                            st.dataframe(group_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("**最近邻一圈斑点明细**")
                        # 再显示最近邻一圈斑点明细表（带分组标签）
                        df_data = []
                        for idx, spot in enumerate(nearest_ring_spots, 1):
                            group_label = spot.get('group_label', '')
                            member_idx = spot.get('group_member_index', None)
                            if group_label and member_idx is not None:
                                group_name = f"{group_label}{member_idx}"
                            elif group_label:
                                group_name = group_label
                            else:
                                group_name = "-"
                            
                            df_data.append({
                                '序号': idx,
                                '分组标识': group_name,
                                'x (px)': spot.get('x', 0),
                                'y (px)': spot.get('y', 0),
                                '距离 (px)': spot.get('r_pix', 0),
                                'd-spacing (nm)': spot.get('d_spacing_nm', 0) if spot.get('d_spacing_nm', 0) > 0 else 'N/A',
                                '角度 (度)': spot.get('angle_deg', 0),
                                '强度': spot.get('intensity', 0),
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        st.divider()
                        
                        # ========== 误差分析 ==========
                        st.subheader("📊 误差分析")
                        
                        # 使用分组标签 a,b,c... 来展示误差分析
                        distance_groups = result.get('distance_groups', [])
                        
                        # 角度差：对每一对分组，使用组内所有连线之间的“最小夹角”
                        angle_pairs = []
                        angle_differences = []
                        if distance_groups and len(distance_groups) >= 2:
                            for i in range(len(distance_groups)):
                                for j in range(i + 1, len(distance_groups)):
                                    g1 = distance_groups[i]
                                    g2 = distance_groups[j]
                                    label1 = g1.get('label', '')
                                    label2 = g2.get('label', '')
                                    spots1 = g1.get('spots', [])
                                    spots2 = g2.get('spots', [])
                                    
                                    # 取两组中所有斑点的角度
                                    angles1 = [float(s.get('angle_deg', 0.0)) for s in spots1]
                                    angles2 = [float(s.get('angle_deg', 0.0)) for s in spots2]
                                    
                                    if not angles1 or not angles2:
                                        continue
                                    
                                    min_angle_diff = None
                                    for a1 in angles1:
                                        for a2 in angles2:
                                            diff = abs(a1 - a2)
                                            # 线与线的夹角取最小值（0-180 范围）
                                            diff = min(diff, 360.0 - diff)
                                            if (min_angle_diff is None) or (diff < min_angle_diff):
                                                min_angle_diff = diff
                                    
                                    if min_angle_diff is not None:
                                        angle_pairs.append(f"{label1}-{label2}")
                                        angle_differences.append(round(min_angle_diff, 2))
                            
                        # 距离比值：任意两个不同分组之间，形成 a-b, a-c, b-c 这种组对
                        ratio_pairs = []
                        distance_ratios = []
                        if distance_groups and len(distance_groups) >= 2:
                            for i in range(len(distance_groups)):
                                for j in range(i + 1, len(distance_groups)):
                                    g1 = distance_groups[i]
                                    g2 = distance_groups[j]
                                    label1 = g1.get('label', '')
                                    label2 = g2.get('label', '')
                                    r1 = float(g1.get('mean_r_pix', 0.0))
                                    r2 = float(g2.get('mean_r_pix', 0.0))
                                    if r1 > 1e-5 and r2 > 1e-5:
                                        r_min = min(r1, r2)
                                        r_max = max(r1, r2)
                                        ratio = r_min / r_max
                                        ratio_pairs.append(f"{label1}-{label2}")
                                        distance_ratios.append(round(ratio, 4))
                        
                        if distance_groups and len(distance_groups) >= 2 and angle_differences:
                            col_err1, col_err2 = st.columns(2)
                            
                            with col_err1:
                                st.markdown("**角度差分析**")
                                angle_df = pd.DataFrame({
                                    '组对': angle_pairs,
                                    '角度差 (度)': angle_differences,
                                })
                                st.dataframe(angle_df, use_container_width=True, hide_index=True)
                                
                                if len(angle_differences) > 0:
                                    avg_angle_diff = np.mean(angle_differences)
                                    std_angle_diff = np.std(angle_differences)
                                    min_angle_diff = np.min(angle_differences)
                                    st.metric("平均角度差", f"{avg_angle_diff:.2f}°")
                                    st.metric("角度差标准差", f"{std_angle_diff:.2f}°")
                                    st.metric("最小角度差", f"{min_angle_diff:.2f}°")
                            
                            with col_err2:
                                st.markdown("**距离比值分析**")
                                if distance_ratios:
                                    ratio_df = pd.DataFrame({
                                        '组对': ratio_pairs,
                                        '距离比值': distance_ratios,
                                    })
                                    st.dataframe(ratio_df, use_container_width=True, hide_index=True)
                                    
                                    avg_ratio = np.mean(distance_ratios)
                                    std_ratio = np.std(distance_ratios)
                                    min_ratio = np.min(distance_ratios)
                                    st.metric("平均距离比值", f"{avg_ratio:.4f}")
                                    st.metric("距离比值标准差", f"{std_ratio:.4f}")
                                    st.metric("最小距离比值", f"{min_ratio:.4f}")
                                else:
                                    st.info("距离比值数据不足")
                        else:
                            st.warning("⚠️ 最近邻一圈斑点数量不足，无法进行误差分析（需要至少2个分组）")
                    else:
                        st.warning("⚠️ 未检测到最近邻一圈斑点")
                    
                    st.divider()
                    
                    # ========== 晶面参考数据 ==========
                    st.subheader("📚 晶面参考数据")
                    
                    with st.expander("🔬 FCC 结构（面心立方）晶面数据", expanded=True):
                        st.markdown("""
                        ### FCC 结构晶面间距比值（以 d₁₀₀ = 1 为基准）

                        对于立方晶系，晶面间距公式为：

                        $$d_{hkl} = \\frac{a}{\\sqrt{h^2 + k^2 + l^2}}$$

                        其中 $a$ 是晶格常数，$(hkl)$ 是米勒指数。

                        | 晶面 | 米勒指数 | $\\sqrt{h^2 + k^2 + l^2}$ | 相对间距比值 | 说明 |
                        |---|---|---|---|---|
                        | (100) | (1, 0, 0) | $\\sqrt{1} = 1$ | **1.000** | 基准 |
                        | (110) | (1, 1, 0) | $\\sqrt{2} \\approx 1.414$ | **0.7071** | $d_{110}/d_{100} = 1/\\sqrt{2}$ |
                        | (111) | (1, 1, 1) | $\\sqrt{3} \\approx 1.732$ | **0.5774** | $d_{111}/d_{100} = 1/\\sqrt{3}$ |

                        **距离比值关系**：
                        - $d_{100} : d_{110} : d_{111} = 1 : 0.7071 : 0.5774$
                        - $d_{110}/d_{111} = \\sqrt{2}/\\sqrt{3} \\approx 0.8165$

                        ### FCC 结构晶面间夹角

                        晶面法向量之间的夹角计算公式：

                        $$\\cos\\theta = \\frac{h_1h_2 + k_1k_2 + l_1l_2}{\\sqrt{h_1^2 + k_1^2 + l_1^2} \\cdot \\sqrt{h_2^2 + k_2^2 + l_2^2}}$$

                        | 晶面对 | 计算 | 理论夹角 |
                        |---|---|---|
                        | (100) 与 (110) | $\\cos\\theta = \\frac{1}{\\sqrt{2}} \\approx 0.7071$ | **45.00°** |
                        | (100) 与 (111) | $\\cos\\theta = \\frac{1}{\\sqrt{3}} \\approx 0.5774$ | **54.74°** |
                        | (110) 与 (111) | $\\cos\\theta = \\frac{2}{\\sqrt{6}} \\approx 0.8165$ | **35.26°** |

                        **应用说明**：
                        - 在 SAED 图中，斑点到中心的径向距离比应接近晶面间距比值的倒数
                        - 斑点之间的角度差可用于验证晶面标定的准确性
                        - 若实测比值和角度与理论值相符，则标定正确
                        """)
                    
                    with st.expander("💎 A4 结构（金刚石立方）晶面数据", expanded=True):
                        st.markdown("""
### A4 结构晶面间距比值（以 d₁₀₀ = 1 为基准）

A4 结构（金刚石立方结构）属于立方晶系，晶面间距公式与 FCC 相同：

$$d_{hkl} = \\frac{a}{\\sqrt{h^2 + k^2 + l^2}}$$

其中 $a$ 是晶格常数，$(hkl)$ 是米勒指数。

| 晶面 | 米勒指数 | $\\sqrt{h^2 + k^2 + l^2}$ | 相对间距比值 | 说明 |
|---|---|---|---|---|
| (100) | (1, 0, 0) | $\\sqrt{1} = 1$ | **1.000** | 基准 |
| (110) | (1, 1, 0) | $\\sqrt{2} \\approx 1.414$ | **0.7071** | $d_{110}/d_{100} = 1/\\sqrt{2}$ |
| (111) | (1, 1, 1) | $\\sqrt{3} \\approx 1.732$ | **0.5774** | $d_{111}/d_{100} = 1/\\sqrt{3}$ |

**距离比值关系**：
- $d_{100} : d_{110} : d_{111} = 1 : 0.7071 : 0.5774$
- $d_{110}/d_{111} = \\sqrt{2}/\\sqrt{3} \\approx 0.8165$

> **注意**：A4 结构的晶面间距比值与 FCC 相同，因为两者都属于立方晶系。但 A4 结构的原子排列更复杂（每个晶胞有 8 个原子），在衍射强度上可能有所不同。

### A4 结构晶面间夹角

由于 A4 结构也是立方晶系，晶面间夹角与 FCC 相同：

| 晶面对 | 计算 | 理论夹角 |
|---|---|---|
| (100) 与 (110) | $\\cos\\theta = \\frac{1}{\\sqrt{2}} \\approx 0.7071$ | **45.00°** |
| (100) 与 (111) | $\\cos\\theta = \\frac{1}{\\sqrt{3}} \\approx 0.5774$ | **54.74°** |
| (110) 与 (111) | $\\cos\\theta = \\frac{2}{\\sqrt{6}} \\approx 0.8165$ | **35.26°** |

**应用说明**：
- A4 结构常见于硅（Si）、锗（Ge）、金刚石（C）等材料
- 在 SAED 分析中，可通过距离比值和角度差来识别和验证 A4 结构
- 虽然间距比值与 FCC 相同，但衍射强度分布可能不同，需要结合强度信息进行判断
                            """)
                    
                    # ========== 可视化图表 ==========
                    if nearest_ring_spots:
                        st.subheader("📈 可视化分析")
                        
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            st.markdown("**最近邻一圈 d-spacing 分布**")
                            d_values = [s['d_spacing_nm'] for s in nearest_ring_spots if s['d_spacing_nm'] > 0]
                            if d_values:
                                fig_d = go.Figure(data=[go.Histogram(x=d_values, nbinsx=min(20, len(d_values)))])
                                fig_d.update_layout(
                                    xaxis_title="d-spacing (nm)",
                                    yaxis_title="频数",
                                    height=400,
                                )
                                st.plotly_chart(fig_d, use_container_width=True)
                            else:
                                st.info("未标定，无法显示 d-spacing 分布")
                        
                        with col_viz2:
                            st.markdown("**角度差分布**")
                            if angle_differences:
                                fig_angle_diff = go.Figure(data=[go.Histogram(x=angle_differences, nbinsx=min(20, len(angle_differences)))])
                                fig_angle_diff.update_layout(
                                    xaxis_title="角度差 (度)",
                                    yaxis_title="频数",
                                    height=400,
                                )
                                st.plotly_chart(fig_angle_diff, use_container_width=True)
                            else:
                                st.info("角度差数据不足")
                        
                        # ========== 极坐标图（仅显示最近邻一圈） ==========
                        st.markdown("**最近邻一圈斑点极坐标分布图**")
                        if nearest_ring_spots:
                            # 创建极坐标散点图
                            fig_polar = go.Figure()
                            
                            # 为了让色标刻度更清晰，只创建一条带 colorbar 的隐形 trace，
                            # 实际斑点使用统一的颜色，由 hover 显示强度数值。
                            intensities = [float(s['intensity']) for s in nearest_ring_spots]
                            if intensities:
                                fig_polar.add_trace(go.Scatterpolar(
                                    r=[0],
                                    theta=[0],
                                    mode='markers',
                                    marker=dict(
                                        size=0.1,
                                        color=intensities,
                                        colorscale='Hot',
                                        showscale=True,
                                        colorbar=dict(
                                            title=dict(text="强度", side='right'),
                                            tickfont=dict(size=10),
                                            len=0.8,
                                        ),
                                    ),
                                    hoverinfo='skip',
                                    showlegend=False,
                                ))
                            
                            for idx, spot in enumerate(nearest_ring_spots):
                                angle_deg = spot['angle_deg']
                                r_value = spot['r_pix']
                                intensity = spot['intensity']
                                
                                fig_polar.add_trace(go.Scatterpolar(
                                    r=[r_value],
                                    theta=[angle_deg],
                                    mode='markers+text',
                                    text=[str(idx + 1)],
                                    textposition='top center',
                                    marker=dict(size=15, color='orange'),
                                    textfont=dict(size=12, color='white'),
                                    name=f"斑点 {idx + 1}",
                                    hovertemplate=f'斑点 {idx + 1}<br>d-spacing: {spot["d_spacing_nm"]:.4f} nm<br>角度: %{{theta}}°<br>距离: %{{r}} px<br>强度: {intensity:.1f}<extra></extra>',
                                ))
                            
                            max_r = max([s['r_pix'] for s in nearest_ring_spots], default=100)
                            fig_polar.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, max_r * 1.2]),
                                    angularaxis=dict(visible=True),
                                ),
                                showlegend=False,
                                height=500,
                            )
                            st.plotly_chart(fig_polar, use_container_width=True)
                    else:
                        st.warning("⚠️ 未检测到最近邻一圈斑点")
                
                else:
                    # 错误状态
                    st.error(f"❌ 分析失败: {result.get('error_message', '未知错误')}")
                    if 'error_message' in result:
                        with st.expander("错误详情"):
                            st.code(result['error_message'])
        
        finally:
            # 清理临时文件
            pass

