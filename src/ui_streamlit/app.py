"""
TEM Calibrator - Streamlit 主入口。
"""

import streamlit as st

st.set_page_config(
    page_title="TEM Calibrator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 主页面内容
st.title("🔬 TEM 自动化分析系统")
st.markdown("""
### 欢迎使用 TEM 自动化分析系统！

这是一个专业的透射电子显微镜（TEM）图像分析工具，提供全自动化的测量和分析功能。

**主要功能模块：**
- 📏 **膜厚校准**: 基于 FFT 的周期性层间距分析
- 🔬 **晶面间距**: 高分辨晶面间距校准
- 📐 **SAED 校准**: 选区衍射标定
- 📊 **漂移校准**: 图像漂移/畸变校准
- 🧪 **污染率**: 污染率表征

请在左侧导航栏选择功能模块开始分析。
""")

# 侧边栏导航
with st.sidebar:
    st.markdown("### 📋 功能模块")
    
    # 使用按钮样式的链接（通过 markdown）
    st.markdown("""
    - [📏 膜厚校准](1_Thickness)
    - [🔬 晶面间距](2_HR_Spacing)
    - [📐 SAED 校准](3_SAED)
    - [📊 漂移校准](4_Drift)
    - [🧪 污染率](5_Contamination)
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ 关于")
    st.info("""
    **TEM 自动化分析系统**
    
    版本: v2.0
    
    支持全自动图像分析，
    无需手动操作即可完成
    测量和校准。
    """)

