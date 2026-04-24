"""
TEM Calibrator - 桌面应用主入口
使用 pywebview 嵌入 Streamlit 应用
"""

import sys
import threading
import time
import subprocess
from pathlib import Path
import webview

# 全局变量：保存 Streamlit 子进程句柄，方便在窗口关闭时结束进程
streamlit_process = None


def start_streamlit():
    """在后台线程中启动 Streamlit 服务器"""
    global streamlit_process
    # 获取项目根目录
    base_path = Path(__file__).parent
    app_path = base_path / "src" / "ui_streamlit" / "app.py"
    
    # 设置 Streamlit 配置
    import os
    os.environ['STREAMLIT_SERVER_PORT'] = '8502'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_THEME_BASE'] = 'light'
    
    # 启动 Streamlit
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(app_path),
        "--server.port=8502",
        "--server.address=localhost",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    
    # 设置工作目录为项目根目录，使用 Popen 启动，后续可以主动关闭
    streamlit_process = subprocess.Popen(cmd, cwd=str(base_path))

def main():
    """主函数"""
    # 在后台线程启动 Streamlit
    streamlit_thread = threading.Thread(target=start_streamlit, daemon=True)
    streamlit_thread.start()
    
    # 等待 Streamlit 启动（最多等待 15 秒）
    import urllib.request
    max_wait = 15
    waited = 0
    while waited < max_wait:
        try:
            # 注意：这里的端口要和上面启动 Streamlit 时保持一致（8502）
            urllib.request.urlopen('http://localhost:8502', timeout=1)
            break
        except:
            time.sleep(0.5)
            waited += 0.5
    
    if waited >= max_wait:
        print("错误: Streamlit 服务器启动超时")
        return
    
    # 创建 webview 窗口（指向 8502 端口，确保使用当前代码）
    window = webview.create_window(
        title='TEM 自动化分析系统 v2.0',
        url='http://localhost:8502',
        width=1400,
        height=900,
        min_size=(1200, 800),
        resizable=True,
    )
    
    try:
        # 启动 webview（阻塞，直到用户关闭窗口）
        webview.start(debug=False)
    finally:
        # 窗口关闭后，尝试关闭 Streamlit 子进程，释放 8502 端口
        global streamlit_process
        if streamlit_process is not None:
            try:
                streamlit_process.terminate()
                streamlit_process.wait(timeout=5)
            except Exception:
                pass
            streamlit_process = None

if __name__ == '__main__':
    main()
