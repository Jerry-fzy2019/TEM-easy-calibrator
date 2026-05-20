"""Desktop entry point for TEM Easy Calibrator.

This file can run from source or from a PyInstaller bundle. In source mode it
starts Streamlit with the current Python interpreter. In frozen mode it launches
a second copy of the bundled executable as the Streamlit server process, then
opens the local app with pywebview.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import webview


STREAMLIT_PORT = 8502
SERVER_FLAG = "--tem-streamlit-server"
streamlit_process: subprocess.Popen | None = None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def base_path() -> Path:
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def streamlit_app_path() -> Path:
    return base_path() / "src" / "ui_streamlit" / "app.py"


def configure_streamlit() -> None:
    os.environ["STREAMLIT_SERVER_PORT"] = str(STREAMLIT_PORT)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_THEME_BASE"] = "light"


def run_streamlit_server() -> None:
    """Run Streamlit inside the current process.

    This is used by the child process created from a frozen PyInstaller app.
    """
    configure_streamlit()
    from streamlit.web import bootstrap

    bootstrap.run(str(streamlit_app_path()), False, [], {})


def start_streamlit() -> None:
    """Start the Streamlit server as a child process."""
    global streamlit_process
    configure_streamlit()

    if is_frozen():
        cmd = [sys.executable, SERVER_FLAG]
    else:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(streamlit_app_path()),
            f"--server.port={STREAMLIT_PORT}",
            "--server.address=localhost",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]

    popen_kwargs: dict = {"cwd": str(base_path())}
    if sys.platform.startswith("win") and hasattr(subprocess, "CREATE_NO_WINDOW"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    streamlit_process = subprocess.Popen(cmd, **popen_kwargs)


def wait_until_ready(timeout_seconds: float = 20.0) -> bool:
    url = f"http://localhost:{STREAMLIT_PORT}"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).close()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def stop_streamlit() -> None:
    global streamlit_process
    if streamlit_process is None:
        return

    try:
        streamlit_process.terminate()
        streamlit_process.wait(timeout=5)
    except Exception:
        try:
            streamlit_process.kill()
        except Exception:
            pass
    finally:
        streamlit_process = None


def main() -> None:
    if SERVER_FLAG in sys.argv:
        run_streamlit_server()
        return

    start_streamlit()

    if not wait_until_ready():
        stop_streamlit()
        print("Error: Streamlit server startup timed out.")
        return

    try:
        webview.create_window(
            title="TEM Easy Calibrator",
            url=f"http://localhost:{STREAMLIT_PORT}",
            width=1400,
            height=900,
            min_size=(1200, 800),
            resizable=True,
        )
        webview.start(debug=False)
    finally:
        stop_streamlit()


if __name__ == "__main__":
    main()
