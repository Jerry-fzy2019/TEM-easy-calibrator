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
import traceback
import urllib.request
from pathlib import Path


STREAMLIT_PORT = 8502
SERVER_FLAG = "--tem-streamlit-server"
streamlit_process: subprocess.Popen | None = None
log_handle = None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def base_path() -> Path:
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def streamlit_app_path() -> Path:
    return base_path() / "src" / "ui_streamlit" / "app.py"


def app_data_path() -> Path:
    if sys.platform.startswith("win"):
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / "TEM Easy Calibrator"
    return Path.home() / ".tem-easy-calibrator"


def log_path() -> Path:
    return app_data_path() / "logs" / "launcher.log"


def write_log(message: str) -> None:
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def configure_streamlit() -> None:
    os.environ["STREAMLIT_SERVER_PORT"] = str(STREAMLIT_PORT)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_THEME_BASE"] = "light"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"


def apply_streamlit_runtime_config() -> None:
    """Set Streamlit options directly for embedded/frozen startup."""
    from streamlit import config as st_config

    options = {
        "server.port": STREAMLIT_PORT,
        "server.address": "localhost",
        "server.headless": True,
        "server.fileWatcherType": "none",
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
        "theme.base": "light",
    }
    for key, value in options.items():
        st_config.set_option(key, value)


def run_streamlit_server() -> None:
    """Run Streamlit inside the current process.

    This is used by the child process created from a frozen PyInstaller app.
    """
    try:
        configure_streamlit()
        app_path = streamlit_app_path()
        write_log(f"Starting Streamlit server. frozen={is_frozen()} app_path={app_path}")
        if not app_path.exists():
            raise FileNotFoundError(f"Streamlit app was not found: {app_path}")

        apply_streamlit_runtime_config()
        write_log("Streamlit runtime config applied.")

        from streamlit.web import bootstrap

        write_log("Streamlit bootstrap imported.")
        bootstrap.run(str(app_path), None, [], {})
    except Exception:
        write_log("Streamlit server failed:\n" + traceback.format_exc())
        raise


def start_streamlit() -> None:
    """Start the Streamlit server as a child process."""
    global log_handle, streamlit_process
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

    if is_frozen():
        log_path().parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path().open("a", encoding="utf-8")
        popen_kwargs["stdout"] = log_handle
        popen_kwargs["stderr"] = subprocess.STDOUT

    write_log(f"Launching Streamlit process: {cmd}")
    streamlit_process = subprocess.Popen(cmd, **popen_kwargs)


def wait_until_ready(timeout_seconds: float = 60.0) -> bool:
    url = f"http://localhost:{STREAMLIT_PORT}/_stcore/health"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).close()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def stop_streamlit() -> None:
    global log_handle, streamlit_process
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
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
            log_handle = None


def main() -> None:
    if SERVER_FLAG in sys.argv:
        run_streamlit_server()
        return

    write_log(f"Launcher started. frozen={is_frozen()} executable={sys.executable}")
    start_streamlit()

    if not wait_until_ready():
        stop_streamlit()
        write_log("Streamlit server startup timed out.")
        print(f"Error: Streamlit server startup timed out. See log: {log_path()}")
        return

    try:
        import webview

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
