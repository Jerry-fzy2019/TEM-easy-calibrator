"""
图像 I/O 相关封装。

实现来源于旧版 `tem_calibration.image_processor.load_image`，
在此作为 core 层统一的图像加载入口，不涉及任何 UI 逻辑。
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .config import (
    CONTRAST_ENHANCEMENT_ALPHA,
    CONTRAST_ENHANCEMENT_BETA,
    NOISE_REDUCTION_KERNEL,
)

try:
    import tifffile

    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


def load_image(image_path: str | Path | object) -> Optional[np.ndarray]:
    """加载图像文件（支持多种格式，特别是 TIF）。

    返回灰度 `uint8` 数组，失败时返回 ``None``。
    """
    if image_path is None:
        return None

    # 处理路径：如果是 Path 对象，直接使用；如果是其他对象且有 name 属性（如 UploadFile），使用 name
    if isinstance(image_path, Path):
        path = image_path
    elif hasattr(image_path, "name") and not isinstance(image_path, (str, Path)):
        # 兼容 UploadFile 等对象，但不影响 Path 和 str
        path = Path(str(image_path.name))
    else:
        path = Path(str(image_path))

    if not path.exists():
        print(f"[DEBUG] load_image: 文件不存在: {path}")
        return None

    ext = path.suffix.lower()
    print(f"[DEBUG] load_image: 文件扩展名: {ext}")

    try:
        # 优先使用 tifffile 处理 TIF
        if ext in [".tif", ".tiff"]:
            print("[DEBUG] load_image: 检测到 TIF 格式，尝试使用 tifffile 加载...")
            if TIFFFILE_AVAILABLE:
                try:
                    image = tifffile.imread(str(path))
                    print(
                        f"[DEBUG] load_image: tifffile 加载成功，原始形状: {image.shape}, 类型: {image.dtype}"
                    )
                    if len(image.shape) == 3:
                        print("[DEBUG] load_image: RGB 图像，转换为灰度")
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    elif len(image.shape) != 2:
                        print("[DEBUG] load_image: 多通道图像，取第一个通道")
                        image = image[:, :, 0]
                    result = image.astype(np.uint8)
                    print(
                        f"[DEBUG] load_image: 最终图像形状: {result.shape}, 类型: {result.dtype}"
                    )
                    return result
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[DEBUG] load_image: tifffile 加载失败，尝试其他方法: {e}"
                    )

            # 回退到 PIL
            print("[DEBUG] load_image: 尝试使用 PIL 加载 TIF...")
            try:
                pil_image = Image.open(path)
                print(
                    f"[DEBUG] load_image: PIL 加载成功，模式: {pil_image.mode}, 尺寸: {pil_image.size}"
                )
                if pil_image.mode != "L":
                    print(
                        f"[DEBUG] load_image: 转换模式 {pil_image.mode} -> L (灰度)"
                    )
                    pil_image = pil_image.convert("L")
                image_np = np.array(pil_image).astype(np.uint8)
                print(
                    f"[DEBUG] load_image: PIL 最终图像形状: {image_np.shape}, 类型: {image_np.dtype}"
                )
                return image_np
            except Exception as e:  # noqa: BLE001
                print(f"[DEBUG] load_image: PIL 加载 TIF 失败: {e}")

        # 其他格式使用 OpenCV
        print("[DEBUG] load_image: 尝试使用 OpenCV 加载...")
        image_cv = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image_cv is not None:
            print(
                f"[DEBUG] load_image: OpenCV 加载成功，形状: {image_cv.shape}, 类型: {image_cv.dtype}"
            )
            return image_cv
        print("[DEBUG] load_image: OpenCV 加载返回 None")

        # 最后再尝试 PIL
        print("[DEBUG] load_image: 尝试使用 PIL 作为最后手段...")
        try:
            pil_image = Image.open(path)
            print(f"[DEBUG] load_image: PIL 加载成功，模式: {pil_image.mode}")
            if pil_image.mode != "L":
                pil_image = pil_image.convert("L")
            image_np = np.array(pil_image).astype(np.uint8)
            print(
                f"[DEBUG] load_image: PIL 最终图像形状: {image_np.shape}, 类型: {image_np.dtype}"
            )
            return image_np
        except Exception as e:  # noqa: BLE001
            print(f"[DEBUG] load_image: PIL 加载失败: {e}")
            return None

    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"[DEBUG] load_image: 加载图像失败: {e}")
        print(f"[DEBUG] load_image: 异常堆栈:\n{traceback.format_exc()}")
        return None


