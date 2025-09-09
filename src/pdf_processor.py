"""
PDF processing utilities for CAD line extraction system.
"""

from pathlib import Path
from typing import List, Tuple
import logging

import numpy as np
import fitz

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exceptions import PDFProcessingError
from memory_manager import MemoryManager, optimize_image_for_processing

try:
    from config import PERFORMANCE_CONFIG
except Exception:
    PERFORMANCE_CONFIG = {"max_image_dimension": 6000}

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF file processing and conversion to images."""

    def __init__(self):
        self.memory_manager = MemoryManager()

    def render_pdf_to_images(self, pdf_path: Path, dpi: int = 600) -> List[Tuple[int, np.ndarray]]:
        """Render PDF pages to high-resolution images with memory optimization.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering (default 600)

        Returns:
            List of (page_index, image_array) tuples
        """
        images: List[Tuple[int, np.ndarray]] = []

        try:
            logger.info(f"PDF 열기: {pdf_path}")
            doc = fitz.open(pdf_path.as_posix())
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            total_pages = len(doc)
            logger.info(f"{total_pages} 페이지 처리 중 (DPI: {dpi})")

            for page_index in range(total_pages):
                try:
                    with self.memory_manager.memory_guard(f"PDF 페이지 {page_index} 렌더링"):
                        page = doc.load_page(page_index)
                        pix = page.get_pixmap(matrix=matrix, alpha=False)
                        img = np.frombuffer(pix.samples, dtype=np.uint8)
                        img = img.reshape(pix.height, pix.width, pix.n)  # H, W, C

                        if pix.n == 4:
                            import cv2
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        elif pix.n == 3:
                            # PyMuPDF gives RGB; convert to BGR for correct saving via cv2.imwrite
                            import cv2
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        # 메모리 최적화: 큰 이미지를 최적화
                        if max(img.shape[:2]) > PERFORMANCE_CONFIG["max_image_dimension"]:
                            logger.debug(f"페이지 {page_index} 이미지 크기 최적화: {img.shape} -> 최적화 적용")
                            img = optimize_image_for_processing(img, PERFORMANCE_CONFIG["max_image_dimension"])

                        images.append((page_index, img))
                        logger.debug(f"페이지 {page_index + 1}/{total_pages} 렌더링 완료: {img.shape}")

                        # 메모리 사용량 체크
                        if self.memory_manager.is_memory_limit_exceeded():
                            logger.warning(".1f")
                            self.memory_manager.force_garbage_collection()

                except Exception as e:
                    logger.warning(f"페이지 {page_index} 렌더링 실패 (건너뜀): {e}")
                    continue

            doc.close()

            if not images:
                raise PDFProcessingError("PDF에서 렌더링된 페이지가 없습니다", str(pdf_path))

            logger.info(f"총 {len(images)} 페이지 렌더링 완료")
            return images

        except FileNotFoundError:
            error_msg = f"PDF 파일을 찾을 수 없습니다: {pdf_path}"
            logger.error(error_msg)
            raise PDFProcessingError(error_msg, str(pdf_path))
        except PermissionError:
            error_msg = f"PDF 파일에 접근 권한이 없습니다: {pdf_path}"
            logger.error(error_msg)
            raise PDFProcessingError(error_msg, str(pdf_path))
        except Exception as e:
            error_msg = f"PDF 처리 중 예상치 못한 오류: {str(e)}"
            logger.error(error_msg)
            if hasattr(e, '__class__') and 'PDFProcessingError' in str(type(e)):
                raise
            raise PDFProcessingError(error_msg, str(pdf_path))

    def validate_pdf(self, pdf_path: Path) -> bool:
        """Validate PDF file integrity.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF is valid, False otherwise
        """
        try:
            doc = fitz.open(pdf_path.as_posix())
            page_count = len(doc)
            doc.close()
            return page_count > 0
        except Exception:
            return False

    def get_pdf_info(self, pdf_path: Path) -> dict:
        """Get PDF file information.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing PDF metadata
        """
        try:
            doc = fitz.open(pdf_path.as_posix())
            info = {
                "page_count": len(doc),
                "file_size": pdf_path.stat().st_size,
                "metadata": doc.metadata
            }
            doc.close()
            return info
        except Exception as e:
            raise PDFProcessingError(f"PDF 정보 읽기 실패: {str(e)}", str(pdf_path))
