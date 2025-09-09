"""
Configuration validation utilities for CAD line extraction system.
Falls back to sane defaults when config.py is not present.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, List
import logging


def _load_config_module():
    """Load user config.py if present; otherwise provide defaults."""
    try:
        from config import (  # type: ignore
            ROOT_DIR as _ROOT_DIR,
            OUTPUT_DIR as _OUTPUT_DIR,
            WEB_DIR as _WEB_DIR,
            PDF_CONFIG as _PDF_CONFIG,
            IMAGE_CONFIG as _IMAGE_CONFIG,
            PERFORMANCE_CONFIG as _PERFORMANCE_CONFIG,
        )
        class _Cfg:
            ROOT_DIR = _ROOT_DIR
            OUTPUT_DIR = _OUTPUT_DIR
            WEB_DIR = _WEB_DIR
            PDF_CONFIG = _PDF_CONFIG
            IMAGE_CONFIG = _IMAGE_CONFIG
            PERFORMANCE_CONFIG = _PERFORMANCE_CONFIG
        return _Cfg
    except Exception:
        class _DefaultCfg:
            ROOT_DIR = Path(__file__).resolve().parent.parent
            OUTPUT_DIR = ROOT_DIR / "outputs" / "lines"
            WEB_DIR = ROOT_DIR
            PDF_CONFIG = {"default_dpi": 600, "fallback_dpi": 300, "max_workers": 1}
            IMAGE_CONFIG = {
                "deskew_angle_search": 7.0,
                "canny_low_threshold": 50,
                "canny_high_threshold": 150,
                "min_absolute_line_length": 10,
            }
            PERFORMANCE_CONFIG = {
                "memory_limit_mb": 2048,
                "processing_timeout_seconds": 300,
            }
        return _DefaultCfg

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration settings and system requirements."""

    @staticmethod
    def validate_paths() -> List[str]:
        """Validate required paths and directories.

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        # Load config or defaults
        cfg = _load_config_module()
        ROOT_DIR = cfg.ROOT_DIR
        OUTPUT_DIR = cfg.OUTPUT_DIR
        WEB_DIR = cfg.WEB_DIR

        dirs_to_check = [
            (ROOT_DIR, "프로젝트 루트 디렉토리"),
            (OUTPUT_DIR, "출력 디렉토리"),
            (WEB_DIR, "웹 디렉토리")
        ]

        for dir_path, description in dirs_to_check:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                if not dir_path.exists():
                    errors.append(f"{description}를 생성할 수 없습니다: {dir_path}")
            except Exception as e:
                errors.append(f"{description} 생성 실패: {str(e)}")

        return errors

    @staticmethod
    def validate_pdf_config() -> List[str]:
        """Validate PDF processing configuration.

        Returns:
            List of validation errors
        """
        errors = []

        PDF_CONFIG = _load_config_module().PDF_CONFIG

        # Validate DPI values
        if not (100 <= PDF_CONFIG["default_dpi"] <= 1200):
            errors.append(f"DPI 값이 범위를 벗어났습니다: {PDF_CONFIG['default_dpi']} (100-1200)")

        if not (100 <= PDF_CONFIG["fallback_dpi"] <= 1200):
            errors.append(f"폴백 DPI 값이 범위를 벗어났습니다: {PDF_CONFIG['fallback_dpi']} (100-1200)")

        # Validate worker count
        if PDF_CONFIG["max_workers"] < 1:
            errors.append(f"최대 워커 수가 잘못되었습니다: {PDF_CONFIG['max_workers']} (최소 1)")

        return errors

    @staticmethod
    def validate_image_config() -> List[str]:
        """Validate image processing configuration.

        Returns:
            List of validation errors
        """
        errors = []

        IMAGE_CONFIG = _load_config_module().IMAGE_CONFIG

        # Validate deskew angle
        if not (0 < IMAGE_CONFIG["deskew_angle_search"] <= 45):
            errors.append(f"Deskew 각도 범위가 잘못되었습니다: {IMAGE_CONFIG['deskew_angle_search']} (0-45)")

        # Validate Canny thresholds
        if not (0 <= IMAGE_CONFIG["canny_low_threshold"] < IMAGE_CONFIG["canny_high_threshold"] <= 300):
            errors.append(f"Canny 임계값이 잘못되었습니다: Low={IMAGE_CONFIG['canny_low_threshold']}, High={IMAGE_CONFIG['canny_high_threshold']}")

        # Validate line length parameters
        if IMAGE_CONFIG["min_absolute_line_length"] < 1:
            errors.append(f"최소 선 길이가 잘못되었습니다: {IMAGE_CONFIG['min_absolute_line_length']} (최소 1)")

        return errors

    @staticmethod
    def validate_performance_config() -> List[str]:
        """Validate performance configuration.

        Returns:
            List of validation errors
        """
        errors = []

        PERFORMANCE_CONFIG = _load_config_module().PERFORMANCE_CONFIG

        # Validate memory limits
        if PERFORMANCE_CONFIG["memory_limit_mb"] < 512:
            errors.append(f"메모리 제한이 너무 낮습니다: {PERFORMANCE_CONFIG['memory_limit_mb']}MB (최소 512MB)")

        # Validate timeout
        if PERFORMANCE_CONFIG["processing_timeout_seconds"] < 60:
            errors.append(f"처리 타임아웃이 너무 짧습니다: {PERFORMANCE_CONFIG['processing_timeout_seconds']}초 (최소 60초)")

        return errors

    @staticmethod
    def validate_dependencies() -> List[str]:
        """Validate required dependencies.

        Returns:
            List of validation errors
        """
        errors = []

        # Check required packages
        required_packages = [
            ("numpy", "NumPy"),
            ("cv2", "OpenCV"),
            ("fitz", "PyMuPDF"),
            ("PIL", "Pillow"),
            ("streamlit", "Streamlit")
        ]

        for module_name, package_name in required_packages:
            try:
                __import__(module_name)
            except ImportError:
                errors.append(f"필수 패키지가 설치되지 않았습니다: {package_name} ({module_name})")

        # Check optional packages
        optional_packages = [
            ("skimage", "scikit-image (선택사항)"),
            ("shapely", "Shapely (선택사항)")
        ]

        for module_name, package_name in optional_packages:
            try:
                __import__(module_name)
            except ImportError:
                logger.warning(f"선택 패키지가 설치되지 않았습니다: {package_name}")

        return errors

    @staticmethod
    def validate_system_resources() -> List[str]:
        """Validate system resources.

        Returns:
            List of validation warnings/issues
        """
        warnings = []

        try:
            import psutil

            # Check available memory
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
            if available_memory < 2:
                warnings.append(".1f")

            # Check available disk space
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024 * 1024 * 1024)
            if free_space_gb < 5:
                warnings.append(".1f")

        except ImportError:
            warnings.append("psutil이 설치되지 않아 시스템 리소스 검사를 수행할 수 없습니다")

        return warnings

    @classmethod
    def validate_all(cls) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks.

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Path validation
        errors.extend(cls.validate_paths())

        # Configuration validation
        errors.extend(cls.validate_pdf_config())
        errors.extend(cls.validate_image_config())
        errors.extend(cls.validate_performance_config())

        # Dependency validation
        errors.extend(cls.validate_dependencies())

        # System resource validation
        warnings.extend(cls.validate_system_resources())

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("모든 설정 검증을 통과했습니다")
        else:
            logger.error(f"설정 검증 실패: {len(errors)}개의 오류 발견")

        return is_valid, errors, warnings


def print_validation_report(is_valid: bool, errors: List[str], warnings: List[str]) -> None:
    """Print validation report.

    Args:
        is_valid: Whether validation passed
        errors: List of errors
        warnings: List of warnings
    """
    print("\n" + "="*50)
    print("CAD 도면 라인 추출 시스템 - 설정 검증 보고서")
    print("="*50)

    if is_valid:
        print("✅ 모든 검증을 통과했습니다!")
    else:
        print("❌ 검증 실패 - 다음 오류들을 수정해주세요:")

        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    if warnings:
        print("\n⚠️  경고 사항들:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    print("="*50)
