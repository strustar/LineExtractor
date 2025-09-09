"""
Custom exceptions for CAD line extraction system.
"""

from typing import Optional


class CADProcessingError(Exception):
    """Base exception for CAD processing errors."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class PDFProcessingError(CADProcessingError):
    """Exception raised for PDF processing errors."""

    def __init__(self, message: str, pdf_path: Optional[str] = None):
        self.pdf_path = pdf_path
        full_message = f"PDF 처리 오류: {message}"
        if pdf_path:
            full_message += f" (파일: {pdf_path})"
        super().__init__(full_message, "PDF_ERROR")


class ImageProcessingError(CADProcessingError):
    """Exception raised for image processing errors."""

    def __init__(self, message: str, operation: Optional[str] = None):
        self.operation = operation
        full_message = f"이미지 처리 오류: {message}"
        if operation:
            full_message += f" (작업: {operation})"
        super().__init__(full_message, "IMAGE_ERROR")


class LineDetectionError(CADProcessingError):
    """Exception raised for line detection errors."""

    def __init__(self, message: str, algorithm: Optional[str] = None):
        self.algorithm = algorithm
        full_message = f"선 검출 오류: {message}"
        if algorithm:
            full_message += f" (알고리즘: {algorithm})"
        super().__init__(full_message, "LINE_DETECTION_ERROR")


class ConfigurationError(CADProcessingError):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        full_message = f"설정 오류: {message}"
        if config_key:
            full_message += f" (설정키: {config_key})"
        super().__init__(full_message, "CONFIG_ERROR")


class ValidationError(CADProcessingError):
    """Exception raised for validation errors."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        full_message = f"유효성 검사 오류: {message}"
        if field:
            full_message += f" (필드: {field})"
        super().__init__(full_message, "VALIDATION_ERROR")

