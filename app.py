import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from PIL import Image
import subprocess
import sys
import time
import os

# Import required libraries
import cv2
import fitz  # PyMuPDF
import ezdxf

# Set availability flags
HAS_OPENCV = True
HAS_PYMUPDF = True

# Import custom exceptions and functions
try:
    sys.path.append(str(Path(__file__).parent))
    from src.exceptions import (
        CADProcessingError, PDFProcessingError, ImageProcessingError,
        LineDetectionError, ConfigurationError, ValidationError
    )
    from src.main import build_handwriting_masks_fast
    from src.main import build_handwriting_color_mask
    from src.main import build_handwriting_strict_mask
    from src.main import build_printed_mask
except ImportError as e:
    # Fallback for missing modules - psutil is optional
    if "psutil" not in str(e):
        st.warning(f"일부 모듈을 불러올 수 없습니다: {e}")
    
    # Define fallback exception classes
    class CADProcessingError(Exception): pass
    class PDFProcessingError(Exception): pass
    class ImageProcessingError(Exception): pass
    class LineDetectionError(Exception): pass
    class ConfigurationError(Exception): pass
    class ValidationError(Exception): pass
    
    # Define fallback functions
    def build_handwriting_masks_fast(gray, downscale=0.5):
        return np.zeros_like(gray), np.zeros_like(gray)
    def build_handwriting_color_mask(image_bgr, sat_thr=60, diff_thr=35):
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    def build_handwriting_strict_mask(image_bgr, gray_rot):
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    def build_printed_mask():
        return np.zeros((100, 100), dtype=np.uint8)


ROOT = Path(__file__).resolve().parent
DEFAULT_PAGE = ROOT / "outputs" / "lines" / "page_000"


def read_image(path: Path) -> Image.Image:
	try:
		img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
		if img is None:
			if not path.exists():
				raise ImageProcessingError(f"이미지 파일이 존재하지 않습니다: {path}", "file_read")
			else:
				raise ImageProcessingError(f"이미지를 불러올 수 없습니다 (손상된 파일일 수 있음): {path}", "file_read")

		# Keep original BGR -> convert to RGB once for correct colors
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return Image.fromarray(img)

	except cv2.error as e:
		raise ImageProcessingError(f"OpenCV 이미지 처리 오류: {str(e)}", "opencv_processing") from e
	except Exception as e:
		if isinstance(e, ImageProcessingError):
			raise
		raise ImageProcessingError(f"이미지 읽기 중 예상치 못한 오류: {str(e)}", "image_read") from e


def load_lines(path: Path) -> List[Dict[str, Any]]:
	try:
		if not path.exists():
			st.warning(f"라인 데이터 파일이 존재하지 않습니다: {path}")
			return []

		data = json.loads(path.read_text(encoding="utf-8"))
		lines = data.get("lines", [])

		if not isinstance(lines, list):
			st.warning(f"잘못된 JSON 형식: {path}")
			return []

		return lines

	except json.JSONDecodeError as e:
		st.error(f"JSON 파일 파싱 오류: {path} - {str(e)}")
		return []
	except UnicodeDecodeError as e:
		st.error(f"파일 인코딩 오류: {path} - {str(e)}")
		return []
	except Exception as e:
		st.error(f"라인 데이터 로드 중 오류: {path} - {str(e)}")
		return []


def draw_lines(canvas: np.ndarray, lines: List[Dict[str, Any]], color_fn, line_width: int = 2) -> np.ndarray:
	out = canvas.copy()
	for ln in lines:
		x0, y0, x1, y1 = ln["x0"], ln["y0"], ln["x1"], ln["y1"]
		length = float(np.hypot(x1 - x0, y1 - y0))
		color = color_fn(length)
		# NOTE: base array is RGB; draw with RGB tuple to avoid BGR swap
		rgb = (int(color[0]), int(color[1]), int(color[2]))
		cv2.line(out, (x0, y0), (x1, y1), rgb, line_width, lineType=cv2.LINE_AA)
	return out


def make_color_fn(mode: str, bins: list[tuple[float, str]], grad_start: str, grad_end: str, lo: float, hi: float):
	def hex_to_rgb(h: str):
		h = h.lstrip('#')
		return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
	gs = hex_to_rgb(grad_start)
	ge = hex_to_rgb(grad_end)
	rules = [(thr, hex_to_rgb(col)) for thr, col in bins]
	def color_fn(length: float):
		if mode == "bins":
			for thr, col in rules:
				if length <= thr:
					return col
			return rules[-1][1]
		# gradient
		t = 0.0 if hi <= lo else max(0.0, min(1.0, (length - lo) / (hi - lo)))
		r = int(gs[0] + (ge[0] - gs[0]) * t)
		g = int(gs[1] + (ge[1] - gs[1]) * t)
		b = int(gs[2] + (ge[2] - gs[2]) * t)
		return (r, g, b)
	return color_fn

# ==== 전처리(배치와 동일) ====

def deskew_by_hough(gray: np.ndarray, angle_search_deg: float = 7.0) -> Tuple[np.ndarray, float]:
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=200)
	if lines is None:
		return gray, 0.0
	angles = []
	for rho_theta in lines:
		rho, theta = rho_theta[0]
		deg = (theta * 180.0 / np.pi) - 90.0
		if -angle_search_deg <= deg <= angle_search_deg:
			angles.append(deg)
	if not angles:
		return gray, 0.0
	angle = float(np.median(angles))
	h, w = gray.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	return rot, angle


def simple_binarize(gray: np.ndarray) -> np.ndarray:
	return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 10)


def rotate_rgb(image_rgb: np.ndarray, angle: float) -> np.ndarray:
	h, w = image_rgb.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	return cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


st.set_page_config(
    page_title="CAD 도면 라인 추출 시스템", 
    layout="wide", 
    page_icon="📐",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "### CAD 라인 추출\n\n엔지니어링 도면에서 라인 세그먼트를\n실시간으로 추출하고 시각화하는 도구"
    }
)

# 메인 헤더
st.markdown("""
<div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1);'>
    <h1 style='color: white; margin: 0; font-size: 2.5em; font-weight: 700;'>
        📐 CAD 라인 추출 시스템
    </h1>
    <p style='color: rgba(255,255,255,0.9); margin: 1rem 0 0 0; font-size: 1.2em; font-weight: 300;'>
        🚀 실시간 라인 검출 • 손글씨 제거 • 연결성 분석
    </p>
</div>
""", unsafe_allow_html=True)

# 처리 상태 알림
status_placeholder = st.empty()

# 메인 콘텐츠 영역

with st.sidebar:
	st.title("⚙️ 설정")
	
	# PDF 업로드 섹션
	st.markdown("### 📄 PDF 파일")
	
	# 기본 풀도면이 있으면 자동 로드
	uploaded_pdf_bytes = st.session_state.get("uploaded_pdf_bytes")
	full_pdf_path = ROOT / "풀도면.pdf"
	
	if uploaded_pdf_bytes is None and full_pdf_path.exists():
		with open(full_pdf_path.as_posix(), "rb") as f:
			st.session_state["uploaded_pdf_bytes"] = f.read()
		uploaded_pdf_bytes = st.session_state["uploaded_pdf_bytes"]
		st.success("✅ 풀도면.pdf 자동 로드됨")
	
	upload = st.file_uploader("파일 선택", type=["pdf"], help="CAD 도면 PDF 파일을 업로드하세요")
	if upload is not None:
		st.session_state["uploaded_pdf_bytes"] = upload.getvalue()
		uploaded_pdf_bytes = st.session_state["uploaded_pdf_bytes"]
		st.success("✅ 파일 업로드 완료")
	
	# PDF 정보 표시
	source_mode = "업로드(PDF)"  # 항상 업로드 모드

	# PDF 페이지 정보 및 선택
	if uploaded_pdf_bytes:
		try:
			doc = fitz.open(stream=uploaded_pdf_bytes, filetype="pdf")
			if doc.page_count > 1:
				upload_page_index = st.selectbox(
					"페이지 선택", 
					range(doc.page_count),
					index=0,
					format_func=lambda x: f"페이지 {x+1}",
					help=f"총 {doc.page_count}개 페이지"
				)
				# 세션에 저장하여 다른 곳에서 사용 가능하게 함
				st.session_state["upload_page_idx"] = upload_page_index
			else:
				upload_page_index = 0
				st.session_state["upload_page_idx"] = 0
				st.info(f"📑 총 1페이지")
		except Exception:
			upload_page_index = 0
			st.session_state["upload_page_idx"] = 0
			st.error("PDF 파일을 읽을 수 없습니다")
	else:
		upload_page_index = 0
		st.session_state["upload_page_idx"] = 0

	st.divider()
	
	# 라인 검출 설정
	st.markdown("### 🔍 라인 검출")
	algo = st.selectbox("알고리즘", ["lsd", "hough", "morph", "fused"], index=0, help="LSD 권장 (빠르고 정확)")
	
	# 동적 최소 길이 기본값
	dynamic_min_len = 35  # 기본값을 45로 조정 (노이즈 감소)
	
	min_len = st.slider("최소 길이", 0, 100, dynamic_min_len, step=5, help="짧은 선 제거 (0에 가까울수록 더 많은 라인 검출)")
	len_guide = "🔴 높음(40+): 긴선만" if min_len > 40 else "🟡 중간(15-40): 보통길이" if min_len > 15 else "🟢 낮음(~15): 짧은선도"
	st.caption(f"→ {len_guide}")
	
	col1, col2 = st.columns(2)
	with col1:
		ortho = st.checkbox("직교선만", value=True, help="수직/수평만 (체크 시 대각선 제외)")
	with col2:
		line_width = st.slider("선 굵기", 1, 5, 2, help="시각화용")

	# 손글씨 제거 설정
	st.markdown("### ✍️ 손글씨 제거")
	enable_handwriting_removal = st.checkbox("손글씨 제거 활성화", value=True, help="체크하면 손글씨를 자동으로 제거합니다")
	
	if enable_handwriting_removal:
		hw_mode = st.selectbox("검출 강도", ["strict", "union", "color", "stroke"], index=0, help="strict: 안전하게 | union: 적극적으로")
		
		# 상세 설정을 팝오버로 숨김
		with st.expander("🔧 상세 설정", expanded=False):
			col1, col2 = st.columns(2)
			
			with col1:
				st.markdown("**색상 검출 설정**")
				hw_sat = st.slider("채도 임계값", 0, 200, 60, step=5, help="낮을수록 더 많은 색상 검출")
				sat_guide = "🔴 높음(100+): 진한색만" if hw_sat > 100 else "🟡 중간(40-100): 보통색" if hw_sat > 40 else "🟢 낮음(~40): 연한색도"
				st.caption(f"→ {sat_guide}")
				
				hw_chroma = st.slider("크로마 임계값", 0, 50, 25, step=1, help="LAB 색공간의 크로마 값")
				chroma_guide = "🔴 높음(35+): 선명한색만" if hw_chroma > 35 else "🟡 중간(15-35): 일반색" if hw_chroma > 15 else "🟢 낮음(~15): 회색조도"
				st.caption(f"→ {chroma_guide}")
				
				hw_spread = st.slider("색차 임계값", 0, 100, 35, step=5, help="RGB 채널 간 차이")
				spread_guide = "🔴 높음(50+): 다채로운색만" if hw_spread > 50 else "🟡 중간(20-50): 보통차이" if hw_spread > 20 else "🟢 낮음(~20): 단색조도"
				st.caption(f"→ {spread_guide}")
			
			with col2:
				st.markdown("**형태학적 설정**")
				hw_edge_low = st.slider("엣지 하한", 30, 120, 80, step=5, help="Canny 엣지 검출 하한")
				edge_low_guide = "🔴 높음(90+): 강한윤곽만" if hw_edge_low > 90 else "🟡 중간(60-90): 보통윤곽" if hw_edge_low > 60 else "🟢 낮음(~60): 약한윤곽도"
				st.caption(f"→ {edge_low_guide}")
				
				hw_edge_high = st.slider("엣지 상한", 120, 250, 180, step=10, help="Canny 엣지 검출 상한")
				edge_high_guide = "🔴 높음(200+): 매우선명" if hw_edge_high > 200 else "🟡 중간(150-200): 보통선명" if hw_edge_high > 150 else "🟢 낮음(~150): 부드러운"
				st.caption(f"→ {edge_high_guide}")
				
				show_mask_debug = st.checkbox("마스크 보기", help="검출 영역 확인")
			
			# 고급 설정
			st.markdown("**🔬 고급 설정**")
			adv_col1, adv_col2 = st.columns(2)
			with adv_col1:
				hw_min_area = st.slider("최소 영역", 5, 50, 16, step=1, help="작은 노이즈 제거")
				area_guide = "🔴 높음(30+): 큰것만" if hw_min_area > 30 else "🟡 중간(15-30): 보통크기" if hw_min_area > 15 else "🟢 낮음(~15): 작은것도"
				st.caption(f"→ {area_guide}")
				
				hw_max_aspect = st.slider("최대 종횡비", 3.0, 15.0, 7.0, step=0.5, help="긴 선 제거")
				aspect_guide = "🔴 높음(10+): 긴선허용" if hw_max_aspect > 10 else "🟡 중간(5-10): 보통비율" if hw_max_aspect > 5 else "🟢 낮음(~5): 정사각형만"
				st.caption(f"→ {aspect_guide}")
				
			with adv_col2:
				hw_coverage_limit = st.slider("커버리지 제한", 0.1, 0.5, 0.2, step=0.05, help="전체 이미지 대비 최대 손글씨 비율")
				coverage_guide = "🔴 높음(0.3+): 관대함" if hw_coverage_limit > 0.3 else "🟡 중간(0.15-0.3): 보통" if hw_coverage_limit > 0.15 else "🟢 낮음(~0.15): 엄격함"
				st.caption(f"→ {coverage_guide}")
	else:
		# 기본값 사용
		hw_mode = "strict"
		hw_sat = 60
		hw_spread = 35
		hw_chroma = 25
		hw_edge_low = 80
		hw_edge_high = 180
		hw_min_area = 16
		hw_max_aspect = 7.0
		hw_coverage_limit = 0.2
		show_mask_debug = False

	st.divider()
	
	# DPI 설정
	st.markdown("### 🎨 품질 설정")
	if uploaded_pdf_bytes:
		dpi_val = st.slider("렌더링 DPI", 200, 600, 300, step=50, help="높을수록 선명하지만 메모리 사용량 증가")
	else:
		dpi_val = 300
		st.info("📄 PDF 업로드 후 설정 가능")

	# 연결성 분석 설정 - 상세 설정처럼 기본값 표시
	with st.expander("🔗 연결성 분석", expanded=False):
		st.markdown("**기본 설정**")
		conn_col1, conn_col2 = st.columns(2)
		with conn_col1:
			connectivity_tolerance = st.slider("연결 허용 거리", 5, 100, 40, step=5, help="이 거리 내의 라인들을 연결된 것으로 간주")
			min_connected_length = st.slider("최소 연결 길이", 10, 200, 50, step=10, help="연결된 라인 그룹의 최소 총 길이")
			
		with conn_col2:
			angle_tolerance = st.slider("각도 허용 오차", 5, 90, 30, step=5, help="연결 시 허용되는 각도 차이 (도)")
			min_segment_ratio = st.slider("최소 세그먼트 비율", 0.05, 0.8, 0.15, step=0.05, help="전체 길이 대비 각 세그먼트의 최소 비율")
		
		# 사용자 가이드
		st.markdown("""**📚 사용 가이드:**
		- **연결 거리**: 더 크면 더 멀리 떨어진 선도 연결
		- **최소 길이**: 이보다 짧은 그룹은 제외
		- **각도 오차**: 작을수록 평행한 선만 연결
		- **비율**: 너무 짧은 세그먼트 제외""")
	
	# 연결성 분석은 항상 활성화된 상태로 설정값만 사용
	enable_connectivity = True

	# Hough 파라미터 (필요시만)
	if algo == "hough":
		st.markdown("### ⚙️ Hough 세부조정")
		hough_col1, hough_col2 = st.columns(2)
		with hough_col1:
			can_low = st.slider("Canny 하한", 30, 100, 50, step=5)
			can_high = st.slider("Canny 상한", 100, 200, 150, step=5)
		with hough_col2:
			hough_thresh = st.slider("검출 임계값", 20, 100, 50, step=5)
			max_gap = st.slider("연결 간격", 2, 10, 4, step=1)
	else:
		# 기본값
		can_low = 50
		can_high = 150
		hough_thresh = 50
		max_gap = 4

	# 시각화 (기본값 사용)
	color_mode = "gradient"
	grad_start = "#0000ff"
	grad_end = "#ff0000"  
	map_min = 0
	map_max = 0
	bins = [(999999.0, "#ff0000")]
	mode = "gradient"

# 라인 연결성 분석 함수
def analyze_line_connectivity(lines: List[Dict[str, Any]], tolerance: float = 20.0, 
                             angle_tolerance: float = 15.0) -> List[List[int]]:
	"""라인들을 연결성에 따라 그룹화합니다."""
	if not lines:
		return []
	
	import math
	from collections import defaultdict
	
	def distance_point_to_point(p1, p2):
		return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
	
	def line_angle(line):
		dx = line["x1"] - line["x0"]
		dy = line["y1"] - line["y0"]
		return math.atan2(dy, dx) * 180 / math.pi
	
	def angle_difference(a1, a2):
		diff = abs(a1 - a2)
		return min(diff, 180 - diff)
	
	# 각 라인의 끝점들
	endpoints = []
	for i, line in enumerate(lines):
		endpoints.append(((line["x0"], line["y0"]), i, "start"))
		endpoints.append(((line["x1"], line["y1"]), i, "end"))
	
	# 연결 그래프 생성
	connections = defaultdict(list)
	
	for i in range(len(endpoints)):
		for j in range(i + 1, len(endpoints)):
			point1, line1_idx, _ = endpoints[i]
			point2, line2_idx, _ = endpoints[j]
			
			if line1_idx == line2_idx:  # 같은 라인의 끝점들
				continue
				
			dist = distance_point_to_point(point1, point2)
			
			if dist <= tolerance:
				# 각도 차이 확인
				angle1 = line_angle(lines[line1_idx])
				angle2 = line_angle(lines[line2_idx])
				
				if angle_difference(angle1, angle2) <= angle_tolerance:
					connections[line1_idx].append(line2_idx)
					connections[line2_idx].append(line1_idx)
	
	# 연결된 컴포넌트 찾기
	visited = set()
	groups = []
	
	for line_idx in range(len(lines)):
		if line_idx not in visited:
			group = []
			stack = [line_idx]
			
			while stack:
				current = stack.pop()
				if current not in visited:
					visited.add(current)
					group.append(current)
					
					for neighbor in connections[current]:
						if neighbor not in visited:
							stack.append(neighbor)
			
			if group:
				groups.append(group)
	
	return groups

def filter_connected_lines(lines: List[Dict[str, Any]], groups: List[List[int]], 
                          min_total_length: float = 50.0, 
                          min_segment_ratio: float = 0.15) -> List[Dict[str, Any]]:
	"""연결된 라인 그룹을 필터링합니다."""
	import math
	
	def line_length(line):
		return math.sqrt((line["x1"] - line["x0"])**2 + (line["y1"] - line["y0"])**2)
	
	filtered_lines = []
	
	for group in groups:
		group_lines = [lines[i] for i in group]
		total_length = sum(line_length(line) for line in group_lines)
		
		# 더 관대한 조건: 총 길이가 기준 이상이거나 그룹에 3개 이상 라인이 있으면 포함
		if total_length >= min_total_length or len(group_lines) >= 3:
			# 각 세그먼트 비율 검사를 더 관대하게
			valid_lines = []
			for line in group_lines:
				length = line_length(line)
				# 비율 조건을 만족하거나, 절대 길이가 10 이상이면 유효
				if length >= total_length * min_segment_ratio or length >= 10:
					valid_lines.append(line)
			
			# 유효한 라인이 있거나, 단일 긴 라인이면 전체 그룹 포함
			if valid_lines or (len(group_lines) == 1 and line_length(group_lines[0]) >= 20):
				filtered_lines.extend(group_lines)
		else:
			# 작은 그룹도 개별 라인이 충분히 길면 포함
			for line in group_lines:
				if line_length(line) >= 15:
					filtered_lines.append(line)
	
	return filtered_lines

# 향상된 손글씨 검출 함수
def build_handwriting_strict_mask_enhanced(image_bgr: np.ndarray, gray_rot: np.ndarray,
                                         sat_max: int = 50, chroma_max: int = 25,
                                         edge_low: int = 80, edge_high: int = 180,
                                         min_area: int = 16, max_aspect_ratio: float = 7.0,
                                         coverage_limit: float = 0.2) -> np.ndarray:
	"""향상된 strict handwriting mask with configurable parameters."""
	try:
		H, W = gray_rot.shape[:2]
		
		# 파라미터 타입 안전성 보장
		sat_max = int(float(sat_max)) if not isinstance(sat_max, int) else sat_max
		chroma_max = int(float(chroma_max)) if not isinstance(chroma_max, int) else chroma_max
		edge_low = int(float(edge_low)) if not isinstance(edge_low, int) else edge_low
		edge_high = int(float(edge_high)) if not isinstance(edge_high, int) else edge_high
		min_area = int(float(min_area)) if not isinstance(min_area, int) else min_area
		max_aspect_ratio = float(max_aspect_ratio) if not isinstance(max_aspect_ratio, float) else max_aspect_ratio
		coverage_limit = float(coverage_limit) if not isinstance(coverage_limit, float) else coverage_limit
		
		# LAB 크로마 magnitude
		lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
		L, a, b = cv2.split(lab)
		chroma = cv2.magnitude(a.astype(np.float32) - 128.0, b.astype(np.float32) - 128.0)
		chroma = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		thr_ch = max(15, int(cv2.threshold(chroma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))
		thr_ch = min(thr_ch, chroma_max)  # 사용자 설정 적용
		mask_ch = (chroma > thr_ch).astype(np.uint8) * 255
		
		# HSV saturation + channel spread
		hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
		mask_sat = (hsv[:, :, 1] > sat_max).astype(np.uint8) * 255  # 사용자 설정 적용
		bgr_max = cv2.max(cv2.max(image_bgr[:, :, 0], image_bgr[:, :, 1]), image_bgr[:, :, 2])
		bgr_min = cv2.min(cv2.min(image_bgr[:, :, 0], image_bgr[:, :, 1]), image_bgr[:, :, 2])
		mask_spread = (cv2.subtract(bgr_max, bgr_min) > 30).astype(np.uint8) * 255
		
		# Edge gating (사용자 파라미터 적용)
		edges = cv2.Canny(gray_rot, edge_low, edge_high, apertureSize=3)
		edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
		mask_color = cv2.bitwise_and(cv2.bitwise_and(mask_ch, mask_sat), mask_spread)
		mask = cv2.bitwise_and(mask_color, edges)
		
		# Remove HV structural lines
		min_dim = min(H, W)
		k = max(15, int(min_dim * 0.02))
		h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
		v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
		hv_h = cv2.morphologyEx(gray_rot, cv2.MORPH_OPEN, h_kernel, iterations=1)
		hv_v = cv2.morphologyEx(gray_rot, cv2.MORPH_OPEN, v_kernel, iterations=1)
		hv = cv2.max(hv_h, hv_v)
		_, hv_bin = cv2.threshold(hv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		mask[hv_bin > 0] = 0
		
		# Clean noise and enforce coverage limits
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
		coverage = float(np.count_nonzero(mask)) / float(H * W)
		
		# 사용자 설정 coverage_limit 적용
		if coverage > coverage_limit:
			# Too broad → tighten thresholds
			mask = cv2.bitwise_and(mask, (chroma > thr_ch + 10).astype(np.uint8) * 255)
			mask = cv2.bitwise_and(mask, (hsv[:, :, 1] > sat_max + 20).astype(np.uint8) * 255)
			mask = cv2.bitwise_and(mask, (cv2.subtract(bgr_max, bgr_min) > 45).astype(np.uint8) * 255)
			mask = cv2.bitwise_and(mask, edges)
			mask[hv_bin > 0] = 0
		
		# Connected component filtering (사용자 파라미터 적용)
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		filtered_mask = np.zeros_like(mask)
		
		for contour in contours:
			area = cv2.contourArea(contour)
			if area < min_area:  # 사용자 설정 적용
				continue
			
			# 종횡비 필터링
			x, y, w, h = cv2.boundingRect(contour)
			aspect_ratio = max(w, h) / max(1, min(w, h))
			if aspect_ratio > max_aspect_ratio:  # 사용자 설정 적용
				continue
			
			cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
		
		return filtered_mask
		
	except Exception:
		return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

# DXF 파일 생성 함수
def create_dxf_file(lines: List[Dict[str, Any]], filename: str, scale_factor: float = 1.0) -> bytes:
	"""라인 데이터를 DXF 파일로 변환"""
	try:
		# DXF 문서 생성
		doc = ezdxf.new('R2010')
		msp = doc.modelspace()
		
		# 라인을 DXF에 추가
		for line in lines:
			x0 = float(line["x0"]) * scale_factor
			y0 = float(line["y0"]) * scale_factor 
			x1 = float(line["x1"]) * scale_factor
			y1 = float(line["y1"]) * scale_factor
			
			# Y좌표 뒤집기 (이미지 좌표계 -> CAD 좌표계)
			# 이미지는 상단이 0, CAD는 하단이 0이므로
			# 필요시 이미지 높이값으로 뒤집을 수 있음
			msp.add_line((x0, -y0), (x1, -y1))
		
		# 메모리에서 직접 DXF 생성
		import io
		from io import StringIO
		
		# DXF를 StringIO로 저장
		dxf_stream = StringIO()
		doc.write(dxf_stream)
		dxf_content = dxf_stream.getvalue()
		dxf_stream.close()
		
		# 문자열을 바이트로 변환
		return dxf_content.encode('utf-8')
			
	except Exception as e:
		st.error(f"DXF 파일 생성 중 오류: {str(e)}")
		return b""

# 검출 함수

def detect_lsd(gray_or_bin: np.ndarray) -> List[Dict[str, Any]]:
	lsd = cv2.createLineSegmentDetector()
	lines, _width, _prec, _nfa = lsd.detect(gray_or_bin)
	res = []
	if lines is not None:
		for l in lines:
			x0, y0, x1, y1 = map(int, l[0])
			res.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
	return res


def detect_hough(gray_or_bin: np.ndarray, can_low: int, can_high: int, thr: int, min_len_px: int, gap: int) -> List[Dict[str, Any]]:
	# gray_or_bin: bin 이미지가 더 강함
	edges = cv2.Canny(gray_or_bin, can_low, can_high, apertureSize=3)
	ls = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=thr, minLineLength=min_len_px, maxLineGap=gap)
	res = []
	if ls is not None:
		for l in ls:
			x0, y0, x1, y1 = map(int, l[0])
			res.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
	return res


def detect_morph(bin_img: np.ndarray, thr: int, min_len_px: int, gap: int, kernel_ratio: float = 0.02) -> List[Dict[str, Any]]:
	# 형태학 기반으로 선 성분 강화 후 HoughP
	H, W = bin_img.shape[:2]
	min_dim = min(H, W)
	k = max(15, int(min_dim * kernel_ratio))
	inv = cv2.bitwise_not(bin_img)
	h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
	v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
	h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
	v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
	line_map = cv2.bitwise_or(h_lines, v_lines)
	edges = cv2.Canny(line_map, 50, 150, apertureSize=3)
	ls = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=thr, minLineLength=min_len_px, maxLineGap=gap)
	res: List[Dict[str, Any]] = []
	if ls is not None:
		for l in ls:
			x0, y0, x1, y1 = map(int, l[0])
			res.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
	return res

# 데이터 소스별 이미지/라인 구성(배치와 동일 전처리 적용)
# PDF가 업로드되면 항상 업로드 모드 사용
if st.session_state.get("uploaded_pdf_bytes"):
	# 업로드된 PDF에서 선택 페이지 렌더링 (사용자 설정 DPI 사용)
	doc = fitz.open(stream=st.session_state["uploaded_pdf_bytes"], filetype="pdf")
	upload_page_index = int(st.session_state.get("upload_page_idx", 0))
	page_obj = doc.load_page(upload_page_index)
	dpi = dpi_val  # 사용자가 설정한 DPI 값 사용
	mat = fitz.Matrix(dpi/72.0, dpi/72.0)
	pix = page_obj.get_pixmap(matrix=mat, alpha=False)

	npimg = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

	if pix.n == 4:
		npimg = cv2.cvtColor(npimg, cv2.COLOR_BGRA2RGB)
	elif pix.n == 3:
		pass
	else:
		st.warning(f"예상치 못한 채널 수: {pix.n}")
	
	# 스마트 이미지 크기 최적화 - DPI에 따른 동적 조정
	if dpi_val >= 400:
		max_size = 3500  # 고해상도일 때는 제한
	else:
		max_size = 4500  # 보통 해상도일 때는 관대하게
	
	if npimg.shape[0] > max_size or npimg.shape[1] > max_size:
		scale = max_size / max(npimg.shape[0], npimg.shape[1])
		new_height = int(npimg.shape[0] * scale)
		new_width = int(npimg.shape[1] * scale)
		npimg = cv2.resize(npimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
		st.info(f"📏 이미지 크기 최적화: {new_width}x{new_height} (DPI {dpi_val})")

	gray0 = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
	gray, ang = deskew_by_hough(gray0)
	# 표시용 RGB도 같은 각도로 회전
	npimg_rot = rotate_rgb(npimg, ang)
	image = Image.fromarray(npimg_rot)

	# 이진화 시도
	try:
		bin_img = simple_binarize(gray)
	except Exception:
		_, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	min_len_px = int(min_len)

	# 선 검출 - 배치 처리와 동일한 파라미터 사용
	if algo == "lsd":
		lines = detect_lsd(bin_img)
	elif algo == "morph":
		lines = detect_morph(bin_img, hough_thresh, min_len_px, max_gap)
	else:
		if algo == "fld":
			try:
				from cv2 import ximgproc  # type: ignore
				fld = ximgproc.createFastLineDetector()
				ls = fld.detect(bin_img if bin_img.ndim == 2 else cv2.cvtColor(bin_img, cv2.COLOR_RGB2GRAY))
				lines = []
				if ls is not None:
					for l in ls:
						x0, y0, x1, y1 = map(int, l[0]) if hasattr(l, '__len__') and len(l) == 1 else map(int, l)
						lines.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
			except Exception:
				lines = []
		elif algo == "morph_diag":
			# 간단히 형태학 대각 강화 후 HoughP
			H, W = bin_img.shape[:2]
			min_dim = min(H, W)
			inv = cv2.bitwise_not(bin_img)
			k = max(15, int(min_dim * 0.02))
			base = np.zeros((k, k), dtype=np.uint8)
			cv2.line(base, (0, k // 2), (k - 1, k // 2), 255, 1)
			M45 = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), 45.0, 1.0)
			M135 = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), 135.0, 1.0)
			d1 = cv2.threshold(cv2.warpAffine(base, M45, (k, k)), 127, 255, cv2.THRESH_BINARY)[1]
			d2 = cv2.threshold(cv2.warpAffine(base, M135, (k, k)), 127, 255, cv2.THRESH_BINARY)[1]
			d1_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, d1, iterations=1)
			d2_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, d2, iterations=1)
			line_map = cv2.bitwise_or(d1_lines, d2_lines)
			edges = cv2.Canny(line_map, can_low, can_high, apertureSize=3)
			ls = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=hough_thresh, minLineLength=min_len_px, maxLineGap=max_gap)
			lines = []
			if ls is not None:
				for l in ls:
					x0, y0, x1, y1 = map(int, l[0])
					lines.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
		else:
			lines = detect_hough(bin_img, can_low, can_high, hough_thresh, min_len_px, max_gap)

		# 만약 선이 하나도 검출되지 않았다면 더 낮은 임계값으로 재시도
		if len(lines) == 0 and not st.session_state.get("retry_attempted", False):
			lines = detect_hough(bin_img, max(30, can_low-20), min(200, can_high+50), max(30, hough_thresh-20), min_len_px, max_gap)
			st.session_state["retry_attempted"] = True

	# 선 검출 결과 상태 (더 간단한 피드백)
	if len(lines) < 10:
		# 적은 선 검출은 사이드바에서만 표시
		pass
	elif len(lines) > 5000:
		# 매우 많은 선만 경고 (기존 1000 → 5000으로 상향)
		st.info(f"🔍 많은 선이 검출되었습니다 ({len(lines)}개). 필요시 최소 길이를 조정하세요.")

	# 심각한 오류만 메인 화면에 표시
	if len(lines) == 0:
		st.error("❌ 선이 검출되지 않았습니다")
		with st.expander("🔧 해결 방법 보기"):
			st.markdown("""
			**파라미터 조정 권장사항:**
			- 최소 길이를 0-5로 낮추기
			- 알고리즘을 LSD로 변경
			- Hough 임계값을 30-40으로 낮추기
			""")
			col1, col2 = st.columns(2)
			with col1:
				if st.button("🔄 자동 최적화", key="auto_optimize"):
					st.session_state.can_low = 30
					st.session_state.can_high = 150
					st.session_state.hough_thresh = 30
					st.session_state.min_len = 0
					st.experimental_rerun()
			with col2:
				if st.button("📂 기존결과 전환", key="switch_to_existing"):
					st.session_state.source_mode = "기존결과"
					st.experimental_rerun()
else:
	# 기존 결과 모드: outputs/lines 사용
	selected_page = Path(selected_page_path) if selected_page_path else DEFAULT_PAGE
	orig_path = selected_page / "original.png"
	lines_path = selected_page / (f"lines_fused.json" if algo == "fused" else f"lines_{algo}.json")
	image = read_image(orig_path)
	if recompute:
		# 배치와 동일 전처리 후 재검출
		npimg = np.array(image)
		gray0 = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
		gray, ang = deskew_by_hough(gray0)
		# 표시용 RGB도 같은 각도로 회전
		npimg_rot = rotate_rgb(npimg, ang)
		image = Image.fromarray(npimg_rot)
		try:
			bin_img = simple_binarize(gray)
		except Exception:
			_, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		min_len_px = int(min_len)
		if algo == "lsd":
			lines = detect_lsd(bin_img)
		elif algo == "morph":
			lines = detect_morph(bin_img, hough_thresh, min_len_px, max_gap)
		else:
			lines = detect_hough(bin_img, can_low, can_high, hough_thresh, min_len_px, max_gap)
		st.info(f"🔄 실시간 재검출 완료 - {len(lines)}개 선 검출")
	else:
		lines = load_lines(lines_path)
		st.info(f"📁 저장된 결과 불러옴 - {len(lines)}개 선 검출")

# 후처리 공통 필터
if min_len > 0 or ortho:
	filtered = []
	for ln in lines:
		length = float(np.hypot(ln["x1"] - ln["x0"], ln["y1"] - ln["y0"]))
		if length < float(min_len):
			continue
		if ortho:
			ang = abs(np.degrees(np.arctan2(ln["y1"]-ln["y0"], ln["x1"]-ln["x0"])) )
			dang = min(ang % 90, 90 - (ang % 90))
			if dang > 7:
				continue
		filtered.append(ln)
	lines = filtered

# 연결성 분석 적용
if enable_connectivity and lines:
	try:
		# 라인 그룹화
		groups = analyze_line_connectivity(
			lines, 
			tolerance=connectivity_tolerance, 
			angle_tolerance=angle_tolerance
		)
		
		# 연결된 라인 필터링
		lines = filter_connected_lines(
			lines, 
			groups, 
			min_total_length=min_connected_length,
			min_segment_ratio=min_segment_ratio
		)
		
		# 연결성 분석 결과 표시
		with st.expander(f"🔗 연결성 분석 결과: {len(groups)}개 그룹 → {len(lines)}개 라인", expanded=False):
			st.markdown("**연결성 분석 통계:**")
			group_sizes = [len(group) for group in groups]
			if group_sizes:
				col1, col2, col3 = st.columns(3)
				with col1:
					st.metric("평균 그룹 크기", f"{sum(group_sizes)/len(group_sizes):.1f}")
				with col2:
					st.metric("최대 그룹 크기", max(group_sizes))
				with col3:
					st.metric("단독 라인", f"{group_sizes.count(1)}개")
				
	except Exception as e:
		st.warning(f"연결성 분석 중 오류: {e}")
		# 오류 시 원본 라인 유지

lo = 0.0
hi = max([float(np.hypot(ln["x1"] - ln["x0"], ln["y1"] - ln["y0"])) for ln in lines], default=1.0)
if color_mode == "gradient" and map_max > 0:
	hi = float(map_max)
	lo = float(map_min)

color_fn = make_color_fn(mode, bins, grad_start, grad_end, lo, hi)

# 메인 화면 레이아웃: 2x2 격자
st.divider()

# 원본 이미지와 라인 추출 준비
white = Image.new("RGB", (image.size[0], image.size[1]), color=(255, 255, 255))
line_only = draw_lines(np.array(white), lines, color_fn, line_width)

# 손글씨 제거 처리
try:
    # 현재 표시 중인 이미지에서 바로 수행
    npimg_curr = np.array(image)
    gray_curr = cv2.cvtColor(npimg_curr, cv2.COLOR_RGB2GRAY)
    # 손글씨 마스크 생성 - 여러 방법 조합
    bgr_curr = cv2.cvtColor(npimg_curr, cv2.COLOR_RGB2BGR)
    
    # 각종 마스크 생성 (새로운 파라미터 적용)
    if enable_handwriting_removal:
        try:
            hand_mask_s, printed_mask = build_handwriting_masks_fast(gray_curr, downscale=0.5)
        except:
            hand_mask_s = np.zeros_like(gray_curr, dtype=np.uint8)
            printed_mask = np.zeros_like(gray_curr, dtype=np.uint8)
        
        hand_mask_c = build_handwriting_color_mask(bgr_curr, sat_thr=int(hw_sat), diff_thr=int(hw_spread))
        
        # Strict 마스크에 새로운 파라미터 적용
        try:
            # build_handwriting_strict_mask에 새 파라미터 전달
            hand_mask_strict = build_handwriting_strict_mask_enhanced(
                bgr_curr, gray_curr,
                sat_max=int(float(hw_sat)),
                chroma_max=int(float(hw_chroma)),
                edge_low=int(float(hw_edge_low)),
                edge_high=int(float(hw_edge_high)),
                min_area=int(float(hw_min_area)),
                max_aspect_ratio=float(hw_max_aspect),
                coverage_limit=float(hw_coverage_limit)
            )
        except:
            # 기존 함수 사용 (fallback)
            hand_mask_strict = build_handwriting_strict_mask(bgr_curr, gray_curr)
    else:
        # 손글씨 제거 비활성화 시 빈 마스크
        hand_mask_s = np.zeros_like(gray_curr, dtype=np.uint8)
        hand_mask_c = np.zeros_like(gray_curr, dtype=np.uint8)
        hand_mask_strict = np.zeros_like(gray_curr, dtype=np.uint8)
        printed_mask = np.zeros_like(gray_curr, dtype=np.uint8)
    
    # Union 마스크는 stroke와 color를 합친 것 (더 포괄적)
    mask_union = cv2.bitwise_or(hand_mask_s, hand_mask_c)
    
    # 마스크 강화: 팽창 연산으로 손글씨 영역 확대
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_union_dilated = cv2.dilate(mask_union, kernel, iterations=2)
    hand_mask_s_dilated = cv2.dilate(hand_mask_s, kernel, iterations=1)
    hand_mask_c_dilated = cv2.dilate(hand_mask_c, kernel, iterations=1)
    
    # 자동 강도 조절: 커버리지 기준으로 마스크 선택
    def _cov(m):
        return float(np.count_nonzero(m)) / max(1, m.size)
    
    mode_map = {
        "strict": hand_mask_strict,
        "union": mask_union_dilated,  # 팽창된 union 사용
        "stroke": hand_mask_s_dilated,  # 팽창된 stroke 사용
        "color": hand_mask_c_dilated,  # 팽창된 color 사용
    }
    
    # 모드에 따른 마스크 선택
    if hw_mode in mode_map:
        hand_mask = mode_map[hw_mode]
        selected_name = hw_mode
    else:
        # 기본값은 union (가장 포괄적)
        hand_mask = mask_union_dilated
        selected_name = "union"
    
    # 마스크가 너무 적으면 더 강화
    if _cov(hand_mask) < 0.02:
        # 더 민감한 검출: 채도와 색차 임계값 크게 낮추기
        enhanced_mask = build_handwriting_color_mask(bgr_curr, sat_thr=int(max(15, hw_sat-40)), diff_thr=int(max(10, hw_spread-20)))
        # 기존 마스크와 합치기
        hand_mask = cv2.bitwise_or(hand_mask, enhanced_mask)
        # 추가 팽창으로 더 확실하게 제거
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hand_mask = cv2.dilate(hand_mask, kernel_large, iterations=2)
        selected_name = "enhanced"
    
    # 마스크 커버리지 표시
    coverage = _cov(hand_mask) * 100
    selected_name = f"{selected_name} ({coverage:.1f}%)"

    # 손글씨 제거된 도면 생성
    img_hwless = npimg_curr.copy()
    
    # 손글씨 영역을 흰색으로 대체 (마스크가 있는 모든 영역)
    if hand_mask is not None and hand_mask.size > 0:
        # 3채널 이미지에 마스크 적용
        img_hwless[hand_mask > 0] = [255, 255, 255]
        
        # 추가: 인쇄된 텍스트 마스크도 제거 (옵션)
        if printed_mask is not None and _cov(printed_mask) > 0.01:
            # 인쇄 텍스트도 제거하려면 아래 주석 해제
            # img_hwless[printed_mask > 0] = [255, 255, 255]
            pass

    # 손글씨 제거된 이미지에서 라인 검출
    try:
        bin_curr = simple_binarize(gray_curr)
    except Exception:
        _, bin_curr = cv2.threshold(gray_curr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_hwless = bin_curr.copy()
    bin_hwless[hand_mask > 0] = 255

    min_len_px = int(min_len)
    if algo == "lsd":
        pl_lines = detect_lsd(bin_hwless)
    elif algo == "morph":
        pl_lines = detect_morph(bin_hwless, hough_thresh, min_len_px, max_gap)
    else:
        pl_lines = detect_hough(bin_hwless, can_low, can_high, hough_thresh, min_len_px, max_gap)

    # 동일 후처리 필터 적용
    if min_len > 0 or ortho:
        _filtered = []
        for ln in pl_lines:
            length = float(np.hypot(ln["x1"] - ln["x0"], ln["y1"] - ln["y0"]))
            if length < float(min_len):
                continue
            if ortho:
                ang = abs(np.degrees(np.arctan2(ln["y1"]-ln["y0"], ln["x1"]-ln["x0"])) )
                dang = min(ang % 90, 90 - (ang % 90))
                if dang > 7:
                    continue
            _filtered.append(ln)
        pl_lines = _filtered

    # 손글씨 제거된 라인 이미지 생성
    white2 = Image.new("RGB", (image.size[0], image.size[1]), color=(255, 255, 255))
    pl_img = draw_lines(np.array(white2), pl_lines, color_fn, line_width)

    # 2x2 격자 레이아웃
    top_cols = st.columns(2, gap="small")
    bottom_cols = st.columns(2, gap="small")
    
    # 상단 좌: 원본
    with top_cols[0]:
        st.markdown("**📄 원본 이미지**")
        st.image(np.array(image), channels="RGB", use_column_width=True)
        st.caption("원본 PDF/이미지")
    
    # 상단 우: 원본 라인추출
    with top_cols[1]:
        st.markdown("**🔍 원본 라인추출**")
        st.image(line_only, channels="RGB", use_column_width=True)
        st.caption(f"{len(lines)}개 라인 검출 · {algo.upper()}")
        
        # DXF 다운로드 버튼
        if len(lines) > 0:
            dxf_data = create_dxf_file(lines, f"original_lines_{algo}.dxf")
            if dxf_data:
                st.download_button(
                    label="📥 원본 라인 DXF 다운로드",
                    data=dxf_data,
                    file_name=f"original_lines_{algo}.dxf",
                    mime="application/dxf"
                )
    
    # 하단 좌: 손글씨 제거
    with bottom_cols[0]:
        st.markdown("**🧹 손글씨 제거**")
        if show_mask_debug:
            # 마스크 자체를 표시
            mask_display = np.stack([hand_mask, hand_mask, hand_mask], axis=-1)
            st.image(mask_display, channels="RGB", use_column_width=True)
            st.caption(f"손글씨 마스크 · {selected_name}")
        else:
            # 손글씨 제거된 이미지 표시
            st.image(img_hwless, channels="RGB", use_column_width=True)
            st.caption(f"손글씨 제거됨 · {selected_name}")
    
    # 하단 우: 손글씨 제거 후 라인추출
    with bottom_cols[1]:
        st.markdown("**⚡ 손글씨 제거 후 라인추출**")
        st.image(pl_img, channels="RGB", use_column_width=True)
        st.caption(f"{len(pl_lines)}개 라인 검출 · 정제됨")
        
        # DXF 다운로드 버튼
        if len(pl_lines) > 0:
            dxf_data_clean = create_dxf_file(pl_lines, f"handwriting_removed_lines_{algo}.dxf")
            if dxf_data_clean:
                st.download_button(
                    label="📥 정제된 라인 DXF 다운로드",
                    data=dxf_data_clean,
                    file_name=f"handwriting_removed_lines_{algo}.dxf",
                    mime="application/dxf"
                )

except Exception as e:
    # 에러 시 기본 2열 레이아웃으로 폴백
    st.warning(f"손글씨 제거 처리 중 오류: {e}")
    top_cols = st.columns(2, gap="small")
    with top_cols[0]:
        st.markdown("**📄 원본 이미지**")
        st.image(np.array(image), channels="RGB")
    with top_cols[1]:
        st.markdown("**🔍 라인 추출**")
        st.image(line_only, channels="RGB")
        st.caption(f"{len(lines)}개 라인 검출 · {algo.upper()}")
        
        # DXF 다운로드 버튼
        if len(lines) > 0:
            dxf_data = create_dxf_file(lines, f"lines_{algo}.dxf")
            if dxf_data:
                st.download_button(
                    label="📥 라인 DXF 다운로드",
                    data=dxf_data,
                    file_name=f"lines_{algo}.dxf",
                    mime="application/dxf"
                )

# 손글씨/인쇄문자 나란히 보기 섹션 제거됨 (오류 때문에)

# 사이드바 하단
with st.sidebar:
	st.divider()
	
	# 검출 결과 - 더 간결하고 유용한 정보
	if 'lines' in locals() and lines:
		st.markdown("### 📊 검출 결과")
		result_col1, result_col2 = st.columns(2)
		with result_col1:
			st.metric("라인 개수", f"{len(lines)}개")
		with result_col2:
			if len(lines) > 0:
				avg_length = sum(np.hypot(ln["x1"] - ln["x0"], ln["y1"] - ln["y0"]) for ln in lines) / len(lines)
				st.metric("평균 길이", f"{avg_length:.0f}px")
		
		# 상태별 가이드
		if len(lines) < 10:
			st.warning(f"⚠️ 검출된 선이 적습니다 ({len(lines)}개)")
			st.caption("💡 최소 길이를 낮추거나 LSD 알고리즘 사용 권장")
		elif len(lines) > 5000:
			st.info(f"🔍 많은 선이 검출되었습니다 ({len(lines)}개)")
			st.caption("💡 최소 길이나 임계값 조정으로 필터링 가능")
		else:
			st.success(f"✅ 정상적으로 {len(lines)}개 선 검출됨")
		
	# 도움말은 메인 화면으로 이동됨

# 📚 상세 사용자 가이드 섹션
st.markdown("---")
st.markdown("## 📚 사용자 가이드")

# 탭으로 구성된 상세 가이드
guide_tab1, guide_tab2, guide_tab3, guide_tab4 = st.tabs(["🔍 알고리즘", "⚙️ 매개변수", "✏️ 필기 제거", "🔗 연결성 분석"])

with guide_tab1:
	st.markdown("""
	### 선 검출 알고리즘 비교
	
	**🚀 LSD (Line Segment Detector)**
	- 가장 빠른 검출 속도
	- 일반적인 도면에 최적화
	- 노이즈에 강함
	- **추천**: 대부분의 CAD 도면
	
	**📐 Hough (확률적 허프 변환)**
	- 높은 정확도
	- 복잡한 도면에 효과적
	- 매개변수 조정 가능
	- **추천**: 정밀한 검출이 필요한 경우
	
	**🔧 Morph (형태학 기반)**
	- 노이즈가 많은 도면
	- 얇은 선 강조
	- 전처리 포함
	- **추천**: 스캔 품질이 낮은 도면
	
	**⚡ Fused (융합 알고리즘)**
	- 여러 방법 조합
	- 최고 정확도
	- 처리 시간 증가
	- **추천**: 최고 품질이 필요한 경우
	""")

with guide_tab2:
	st.markdown("""
	### 주요 매개변수 설명
	
	**📏 최소 선 길이**
	- 이보다 짧은 선은 제외
	- 값이 클수록 긴 선만 검출
	- 권장: 이미지 크기의 1-2%
	
	**📊 DPI (해상도)**
	- 높을수록 세밀한 검출
	- 300: 빠른 처리
	- 600: 균형잡힌 품질
	- 1200: 최고 품질 (느림)
	
	**🎯 Canny 임계값**
	- 에지 검출 민감도
	- 하한 50, 상한 150 권장
	- 노이즈 많으면 값 증가
	
	**📐 Hough 임계값**
	- 선 검출 민감도
	- 높을수록 확실한 선만
	- 50-100 권장
	""")

with guide_tab3:
	st.markdown("""
	### 필기/인쇄물 분리 기능
	
	**🖊️ 기본 모드**
	- 일반적인 필기 제거
	- 빠른 처리
	- 대부분의 경우 효과적
	
	**🎨 색상 기반**
	- 색상 차이로 구분
	- 컬러 필기에 효과적
	- HSV 색공간 활용
	
	**✍️ 스트로크 기반**
	- 선의 굵기와 패턴 분석
	- 펜/연필 자국 감지
	- 세밀한 구분 가능
	
	**🔒 엄격 모드**
	- 보수적 제거
	- 도면 손상 최소화
	- 중요한 도면에 권장
	
	**🔄 통합 모드**
	- 여러 방법 조합
	- 최고 정확도
	- 처리 시간 증가
	""")

with guide_tab4:
	st.markdown("""
	### 연결성 분석 활용법
	
	**🔗 연결 허용 거리**
	- 떨어진 선분을 연결
	- 값이 클수록 더 많이 연결
	- 권장: 20-60픽셀
	
	**📏 최소 연결 길이**
	- 연결된 그룹의 최소 길이
	- 너무 짧은 그룹 제외
	- 권장: 전체 길이의 5-10%
	
	**📐 각도 허용 오차**
	- 연결할 선분 간 각도 차이
	- 작을수록 평행한 선만
	- 권장: 15-45도
	
	**📊 최소 세그먼트 비율**
	- 각 세그먼트의 최소 비율
	- 너무 작은 조각 제외
	- 권장: 0.1-0.3
	
	**💡 활용 팁**
	- 복잡한 도면: 엄격한 설정
	- 단순한 도면: 느슨한 설정
	- 점선/파선: 연결 거리 증가
	""")

st.markdown("---")

