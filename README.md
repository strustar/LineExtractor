# 📐 CAD Line Extraction System

## 🎯 프로젝트 개요

CAD 도면에서 선분을 자동으로 검출하고 시각화하는 웹 기반 시스템입니다. PDF 도면을 업로드하면 여러 알고리즘으로 선분을 검출하고, 필기와 인쇄물을 분리하여 정확한 결과를 제공합니다.

## 🚀 주요 기능

- **📄 PDF 처리**: 고해상도 PDF → 이미지 변환
- **🔍 다중 알고리즘**: LSD, Hough, Morph, Fused 선택 가능
- **✏️ 필기 분리**: 5가지 모드로 손글씨/인쇄물 구분
- **🔗 연결성 분석**: 끊어진 선분 자동 연결
- **📊 실시간 시각화**: 색상별 길이 매핑 및 라이브 프리뷰
- **📚 상세 가이드**: 탭별 사용법 및 매개변수 설명

## 🌐 온라인 데모

**Streamlit Cloud에서 바로 사용하기:**  
[https://cad-line-extraction.streamlit.app](https://your-app-url.streamlit.app)

## 🛠 로컬 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/cad-line-extraction.git
cd cad-line-extraction
```

### 2. 가상환경 설정
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 실행
```bash
streamlit run app.py
```

## 🎮 사용법

### 1. PDF 업로드
- 사이드바에서 CAD 도면 PDF 파일 선택
- 지원 형식: PDF (멀티페이지 지원)

### 2. 알고리즘 선택
- **LSD**: 가장 빠름, 일반 도면 추천
- **Hough**: 높은 정확도, 복잡한 도면
- **Morph**: 노이즈가 많은 도면  
- **Fused**: 최고 품질, 처리시간 증가

### 3. 매개변수 조정
- **DPI**: 300(빠름) / 600(균형) / 1200(고품질)
- **최소 길이**: 이미지 크기의 1-2% 권장
- **상세 설정**: Canny, Hough 임계값 세부 조정

### 4. 필기 제거 (선택사항)
- **기본 모드**: 일반적인 필기 제거
- **색상/스트로크 기반**: 정밀한 구분
- **엄격/통합 모드**: 용도에 맞게 선택

### 5. 연결성 분석
- 끊어진 선분 자동 연결
- 연결 거리, 각도 허용 오차 조정
- 복잡한 도면일수록 엄격한 설정 권장

## 📊 성능 특성

- **처리 시간**: 간단한 도면 5-15초, 복잡한 도면 20-60초
- **지원 해상도**: 최대 6000px (자동 다운스케일링)
- **메모리 최적화**: Streamlit Cloud 1GB 제한에 맞춰 최적화

## 📁 프로젝트 구조

```
cad-line-extraction/
├── app.py                # Streamlit 웹 인터페이스
├── requirements.txt      # 의존성 목록
├── src/
│   ├── main.py          # 배치 처리 엔진
│   ├── pdf_processor.py # PDF 처리
│   ├── image_utils.py   # 이미지 전처리
│   ├── exceptions.py    # 예외 처리
│   └── memory_manager.py# 메모리 관리
└── outputs/             # 결과 출력 폴더
```

## 🤝 기여하기

이슈나 개선사항이 있으시면 GitHub Issues에서 제안해 주세요.

## 📄 라이선스

MIT License

---

**Made with ❤️ for CAD Engineers**