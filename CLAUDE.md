# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **CAD Line Extraction System** - a comprehensive tool for extracting and visualizing line segments from engineering drawings (CAD/PDF files). The system combines multiple computer vision algorithms with real-time handwriting/text separation capabilities through a Streamlit web interface.

**Core Functionality:**
- PDF to image conversion with high-DPI rendering
- Multiple line detection algorithms (LSD, Hough, Morphology)
- Real-time handwriting vs. printed text separation
- Interactive line visualization with color-coded length mapping
- Batch processing with memory optimization

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main Streamlit web interface
streamlit run app.py

# Run batch processing on PDFs
python src/main.py

# Type checking
mypy src/ app.py
```

### Testing Commands
```bash
# Process sample documents (if available)
python src/main.py

# Memory profiling during processing
python -m psutil src/main.py
```

## Architecture Overview

### Core Processing Pipeline
1. **PDF Rendering** → High-DPI image conversion (600 DPI default)
2. **Image Preprocessing** → Deskewing, grayscale conversion, adaptive binarization
3. **Line Detection** → Multiple algorithms (LSD, HoughP, Morphological)
4. **Handwriting Separation** → Color-based and stroke-based mask generation
5. **Visualization** → Real-time line overlay with length-based coloring

### Key Modules

**`app.py`** - Main Streamlit interface
- Real-time PDF processing and visualization
- Interactive parameter adjustment for line detection
- Side-by-side comparison of original vs. processed images
- Live handwriting extraction preview

**`src/main.py`** - Batch processing engine
- Multi-page PDF processing pipeline
- Memory-optimized image handling
- Configurable algorithm parameters
- JSON output for detected lines

**`src/exceptions.py`** - Error handling framework
- Domain-specific exceptions (PDFProcessingError, LineDetectionError, etc.)
- Korean error messages with context information
- Structured error codes for debugging

**Support Modules:**
- `src/pdf_processor.py` - PDF rendering and page extraction
- `src/image_utils.py` - Image preprocessing utilities
- `src/memory_manager.py` - Memory optimization and resource management
- `src/config_validator.py` - Configuration validation

### Line Detection Algorithms

**LSD (Line Segment Detector)** - OpenCV's built-in fast line detection
**HoughP (Probabilistic Hough Transform)** - Classical line detection with parameter tuning
**Morphology-based** - Shape-based line enhancement before Hough detection
**Fused** - Combination of multiple algorithms for robust detection

### Output Structure
```
outputs/lines/
├── page_000/
│   ├── original.png          # Source image
│   ├── lines_lsd.json       # LSD detection results
│   ├── lines_hough.json     # Hough detection results
│   ├── hand_only.png        # Extracted handwriting
│   ├── printed_only.png     # Extracted printed text
│   └── mask_*.png          # Detection masks
└── page_001/...
```

## Configuration

### Default Processing Parameters
- **PDF DPI**: 600 (configurable via UI slider)
- **Minimum line length**: Dynamic based on image size (1% of min dimension)
- **Canny edge thresholds**: Low=50, High=150
- **HoughP parameters**: threshold=50, maxGap=4

### Performance Limits
- **Memory limit**: 2GB (configurable in PERFORMANCE_CONFIG)
- **Max image dimension**: 6000px
- **Processing timeout**: 300 seconds

## Key Features

### Real-time Processing
- Live parameter adjustment in Streamlit interface
- Instant visual feedback on line detection changes
- Real-time handwriting extraction with multiple mask modes

### Multi-Algorithm Support
- Algorithm comparison through UI radio buttons
- Automatic fallback to lower thresholds if no lines detected
- Performance-optimized algorithm selection

### Handwriting Separation
- **Stroke-based**: Detects pen strokes vs. printed lines
- **Color-based**: Uses HSV saturation and RGB spread analysis
- **Strict mode**: Conservative extraction for clean separation
- **Union mode**: Combines multiple detection methods

### Memory Management
- Automatic image downsizing for processing
- Process pool executor for batch operations
- Memory usage monitoring and optimization

## Development Notes

### Exception Handling Pattern
All processing functions use domain-specific exceptions with Korean error messages. Always wrap OpenCV and PDF operations in try-catch blocks using the appropriate exception types.

### Image Processing Convention
- Input images are converted from BGR to RGB immediately after loading
- All internal processing uses RGB format
- Grayscale conversion uses OpenCV's COLOR_RGB2GRAY

### Configuration System
The system supports both default configuration and user-defined `config.py` files. New parameters should be added to the appropriate config section (PDF_CONFIG, IMAGE_CONFIG, etc.).

### Performance Considerations
- Images larger than 6000px are automatically downscaled
- PDF rendering defaults to 600 DPI but can be reduced for performance
- Batch processing uses process pools for parallel execution

### Testing Data
- Sample PDFs: `샘플도면.pdf`, `풀도면.pdf`
- The system auto-processes sample documents on first run
- Reference images and task specifications are in Korean language files