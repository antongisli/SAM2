# Aircraft Refueling Detection Pipeline

This repository contains two implementations of a video analysis pipeline for detecting aircraft refueling operations:
1. A modular pipeline with separate components
2. A simplified CLIP-only version

## Setup

```bash
# Clone the repository
git clone https://github.com/antongisli/SAM2.git
cd SAM2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 1. Modular Pipeline

The modular version uses a combination of YOLO, CLIP, and SAM to detect and analyze refueling operations.

### Structure
```
src/pipeline/
├── core/
│   ├── config.py         # Configuration settings
│   ├── pipeline.py       # Main pipeline orchestrator
│   └── video_loader.py   # Video loading utilities
├── stages/
│   ├── detection/
│   │   ├── yolo_detector.py    # Initial object detection
│   │   └── clip_verifier.py    # Object verification
│   └── analysis/
│       └── sam_analyzer.py     # Proximity analysis
└── __init__.py
```

### Usage
```bash
# For local video file
python analyze_video.py input_video.mp4 --skip-frames 1000 --output-dir output

# For YouTube video
python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID" --skip-frames 1000

# Additional options
--skip-frames 1000    # Analyze every 1000th frame
--output-dir output   # Directory for results
--cache-dir cache     # Directory for YouTube video cache
--sam-type vit_b      # SAM model type (vit_h, vit_l, vit_b)
```

## 2. CLIP-Only Pipeline

A simplified version that uses only CLIP to directly detect refueling operations.

### Structure
```
.
├── clip_analyzer.py    # Main CLIP-based detection script
└── src/pipeline/
    └── core/
        └── video_loader.py   # Video loading utilities
```

### Usage
```bash
# For local video
python clip_analyzer.py input_video.mp4 --skip-frames 1000

# For YouTube video
python clip_analyzer.py "https://www.youtube.com/watch?v=VIDEO_ID" --skip-frames 1000

# Additional options
--skip-frames 1000     # Analyze every 1000th frame
--threshold 0.2        # CLIP confidence threshold
--output-dir output    # Directory for results
--cache-dir cache     # Directory for YouTube video cache
```

## 3. Simple Florence Image Analysis

The `simple_florence.py` script provides a simple interface to the Florence-2 vision model, offering multiple modes for analyzing images.

### Usage

```bash
python simple_florence.py <image_path> --mode <mode>
```

Available modes:

1. **objects** - General object detection
```bash
python simple_florence.py image.jpg --mode objects
# Detects common objects like: "truck", "person", "airplane", "wheel"
```

2. **regions** - Salient region detection
```bash
python simple_florence.py image.jpg --mode regions
# Identifies regions of interest without labels
```

3. **caption** - Detailed region descriptions
```bash
python simple_florence.py image.jpg --mode caption
# Provides detailed captions like: "fuel truck parked next to an aircraft"
```

4. **vocab** - Search for specific objects
```bash
python simple_florence.py image.jpg --mode vocab --prompt "fuel truck"
# Finds instances of the specified object (finds one, more accurate)
```

5. **grounding** - Natural language object search
```bash
python simple_florence.py image.jpg --mode grounding --prompt "fuel truck next to the airplane"
# Locates objects based on natural language descriptions
```

### Output

For all modes, the script:
1. Saves an annotated image as `output.jpg`
2. Displays detection results in the terminal, including:
   - Bounding box coordinates
   - Confidence scores
   - Object labels or captions

### Examples

Finding fuel trucks:
```bash
# Using specific object detection
python simple_florence.py airport.jpg --mode vocab --prompt "fuel truck"

# Using natural language search
python simple_florence.py airport.jpg --mode grounding --prompt "fuel truck refueling an airplane"

# Getting detailed scene description
python simple_florence.py airport.jpg --mode caption
```

## Output

Both versions generate:
1. Frame images when refueling is detected
2. JSON file with detection results
3. Detailed logs of the analysis process

The modular version provides more detailed analysis including:
- YOLO detection boxes
- CLIP verification scores
- SAM proximity analysis

The CLIP-only version provides:
- Direct refueling detection scores
- Frame timestamps
- Confidence scores

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- CLIP
- Segment Anything (SAM)
- yt-dlp (for YouTube videos)

See `requirements.txt` for full list of dependencies.

## Notes

- The modular pipeline is more comprehensive but slower
- The CLIP-only version is faster but may miss some details
- Both versions support YouTube videos and local files
- Use `--skip-frames` to control analysis frequency
- Results are saved in the output directory

## License

MIT License - See LICENSE file for details
