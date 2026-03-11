# CAVC: Content-Aware Video Compression

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch GPU](https://img.shields.io/badge/PyTorch-CUDA%20Supported-orange)
![FFmpeg NVENC](https://img.shields.io/badge/FFmpeg-NVENC-success)
![Research](https://img.shields.io/badge/Status-Research%20Validation-lightgrey)

**Content-Aware Video Compression (CAVC)** is a specialized, GPU-accelerated video compression pipeline designed to optimize bandwidth by intelligently separating a video into a **Region of Interest (ROI)** (e.g., persons, vehicles) and a **Non-Region of Interest (N-ROI)** (e.g., the background).

By applying varying compression strengths (Dual-CRF) to these separate streams—preserving the foreground in high quality while aggressively compressing the background—CAVC achieves significant bitrate savings with minimal perceived structural quality loss.

## 🌟 Key Features

- **End-to-End Pipeline**: Fully integrates YOLOv8 Detection -> Mask Generation -> Stream Separation -> Dual-CRF Encoding -> Reconstruction.
- **Hardware Acceleration**: Built with performance in mind using GPU-accelerated PyTorch tensors and **FFmpeg NVENC** (`h264_nvenc` / `hevc_nvenc`) for blazing-fast encoding.
- **Dual-CRF Mechanism**: Independent compression control via Constant Rate Factor (CRF). e.g., `ROI_CRF=25` (High Quality) and `NROI_CRF=40` (High Compression).
- **Advanced Mask Smoothing**: Employs morphological operations (erosion/dilation) and temporal EMA smoothing to prevent ROI flickering.
- **Robust Quality Metrics Engine**: Automatically calculates per-frame **PSNR**, **SSIM**, analyzes compression ratios, bitrate savings, and generates detailed visualizations (Heatmaps, Artifact Maps, Comparison Graphs).
- **Configurable Control**: Easy adjustments for input resolution resizing and FPS downsampling to maximize performance and validation.

## 🏗️ Architecture & Processing Stages

The pipeline follows a modular 4-stage architecture, validated in research contexts (e.g., IEEE T-ITS):

```mermaid
graph TD
    A[Input Video] --> B[S0: ROI Detection<br>(YOLOv8)]
    
    subgraph "CAVC Core Pipeline"
        B --> C[S1: Mask Generation<br>(Morphological & EMA)]
        C --> D[S2: Stream Separation<br>(Foreground vs Background)]
        D --> E[S3: Dual-CRF Encoding<br>(FFmpeg NVENC)]
    end
    
    E --> F[ROI Stream HQ .mp4]
    E --> G[N-ROI Stream LQ .mp4]
    
    F & G --> H[Stream Merger<br>(Reconstruction)]
    H --> I[Compressed Output .mp4]
    
    I --> J[Metrics & Validation]
    J --> K[PSNR/SSIM CSV]
    J --> L[Visualizations Heatmaps]
```

## 📂 Project Structure

```text
📦 CAVC-Project
 ┣ 📜 run_cavc_pipeline.py      # 🚀 Main Entry Point: runs the whole process
 ┣ 📜 cavc_bridge.py            # Orchestrates between entry script and core pipeline
 ┣ 📜 VIDEO_FOR_TESTING.mp4     # Input video file
 ┣ 📜 RUN_INSTRUCTIONS.md       # Quick start guide
 ┣ 📜 Project detail            # In-depth architectural documentation
 ┣ 📜 diagnose_psnr.py          # Script for debugging PSNR calculations
 ┗ 📂 Taimoor folder            # Contains the core engine
    ┗ 📂 cavc                   # The CAVC Core Library
       ┣ 📜 pipeline.py         # Main pipeline integrating S0-S3
       ┣ 📜 encoder.py          # FFmpeg Dual-CRF implementation
       ┣ 📜 mask_generator.py   # Mask creation and temporal smoothing
       ┣ 📜 stream_separator.py # Video stream splitting logic
       ┣ 📜 stream_merger.py    # Merges ROI & N-ROI back together
       ┣ 📜 metrics.py          # Detailed PSNR, SSIM, and plot generation
       ┗ 📜 parallel_pipe.py    # Parallel implementation of the pipeline
```

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python**: 3.8 or higher.
- **PyTorch**: Recommended to install with CUDA support `.to('cuda')`.
- **OpenCV**: `opencv-python` for frame manipulations.
- **Ultralytics**: `ultralytics` package for YOLOv8 models.
- **FFmpeg**: Must be installed and added to your system's `PATH`. For hardware acceleration on NVIDIA GPUs, ensure your FFmpeg build supports NVENC (`h264_nvenc`).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TaimoorHameedi/Deeplearning-based-content-aware-video-compression-.git
   cd Deeplearning-based-content-aware-video-compression-
   ```

2. **Install Python dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version
   pip install opencv-python ultralytics numpy
   ```

3. **Provide YOLO weights:**
   The process requires a YOLO weights file (e.g., `yolo11n.pt` or custom `best.pt`). The bridge will attempt to auto-detect model paths or fallback to a standard YOLO variation.

## 🚀 Usage

### Quick Start

Place your test video named `VIDEO_FOR_TESTING.mp4` in the project root folder. Then simply execute:

```bash
python run_cavc_pipeline.py
```

### Configuration Parameters

You can modify the compression behavior directly inside `run_cavc_pipeline.py` by adjusting the global variables:

```python
# Options: (1920, 1080), (1280, 720), (640, 360), (426, 240)
PROCESSING_RESOLUTION = (1280, 720) 

# Region of Interest CRF (Lower is higher quality - Default: 25)
ROI_CRF = 25

# Non-Region of Interest CRF (Higher is more compression - Default: 40)
NROI_CRF = 40

# Framerate Downsampling (Set int for target FPS, e.g., 15. None for original FPS)
INPUT_FPS = None
```

## 📊 Outputs & Expected Results

After running the pipeline, the following artifacts will be generated:

1. **`temp_cavc/` directory**:
   - `cavc_compressed.mp4`: The final reconstructed Content-Aware compressed video.
   - `roi_stream.mp4`: The isolated foreground stream (High Quality).
   - `nroi_stream.mp4`: The isolated background stream (Low Quality).

2. **`cavc_quality_report.csv`**:
   - A frame-by-frame breakdown of PSNR, SSIM, Compression Ratios, and bitrates.

3. **`visualizations/` directory**:
   - `psnr_graph.png`: Graphical tracking of video quality over time.
   - `ssim_heatmap_frame_*.png`: Graphical heatmaps showing exactly where structural changes occurred.
   - `artifact_map_frame_*.png`: Detailed visualizations highlighting differences/compression artifacts.


