# How to Run the CAVC Pipeline

## Quick Start

Open a terminal and navigate to the project directory:

```bash
cd "c:\Users\SYSLAB\Downloads\Tamoor\Tamoor"
python run_cavc_pipeline.py
```

**What happens:**
- Locates `VIDEO_FOR_TESTING.mp4`.
- Compresses the video using Content-Aware Video Compression (CAVC) which separates ROI (foreground) and NROI (background).
- Re-merges the streams into `temp_cavc/cavc_compressed.mp4`.
- Generates a detailed quality report `cavc_quality_report.csv`.
- Generates visualizations in the `visualizations/` folder.

**Expected Output Files:**
```
temp_cavc/
├── cavc_compressed.mp4           # ✅ CAVC compressed video
├── roi_stream.mp4                # ✅ High-quality ROI
└── nroi_stream.mp4               # ✅ Compressed background

visualizations/
├── psnr_graph.png
├── ssim_heatmap_frame_*.png
└── artifact_map_frame_*.png

cavc_quality_report.csv
```

## Configuration

You can change compression parameters and processing behavior by editing `run_cavc_pipeline.py`:

```python
PROCESSING_RESOLUTION = (1280, 720) 
ROI_CRF = 25   # Lower is better quality (Foreground)
NROI_CRF = 40  # Higher is more compression (Background)
INPUT_FPS = None # Set to target FPS to downsample, or None for original FPS
```
