"""
CAVC Bridge Module
Integrates Content-Aware Video Compression (CAVC) for ROI-based research.

This module supports:
- File-based video input (current implementation)
- Future RTSP stream input (modular design for easy extension)
"""

import sys
import os
import subprocess
from pathlib import Path
import shutil
import time
# HLS Converter removed per user request


class CAVCBridge:
    """
    Bridge for CAVC compression orchestration.
    """
    
    def __init__(self, project_root=None, weights_path=None):
        """
        Initialize the CAVC bridge.
        
        Args:
            project_root: Path to project root (auto-detected if None)
            weights_path: Path to YOLO weights (auto-detected if None)
        """
        # Auto-detect project root
        if project_root is None:
            current_dir = Path(os.getcwd())
            if (current_dir / "Taimoor folder").exists():
                project_root = current_dir / "Taimoor folder"
            elif current_dir.name == "Tamoor":
                project_root = current_dir / "Taimoor folder"
            else:
                project_root = current_dir
        
        self.project_root = Path(project_root)
        self.temp_dir = Path("temp_cavc")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Setup Python path for CAVC imports
        self._setup_python_path()
        
        # Import CAVC pipeline
        self._import_cavc_pipeline()
        
        # Find weights
        self.weights_path = self._find_weights(weights_path)
        
        # Initialize CAVC pipeline
        self.cavc_pipeline = None
        
        # HLS converter removed
        
        print(f"✅ CAVC Bridge initialized")
        print(f"   Project Root: {self.project_root}")
        print(f"   Weights: {self.weights_path}")
    
    def _setup_python_path(self):
        """Setup Python path to import CAVC modules."""
        ultralytics_path = str(self.project_root / "ultralytics")
        project_path = str(self.project_root)
        
        # Remove old imports
        modules_to_kill = [m for m in sys.modules if m.startswith('ultralytics') or m.startswith('cavc')]
        for m in modules_to_kill:
            if m in sys.modules:
                del sys.modules[m]
        
        # Add to path
        if ultralytics_path not in sys.path:
            sys.path.insert(0, ultralytics_path)
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
    
    def _import_cavc_pipeline(self):
        """Import CAVC pipeline modules."""
        try:
            import importlib
            import cavc.pipeline
            import cavc.parallel_pipe
            importlib.reload(cavc.pipeline)
            importlib.reload(cavc.parallel_pipe)
            from cavc.pipeline import CAVCPipeline
            from cavc.parallel_pipe import ParallelCAVCPipeline
            self.CAVCPipeline = CAVCPipeline
            self.ParallelCAVCPipeline = ParallelCAVCPipeline
            print("✅ CAVC pipelines imported successfully")
        except ImportError as e:
            print(f"❌ Error: Could not import CAVC pipeline")
            print(f"   Make sure {self.project_root}/cavc exists")
            raise e
    
    def _find_weights(self, weights_path=None):
        """Find specialized YOLO weights file for CAVC."""
        if weights_path and Path(weights_path).exists():
            return str(weights_path)
        
        # Search in specific research locations
        possible_paths = [
            self.project_root.parent / "runs" / "CAVC_Final_Training" / "weights" / "best.pt",
            self.project_root / "runs" / "CAVC_Final_Training" / "weights" / "best.pt",
            Path("/home/FASTLAB2/Tamoor/runs/CAVC_Final_Training/weights/best.pt")
        ]
        
        for p in possible_paths:
            if p.exists():
                return str(p)
        
        # Fallback to default YOLOv8 as requested
        print("⚠️ WARNING: Custom weights not found, using default yolov8n.pt")
        return "yolov8n.pt"
    
    def initialize_cavc(self, roi_crf=18, nroi_crf=35, use_parallel=True, codec="h264_nvenc"):
        """
        Initialize CAVC pipeline with specified CRF values.
        
        Args:
            roi_crf: CRF for ROI stream (lower = higher quality)
            nroi_crf: CRF for NROI stream (higher = more compression)
            use_parallel: Whether to use the new parallel architecture
            codec: Video codec for encoding (h264_nvenc, hevc_nvenc, libx264, etc.)
        """
        # Auto-fallback for local testing if NVENC isn't supported (e.g. no GPU or outdated driver)
        if "nvenc" in codec:
            print(f"🔍 Checking {codec} hardware availability...")
            try:
                import subprocess
                cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=128x128:d=0.1", "-c:v", codec, "-f", "null", "-"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"⚠️ {codec} failed local check (likely outdated driver or no GPU). Falling back to CPU encoder.")
                    codec = "libx265" if "hevc" in codec or "265" in codec else "libx264"
                else:
                    print(f"✅ {codec} is fully supported by system.")
            except Exception:
                pass

        print(f"🚀 Initializing CAVC Pipeline...")
        print(f"   ROI CRF: {roi_crf} (High Quality)")
        print(f"   NROI CRF: {nroi_crf} (High Compression)")
        print(f"   Parallel Mode: {use_parallel}")
        print(f"   Codec: {codec}")
        
        if use_parallel:
            self.cavc_pipeline = self.ParallelCAVCPipeline(
                self.weights_path,
                roi_crf=roi_crf,
                nroi_crf=nroi_crf,
                codec=codec
            )
        else:
            self.cavc_pipeline = self.CAVCPipeline(
                self.weights_path,
                roi_crf=roi_crf,
                nroi_crf=nroi_crf,
                codec=codec
            )
        
        self.use_parallel = use_parallel
        mode_str = "Parallel Mode" if use_parallel else "Sequential Mode"
        print(f"✅ CAVC Pipeline ready ({codec} | {mode_str})")
    
    def compress_with_cavc(self, input_video, limit_frames=None, resize_res=None, target_fps=None):
        """
        Stage 1: Compress video using CAVC pipeline.
        
        Args:
            input_video: Path to input video file
            limit_frames: Limit processing to N frames (None = all)
            
        Returns:
            dict with CAVC results including output path
        """
        if self.cavc_pipeline is None:
            self.initialize_cavc()
        
        input_path = Path(input_video)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        # Output path in temp directory
        output_name = "cavc_compressed.mp4"
        output_path = str(self.temp_dir / output_name)
        
        print(f"\n{'='*60}")
        print(f"STAGE 1: CAVC COMPRESSION (GPU Accelerated)")
        print(f"{'='*60}")
        print(f"📹 Input: {input_path.name}")
        print(f"💾 Output: {output_path}")
        
        # Run CAVC pipeline
        start_time = time.perf_counter()
        results = self.cavc_pipeline.process_video(
            str(input_path),
            output_path,
            limit_frames=limit_frames,
            resize_res=resize_res,
            target_fps=target_fps
        )
        end_time = time.perf_counter()
        
        # Calculate timing and FPS
        processing_time = end_time - start_time
        frame_count = results.get('frame_count', 0)
        achieved_fps = frame_count / processing_time if processing_time > 0 else 0
        
        results['processing_time'] = processing_time
        results['achieved_fps'] = achieved_fps
        
        # Store for merged comparison later
        self._last_input_video = str(input_path)
        
        # Copy streams to temp directory for reference
        if 'roi_stream' in results:
            shutil.copy(results['roi_stream'], self.temp_dir / "roi_stream.mp4")
        if 'nroi_stream' in results:
            shutil.copy(results['nroi_stream'], self.temp_dir / "nroi_stream.mp4")
        
        print(f"\n✅ CAVC Compression Complete!")
        print(f"   Compression Ratio: {results.get('ratio_x', 0):.1f}x")
        print(f"   Bitrate Savings: {results.get('savings', 0):.1f}%")
        print(f"   Processing Time: {processing_time:.2f}s ({achieved_fps:.2f} FPS)")
        
        return results
    
    # HLS conversion methods removed
    
    def process_pure_cavc(self, input_video, limit_frames=None, resize_res=None, target_fps=None):
        """
        Orchestrates pure CAVC compression without HLS.
        """
        print(f"\n{'='*60}")
        print(f"CAVC ROI-BASED VIDEO COMPRESSION (GPU)")
        print(f"{'='*60}")
        print(f"Input: {input_video}")
        print(f"{'='*60}\n")
        
        # Stage 1: CAVC Compression
        cavc_results = self.compress_with_cavc(input_video, limit_frames, resize_res, target_fps=target_fps)
        
        return {
            "cavc": cavc_results,
            "stream_name": "PURE_CAVC_OUTPUT"
        }
    
    def cleanup(self, keep_cavc_output=True):
        """
        Clean up temporary files.
        
        Args:
            keep_cavc_output: If True, keep CAVC compressed video
        """
        if not keep_cavc_output and self.temp_dir.exists():
            print(f"🧹 Cleaning up temporary files...")
            shutil.rmtree(self.temp_dir)
            print("✅ Cleanup complete")
        else:
            print(f"📁 Temporary files preserved in: {self.temp_dir}")
    
    # Playback methods removed


# Future enhancement: RTSP stream support
class RTSPCAVCBridge(CAVCBridge):
    """
    Extended bridge for real-time RTSP stream processing.
    TODO: Implement in Phase 4
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("⚠️ RTSP support not yet implemented - use file-based processing")
    
    def process_rtsp_to_cavc(self, rtsp_url, stream_name):
        """
        Future: Process RTSP stream → CAVC in real-time.
        """
        raise NotImplementedError("RTSP support coming in Phase 4")


if __name__ == "__main__":
    print("CAVC Bridge Module - Ready for GPU-Accelerated ROI Compression")
    print("Use CAVCBridge for file-based processing")
