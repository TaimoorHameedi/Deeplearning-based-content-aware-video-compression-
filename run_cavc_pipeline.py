import os
import sys
from pathlib import Path
from cavc_bridge import CAVCBridge
import cv2
import numpy as np


# Options: (1920, 1080), (1280, 720), (640, 360), (426, 240)

PROCESSING_RESOLUTION = (1280, 720) 
# -----------------------------------------------
# ROI (Persons): 18 = High quality but efficient
ROI_CRF = 25
# NROI (Background): 50 = Extreme efficiency
NROI_CRF = 40
# ---------------------------------------------------
# FPS Control: Set to desired FPS (e.g., 15) or None for original
INPUT_FPS = None
# ---------------------------------------------------


def find_input_video():
    """Find the test video file."""
    video_name = "VIDEO_FOR_TESTING.mp4"
    
    possible_paths = [
        Path(video_name),
        Path("Taimoor folder") / video_name,
        Path("..") / video_name,
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(
        f"Could not find {video_name}. "
        f"Please ensure it's in the current directory or Taimoor folder."
    )


def main():
    """Main execution function."""
    
    print(" CAVC ROI-BASED VIDEO COMPRESSION")
    print(" Research Validation Pipeline (Independent of HLS)")
    print("="*70)
    print()
    
    # Step 1: Find input video
    print("Step 1: Locating input video...")
    try:
        input_video = find_input_video()
        cap = cv2.VideoCapture(input_video)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"   OK: Found: {input_video}")
        print(f"   Video Stats: {orig_w}x{orig_h} | {orig_fps:.2f} FPS | {orig_total} Total Frames")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        return
    
    # Step 2: Initialize bridge
    print("\nStep 2: Initializing CAVC Bridge...")
    try:
        bridge = CAVCBridge()
        # Apply custom CRF values for ROI and NROI
        bridge.initialize_cavc(roi_crf=ROI_CRF, nroi_crf=NROI_CRF)
    except Exception as e:
        print(f"   Error initializing bridge: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("   OK: Resizing enabled to 1280x720")
    
    # Step 3: Process video through CAVC pipeline
    print("\nStep 3: Running CAVC Compression...")
    print()
    
    try:
        results = bridge.process_pure_cavc(
            input_video=input_video,
            limit_frames=None,  # Full analysis
            resize_res=PROCESSING_RESOLUTION, # Apply user-configurable resizing
            target_fps=INPUT_FPS # Apply user-configurable FPS control
        )
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Display results
    print("\n" + "="*70)
    print(" CAVC COMPRESSION COMPLETE - RESULTS")
    print("="*70)
    
    # CAVC Results
    cavc = results['cavc']
    print("\n📊 CAVC Compression Results:")
    print(f"   • Compression Ratio: {cavc.get('ratio_x', 0):.1f}x")
    print(f"   • Bitrate Savings: {cavc.get('savings', 0):.1f}%")
    print(f"   • Original Bitrate: {cavc.get('orig_bitrate_kbps', 0):.0f} kbps")
    print(f"   • CAVC Bitrate: {cavc.get('final_bitrate_kbps', 0):.0f} kbps")
    print(f"   • ROI Stream: {cavc.get('roi_kb', 0):.1f} KB")
    print(f"   • NROI Stream: {cavc.get('nroi_kb', 0):.1f} KB")
    print(f"   • Total CAVC Processing Time: {cavc.get('processing_time', 0):.2f} seconds")
    print(f"   • CAVC Processing Speed:      {cavc.get('achieved_fps', 0):.2f} FPS")
    
    # New: Frame Delay and FPS Setting
    delays = cavc.get('per_frame_delays', [])
    if delays:
        avg_delay = np.mean(delays) * 1000 # Convert to ms
        print(f"   • Avg Per-Frame Delay:        {avg_delay:.2f} ms")
    
    fps_set = cavc.get('fps_setting', 0)
    orig_fps_val = cavc.get('original_fps', 0)
    if orig_fps_val > 0 and fps_set < orig_fps_val:
        print(f"   • Current FPS Setting:        {fps_set:.2f} FPS (Downsampled from {orig_fps_val:.2f})")
    else:
        print(f"   • Current FPS Setting:        {fps_set:.2f} FPS")
    
    if 'merged_ratio' in cavc:
        print("\n🖼️  Merged Reconstruction Results (Visual Output):")
        print(f"   • Merged File Size: {cavc.get('merged_kb', 0):.1f} KB")
        print(f"   • Merged Compression Ratio: {cavc.get('merged_ratio', 0):.1f}x")
        print(f"   • Merged Bitrate Savings: {cavc.get('merged_savings', 0):.1f}%")
    
    # HLS results removed
    
    # Quality Metrics Assessment
    print("\nQuality Metrics Analysis (PSNR/SSIM)...")
    from cavc.metrics import DetailedMetricsTracker
    tracker = DetailedMetricsTracker()
    
    # Calculate sampling interval for aligned evaluation
    if INPUT_FPS is not None:
        sampling_interval = max(1, round(orig_fps / INPUT_FPS))
    else:
        sampling_interval = 1
    
    # Files to compare
    roi_stream = "temp_cavc/roi_stream.mp4"
    nroi_stream = "temp_cavc/nroi_stream.mp4"
    merged_video = "temp_cavc/cavc_compressed.mp4"
    
    # 1. ROI Quality (Masked)
    cap_orig = cv2.VideoCapture(input_video)
    cap_roi = cv2.VideoCapture(roi_stream)
    roi_psnrs, roi_ssims, roi_coverages = [], [], []
    
    print("   Evaluating ROI Stream...")
    while True:
        # Read from ROI stream (this only has the processed frames)
        ret_r, f_r = cap_roi.read()
        if not ret_r: break
        
        # Read from Original stream
        ret_o, f_o = cap_orig.read()
        if not ret_o: break
        
        # PROCESS: Calculate metrics for the ALIGNED frame
        f_o = cv2.resize(f_o, PROCESSING_RESOLUTION)
        f_r = cv2.resize(f_r, PROCESSING_RESOLUTION)
        mask = cv2.cvtColor(f_r, cv2.COLOR_BGR2GRAY)
        
        _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        coverage = (np.sum(mask_bin > 0) / mask_bin.size) * 100
        roi_coverages.append(coverage)
        
        roi_psnrs.append(tracker.calculate_psnr(f_o, f_r, mask=mask_bin))
        roi_ssims.append(tracker.calculate_ssim(f_o, f_r, mask=mask_bin))

        # DRAIN/SKIP: Move cap_orig to the next sampled position
        for _ in range(sampling_interval - 1):
            cap_orig.read()
    
    cap_orig.release(); cap_roi.release()
    
    # 2. NROI Quality (Masked)
    cap_orig = cv2.VideoCapture(input_video)
    cap_nroi = cv2.VideoCapture(nroi_stream)
    nroi_psnrs, nroi_ssims = [], []
    
    print("   Evaluating NROI Stream...")
    while True:
        ret_n, f_n = cap_nroi.read()
        if not ret_n: break
        
        ret_o, f_o = cap_orig.read()
        if not ret_o: break
        
        f_o = cv2.resize(f_o, PROCESSING_RESOLUTION)
        f_n = cv2.resize(f_n, PROCESSING_RESOLUTION)
        mask = cv2.cvtColor(f_n, cv2.COLOR_BGR2GRAY)
        mask = (mask > 5).astype(np.uint8)
        
        nroi_psnrs.append(tracker.calculate_psnr(f_o, f_n, mask=mask))
        nroi_ssims.append(tracker.calculate_ssim(f_o, f_n, mask=mask))

        # DRAIN/SKIP
        for _ in range(sampling_interval - 1):
            cap_orig.read()
        
    cap_orig.release(); cap_nroi.release()
    
    # 3. Merged Quality (Full Frame)
    cap_orig = cv2.VideoCapture(input_video)
    cap_mrg = cv2.VideoCapture(merged_video)
    mrg_psnrs, mrg_ssims = [], []
    
    # To store representative frame for visualizations
    worst_psnr = float('inf')
    rep_frame_idx = -1
    rep_f_o = None
    rep_f_m = None

    print("   Evaluating Merged Reconstruction...")
    while True:
        ret_m, f_m = cap_mrg.read()
        if not ret_m: break
        
        ret_o, f_o = cap_orig.read()
        if not ret_o: break
        
        f_o = cv2.resize(f_o, PROCESSING_RESOLUTION)
        f_m = cv2.resize(f_m, PROCESSING_RESOLUTION)
        
        m_psnr = tracker.calculate_psnr(f_o, f_m)
        m_ssim = tracker.calculate_ssim(f_o, f_m)
        
        mrg_psnrs.append(m_psnr)
        mrg_ssims.append(m_ssim)
        
        # Record for tracker plots
        idx = len(mrg_psnrs)-1
        tracker.record_frame_metrics(idx, {
            'roi_psnr': roi_psnrs[min(len(roi_psnrs)-1, idx)],
            'nroi_psnr': nroi_psnrs[min(len(nroi_psnrs)-1, idx)],
            'overall_psnr': m_psnr,
            'roi_percentage': roi_coverages[min(len(roi_coverages)-1, idx)]
        })

        # Track worst frame for representative visualizations
        if m_psnr < worst_psnr:
            worst_psnr = m_psnr
            rep_frame_idx = idx
            rep_f_o = f_o.copy()
            rep_f_m = f_m.copy()

        # DRAIN/SKIP
        for _ in range(sampling_interval - 1):
            cap_orig.read()
        
    cap_orig.release(); cap_mrg.release()
    
    # 4. Compression Ratio (Per Frame)
    print("   Extracting Bitrate Statistics...")
    from cavc.metrics import BitrateAnalyzer
    orig_sizes = BitrateAnalyzer.get_per_frame_sizes(input_video)
    roi_sizes = BitrateAnalyzer.get_per_frame_sizes(roi_stream)
    nroi_sizes = BitrateAnalyzer.get_per_frame_sizes(nroi_stream)
    
    # Ensure minimum shared frame count
    num_frames = min(len(orig_sizes), len(roi_sizes), len(nroi_sizes), len(mrg_psnrs))
    
    for idx in range(num_frames):
        o_size = orig_sizes[idx]
        c_size = roi_sizes[idx] + nroi_sizes[idx]
        
        ratio = o_size / c_size if c_size > 0 else 0
        
        # Update existing record (tracker.record_frame_metrics was called in Merged Quality loop)
        # We need to find the correct dict and update it, but record_frame_metrics appends.
        # Let's fix the loop above to include ratio if possible, or just update the list.
        if idx < len(tracker.frame_metrics):
            tracker.frame_metrics[idx]['compression_ratio'] = ratio
    
    # Display Summary
    print("\n" + "="*70)
    print(" FINAL QUALITY METRICS")
    print("="*70)
    print(f"   • ROI Stream (Masked):  {np.mean(roi_psnrs):.2f} dB | SSIM: {np.mean(roi_ssims):.4f}")
    print(f"   • NROI Stream (Masked): {np.mean(nroi_psnrs):.2f} dB | SSIM: {np.mean(nroi_ssims):.4f}")
    print(f"   • Merged Video (Full):  {np.mean(mrg_psnrs):.2f} dB | SSIM: {np.mean(mrg_ssims):.4f}")
    print("="*70)
    
    # Save Reports
    try:
        tracker.save_detailed_report_csv("cavc_quality_report.csv")
        print(f"   Report saved: cavc_quality_report.csv")
    except PermissionError:
        alt_name = f"cavc_quality_report_at_fps_{INPUT_FPS or 'orig'}.csv"
        tracker.save_detailed_report_csv(alt_name)
        print(f"   ⚠️ WARNING: 'cavc_quality_report.csv' is open. Saved to: {alt_name}")

    # 6. Generate Visualizations
    print("\nStep 6: Generating Quality Visualizations...")
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # 6.1 PSNR Comparison Graph
    psnr_plot_path = viz_dir / "psnr_graph.png"
    tracker.plot_psnr_comparison(str(psnr_plot_path))
    
    # 6.2 SSIM Heatmap and Artifact Map for worst frame
    if rep_f_o is not None and rep_f_m is not None:
        ssim_heatmap_path = viz_dir / f"ssim_heatmap_frame_{rep_frame_idx}.png"
        artifact_map_path = viz_dir / f"artifact_map_frame_{rep_frame_idx}.png"
        
        tracker.plot_ssim_heatmap(rep_f_o, rep_f_m, rep_frame_idx, str(ssim_heatmap_path))
        tracker.plot_artifact_map(rep_f_o, rep_f_m, rep_frame_idx, str(artifact_map_path))
        print(f"   Visualizations saved for Frame {rep_frame_idx} (PSNR: {worst_psnr:.2f} dB)")

    
    print("Analysis complete. Review the CAVC metrics above.")
    
    # File locations
    print("\n📁 Output File Locations:")
    print(f"   • CAVC Compressed: temp_cavc/cavc_compressed.mp4")
    print(f"   • ROI Stream: temp_cavc/roi_stream.mp4")
    print(f"   • NROI Stream: temp_cavc/nroi_stream.mp4")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
