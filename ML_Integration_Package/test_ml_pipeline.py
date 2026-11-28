# test_ml_pipeline.py
# =========================================================================
# EventLens Standalone ML Pipeline Verification Script (Windows/VS Code Compatible)
# =========================================================================

import os
import sys
import argparse
import json

try:
    import cv2
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"[!] Dependency Error: {e}")
    print("Please run: pip install -r requirements_ml.txt")
    sys.exit(1)

from eventlens_ml_engine import EventLensMLEngine

def main():
    parser = argparse.ArgumentParser(description="EventLens ML Engine Standalone Video Analysis Verification")
    parser.add_argument("--video", type=str, required=True, help="Path to the input AVI video file")
    parser.add_argument("--assets", type=str, default="./model_assets", help="Path to model assets directory")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold boundary")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[!] Error: Input video file not found at: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.assets):
        print(f"[!] Error: Model assets directory not found at: {args.assets}")
        sys.exit(1)

    print("=== Initializing EventLens ML Engine ===")
    try:
        engine = EventLensMLEngine(model_assets_dir=args.assets, confidence_threshold=args.threshold)
    except Exception as e:
        print(f"[!] Initialization failed: {e}")
        sys.exit(1)

    print(f"\n=== Executing Feature Extraction and MLP Inference for: {os.path.basename(args.video)} ===")
    result = engine.predict_video(args.video)

    print("\n=== PIPELINE INFERENCE RESULTS ===")
    print(json.dumps(result, indent=4))

    if result.get("status") == "success":
        print("\n[+] Verification PASSED! ML Inference executed and verified successfully.")
    else:
        print("\n[!] Verification FAILED. See error details in output.")

if __name__ == '__main__':
    main()
