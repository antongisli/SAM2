"""
Main pipeline orchestrator
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging
import time

from ..stages.detection.yolo_detector import YOLODetector
from ..stages.detection.clip_verifier import CLIPVerifier
from ..stages.analysis.sam_analyzer import SAMAnalyzer
from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class VideoPipeline:
    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        """Initialize pipeline with configuration"""
        self.config = config
        self.frame_count = 0
        
        # Initialize pipeline stages
        logger.info("\n=== Initializing Pipeline Stages ===")
        self.detector = YOLODetector(config)
        self.verifier = CLIPVerifier(config)
        self.analyzer = SAMAnalyzer(config)
        
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Process a single frame through the pipeline
        Returns: (interactions, annotated_frame)
        """
        timing = {}
        
        # Preprocess frame
        t0 = time.time()
        h, w = frame.shape[:2]
        if max(h, w) > self.config.input_size:
            scale = self.config.input_size / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timing['preprocessing'] = time.time() - t0
        
        # Stage 1: YOLO Detection
        t0 = time.time()
        detections = self.detector.detect(rgb_frame)
        timing['yolo'] = time.time() - t0
        
        # Stage 2: CLIP Verification
        t0 = time.time()
        fuel_trucks = self.verifier.verify_fuel_trucks(rgb_frame, detections['vehicles'])
        timing['clip_trucks'] = time.time() - t0
        
        # Only proceed with plane verification if fuel trucks found
        verified_planes = []
        interactions = []
        
        if fuel_trucks:
            # Verify planes with CLIP
            t0 = time.time()
            verified_planes = self.verifier.verify_planes(rgb_frame, detections['planes'])
            timing['clip_planes'] = time.time() - t0
            
            # Stage 3: SAM Analysis (only if both objects present)
            if verified_planes:
                t0 = time.time()
                interactions = self.analyzer.analyze_proximity(
                    rgb_frame, fuel_trucks, verified_planes
                )
                timing['sam'] = time.time() - t0
        
        # Log timing information
        total_time = sum(timing.values())
        timing_str = "\n=== Frame Processing Times ===\n"
        for step, t in timing.items():
            pct = (t / total_time) * 100
            timing_str += f"{step}: {t:.3f}s ({pct:.1f}%)\n"
        timing_str += f"Total: {total_time:.3f}s"
        logger.info(timing_str)
        
        self.frame_count += 1
        return interactions, frame  # TODO: Add frame annotation
