"""
YOLO-based object detector for initial frame filtering
"""

import numpy as np
from ultralytics import YOLO
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, config):
        """Initialize YOLO detector with configuration"""
        self.config = config
        logger.info(f"Loading YOLO model ({config.yolo_model})...")
        self.model = YOLO(config.yolo_model)
        self.model.to(config.device)
        self.class_ids = config.detection_classes
        
    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect objects in a frame
        Returns: Dict with 'vehicles' and 'planes' lists containing detections
        """
        results = self.model(frame)
        return self._filter_results(results[0])
    
    def _filter_results(self, results):
        """Filter YOLO results into vehicle and plane candidates"""
        detections = {
            'vehicles': [],
            'planes': []
        }
        
        for box in results.boxes:
            cls = box.cls[0].cpu().numpy()
            box_coords = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            
            logger.info(f"\nYOLO Detection:")
            logger.info(f"Class: {cls}, Confidence: {confidence:.3f}")
            logger.info(f"Box: {[int(x) for x in box_coords]}")
            
            if cls in self.class_ids['vehicles']:
                detections['vehicles'].append({
                    'box': box_coords,
                    'score': confidence,
                    'class': cls
                })
                logger.info("→ Added to vehicles")
            elif cls in self.class_ids['planes']:
                detections['planes'].append({
                    'box': box_coords,
                    'score': confidence
                })
                logger.info("→ Added to planes")
            else:
                logger.info("→ Ignored (not a vehicle or plane)")
                
        logger.info(f"\nYOLO Summary:")
        logger.info(f"- Found {len(detections['vehicles'])} vehicles")
        logger.info(f"- Found {len(detections['planes'])} planes")
        return detections
