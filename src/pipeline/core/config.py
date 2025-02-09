"""
Configuration for the video analysis pipeline
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import os

@dataclass
class PipelineConfig:
    # YOLO configuration
    yolo_model: str = 'yolov8x.pt'
    detection_classes: Dict[str, List[int]] = None
    
    # CLIP configuration
    clip_model: str = "ViT-B/32"
    fuel_truck_prompts: List[str] = None
    plane_prompts: List[str] = None
    clip_threshold: float = 0.05  # Lowered from 0.3
    
    # SAM configuration
    sam_type: str = 'vit_b'
    proximity_threshold: float = 0.5
    
    # General settings
    input_size: int = 1024
    device: str = None
    
    def __post_init__(self):
        if self.detection_classes is None:
            self.detection_classes = {
                'vehicles': [2, 7],  # car, truck
                'planes': [4]  # airplane
            }
            
        if self.fuel_truck_prompts is None:
            self.fuel_truck_prompts = [
                "an airport fuel truck refueling an aircraft",
                "a fuel tanker truck at an airport",
                "an aviation fuel truck",
                "a regular delivery truck",
                "a cargo truck",
                "a maintenance vehicle",
                "a baggage truck",
                "a catering truck"
            ]
            
        if self.plane_prompts is None:
            self.plane_prompts = [
                # Positive prompts
                "a large commercial airliner with a distinctive tailfin",
                "a white airplane with orange accents and a pointed nose",
                "an easyjet aircraft with a bright orange logo",
                "a plane with a boarding bridge attached to the door",
                # Negative prompts
                "a fuel truck or tanker",
                "an airport bus",
                "airport ground equipment",
                "a baggage cart"
            ]
            
        if self.device is None:
            import torch
            self.device = ("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else 
                         "cpu")

DEFAULT_CONFIG = PipelineConfig()
