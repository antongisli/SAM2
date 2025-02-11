#!/usr/bin/env python3

import cv2
import logging
import argparse
from florence_analyzer import FlorenceAnalyzer, setup_logging

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze image for fuel trucks and planes using Florence-2')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--yolo-threshold', type=float, default=0.3, help='Minimum confidence for YOLO detections')
    parser.add_argument('--device', default='mps', choices=['cuda', 'mps', 'cpu'], help='Device to run models on')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.output_dir, level=logging.INFO)
    
    # Initialize analyzer
    analyzer = FlorenceAnalyzer(device=args.device, yolo_threshold=args.yolo_threshold)
    
    # Load image
    frame = cv2.imread(args.image_path)
    if frame is None:
        logging.error(f"Could not load image: {args.image_path}")
        return
    
    # Analyze frame
    logging.info("Analyzing frame...")
    detections, annotated_frame = analyzer.analyze_frame(frame)
    
    # Print detections
    for detection in detections:
        if detection["type"] == "fuel_truck":
            logging.info(f"Found fuel truck with Florence score: {detection['florence_score']:.3f}")
        else:
            logging.info(f"Found {detection['type']} with YOLO confidence: {detection['yolo_conf']:.3f}")
    
    # Save annotated image
    output_path = f"{args.output_dir}/annotated_image.jpg"
    cv2.imwrite(output_path, annotated_frame)
    logging.info(f"Saved annotated image to: {output_path}")

if __name__ == "__main__":
    main()
