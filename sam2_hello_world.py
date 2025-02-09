import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon
import random
import argparse

def calculate_relationships(masks):
    relationships = []
    for i, mask1 in enumerate(masks):
        for j, mask2 in enumerate(masks):
            if i >= j:  # Skip self-relationships and duplicates
                continue
                
            try:
                # Convert masks to polygons
                contours1 = cv2.findContours((mask1 * 255).astype(np.uint8), 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)[0]
                contours2 = cv2.findContours((mask2 * 255).astype(np.uint8), 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)[0]
                
                # Skip if no valid contours
                if not len(contours1) or not len(contours2):
                    continue
                    
                # Get the largest contour for each mask
                largest_contour1 = max(contours1, key=cv2.contourArea)
                largest_contour2 = max(contours2, key=cv2.contourArea)
                
                # Skip if contours are too small
                if len(largest_contour1) < 4 or len(largest_contour2) < 4:
                    continue
                
                # Convert to polygons
                poly1 = Polygon(largest_contour1.squeeze().reshape(-1, 2))
                poly2 = Polygon(largest_contour2.squeeze().reshape(-1, 2))
                
                if not poly1.is_valid or not poly2.is_valid:
                    continue
                
                # Calculate relationships
                relationship = {
                    "object1": i,
                    "object2": j,
                    "intersects": poly1.intersects(poly2),
                    "contains": poly1.contains(poly2) or poly2.contains(poly1),
                    "distance": poly1.distance(poly2),
                    "area_ratio": poly1.area / poly2.area if poly2.area != 0 else 0
                }
                relationships.append(relationship)
            except Exception as e:
                print(f"Warning: Skipping relationship between objects {i} and {j} due to error: {str(e)}")
                continue
    
    return relationships

def generate_grid_points(image, grid_size=3):
    height, width = image.shape[:2]
    points = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = int((j + 0.5) * width / grid_size)
            y = int((i + 0.5) * height / grid_size)
            points.append([x, y])
            
    # Add some random points
    for _ in range(5):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        points.append([x, y])
        
    return np.array(points)

def print_human_readable_analysis(relationships, scores):
    print("\n=== Object Detection and Relationship Analysis ===")
    print(f"\nFound {len(scores)} objects in the image")
    print("\nObject Confidence Scores:")
    for i, score in enumerate(scores):
        print(f"Object {i}: confidence {score:.3f}")
    
    print("\nObject Relationships:")
    for rel in relationships:
        obj1, obj2 = rel["object1"], rel["object2"]
        print(f"\nObject {obj1} and Object {obj2}:")
        
        # Distance analysis
        distance = rel["distance"]
        if distance < 1:
            proximity = "are touching or very close"
        elif distance < 50:
            proximity = "are close to each other"
        else:
            proximity = "are far apart"
        print(f"- {proximity} (distance: {distance:.1f} pixels)")
        
        # Intersection/Containment
        if rel["contains"]:
            print("- One object contains the other")
        elif rel["intersects"]:
            print("- Objects overlap")
        else:
            print("- Objects are separate")
        
        # Size relationship
        area_ratio = rel["area_ratio"]
        if abs(1 - area_ratio) < 0.2:
            size_rel = "similar in size"
        elif area_ratio > 1:
            size_rel = f"first is {area_ratio:.1f}x larger"
        else:
            size_rel = f"second is {(1/area_ratio):.1f}x larger"
        print(f"- Objects are {size_rel}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SAM2 Object Segmentation Demo')
    parser.add_argument('--image', type=str, default='sample_image.jpg',
                      help='Path to input image (default: sample_image.jpg)')
    parser.add_argument('--checkpoint', type=str, default='sam2_h.pth',
                      help='Path to SAM2 checkpoint file (default: sam2_h.pth)')
    parser.add_argument('--model-type', type=str, default='vit_h',
                      choices=['vit_h', 'vit_l', 'vit_b'],
                      help='SAM model type (default: vit_h)')
    parser.add_argument('--grid-size', type=int, default=3,
                      help='Size of the grid for point sampling (default: 3)')
    parser.add_argument('--random-points', type=int, default=5,
                      help='Number of additional random points (default: 5)')
    parser.add_argument('--output-json', type=str, default='segmentation_results.json',
                      help='Path to output JSON file (default: segmentation_results.json)')
    parser.add_argument('--output-viz', type=str, default='segmentation_visualization.png',
                      help='Path to output visualization file (default: segmentation_visualization.png)')
    
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image '{args.image}'")
        return
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Initializing SAM model...")
    
    # Initialize SAM2
    try:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam.to(device=device)
        print(f"Using device: {device}")
    except FileNotFoundError:
        print(f"Error: Could not find SAM2 checkpoint file '{args.checkpoint}'")
        return
    except Exception as e:
        print(f"Error initializing SAM2 model: {str(e)}")
        return
    
    # Create predictor
    predictor = SamPredictor(sam)
    
    # Set image
    predictor.set_image(img)
    
    # Generate grid points with custom size
    input_points = generate_grid_points(img, grid_size=args.grid_size)
    input_labels = np.ones(len(input_points))  # All points are foreground
    
    print(f"\nAnalyzing image using {len(input_points)} points...")
    
    # Get masks for all points
    all_masks = []
    all_scores = []
    
    for point, label in zip(input_points, input_labels):
        masks, scores, _ = predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([label]),
            multimask_output=True
        )
        # Take the mask with highest score
        best_mask_idx = np.argmax(scores)
        all_masks.append(masks[best_mask_idx])
        all_scores.append(scores[best_mask_idx])
    
    # Calculate relationships between masks
    relationships = calculate_relationships(all_masks)
    
    # Create output JSON
    output = {
        "num_objects": len(all_masks),
        "object_scores": [float(score) for score in all_scores],
        "relationships": relationships,
        "points_used": input_points.tolist()
    }
    
    # Save JSON output
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print human-readable analysis
    print_human_readable_analysis(relationships, all_scores)
    
    # Visualize the results
    plt.figure(figsize=(20, 10))
    
    # Create two subplots
    plt.subplot(1, 2, 1)
    plt.title("All Segments")
    plt.imshow(img)
    
    # Show all masks with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_masks)))
    for i, (mask, color) in enumerate(zip(all_masks, colors)):
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask] = (*color[:3], 0.3)  # Reduce alpha for better visibility
        plt.imshow(colored_mask)
        
        # Add object number to the center of the mask
        y, x = np.where(mask)
        if len(x) > 0 and len(y) > 0:
            center_x = int(x.mean())
            center_y = int(y.mean())
            plt.annotate(f"Object {i}", (center_x, center_y), color='white', 
                        bbox=dict(facecolor='black', alpha=0.7),
                        ha='center', va='center')
    
    # Create a second subplot showing only the main truck body and container
    plt.subplot(1, 2, 2)
    plt.title("Main Components")
    plt.imshow(img)
    
    # Show only the largest connected components
    main_objects = [0, 1, 2, 3, 4, 10, 11, 12]  # Main truck body
    container_objects = [5, 6]  # Likely container parts
    
    # Show main truck body in blue
    combined_truck = np.zeros_like(all_masks[0])
    for i in main_objects:
        combined_truck = combined_truck | all_masks[i]
    colored_mask = np.zeros((*combined_truck.shape, 4))
    colored_mask[combined_truck] = (0, 0, 1, 0.3)  # Blue for truck
    plt.imshow(colored_mask)
    
    # Show container in red
    for i in container_objects:
        colored_mask = np.zeros((*all_masks[i].shape, 4))
        colored_mask[all_masks[i]] = (1, 0, 0, 0.3)  # Red for container
        plt.imshow(colored_mask)
    
    plt.tight_layout()

    # Show input points
    plt.plot(input_points[:, 0], input_points[:, 1], 'rx')
    plt.title(f"SAM2 Segmentation Result - {len(all_masks)} objects detected")
    plt.axis('off')
    plt.savefig(args.output_viz)
    plt.close()
    
    print(f"\nResults saved to '{args.output_json}'")
    print(f"Visualization saved to '{args.output_viz}'")

if __name__ == "__main__":
    main()
