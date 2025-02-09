# SAM2 Hello World Example

This is a simple example demonstrating how to use SAM2 (Segment Anything Model 2) for object segmentation.

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Download the SAM2 model checkpoint:
   - Visit the official SAM2 repository
   - Download the `sam2_h.pth` checkpoint
   - Place it in this directory

3. Prepare a sample image:
   - Place any image you want to segment as `sample_image.jpg` in this directory

## Running the Example

Simply run:
```bash
python sam2_hello_world.py
```

The script will:
1. Load your sample image
2. Initialize SAM2 with the downloaded checkpoint
3. Generate a segmentation mask for the center point of the image
4. Display the result with matplotlib

## Notes

- The script will use GPU if available, otherwise it will fall back to CPU
- The example uses a single point prompt in the center of the image
- You can modify the input points or add more prompts by changing the `input_point` array
