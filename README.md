# CUDA-Image-Processing
CUDA-accelerated image processing implementation featuring box blur and horizontal flip operations. Uses parallel GPU computing for efficient image transformations. Supports grayscale images with configurable blur kernel size.

## Features
- Box blur filter with configurable kernel size
- Horizontal image flipping
- Parallel processing using CUDA
- Support for grayscale images
- Error handling for CUDA operations

## Requirements
- CUDA-capable GPU
- CUDA Toolkit
- C++ compiler
- Input images in raw format

## Usage
1. Compile the program:
```bash
nvcc Box_Blur_Flip.c -o image_processor
```

2. Run the program:
```bash
./image_processor
```

## Implementation Details

### Box Blur
- Uses a configurable kernel size for blur operation
- Averages pixel values within the kernel window
- Handles image boundaries appropriately
- Parallel processing at pixel level

### Horizontal Flip
- Flips image along vertical axis
- Maintains original image dimensions
- Efficient memory access pattern
- Thread-based parallel processing

## Input/Output
- Input: Raw grayscale images
- Output: Two processed images
  - `output_blur.jpg`: Blurred version
  - `output_flip.jpg`: Horizontally flipped version

## Contributors
- Dev Nasra (200905026)
- Harsha Chandra Mallubhotla (200905029)
- Surajit Dutta (200905332)
- Utkarsh Verma (190905324)
