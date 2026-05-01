# Project Proposal

## Project Idea / Overview
I will build a GPU-based image processing pipeline that applies filters such as Sobel edge detection and Gaussian blur. I will implement both CPU and CUDA GPU versions and compare their performance to show how GPU parallelism speeds up image processing.

## Required Libraries / Framework
- CUDA
- C/C++
- stb_image or similar image library

## Potential Risks / Challenges
- Learning CUDA kernel implementation
- Managing memory between CPU and GPU
- Ensuring GPU output matches CPU output
- Handling image input/output correctly

## Plan / Outline
1. Implement CPU version of image filters
2. Implement GPU version using CUDA
3. Compare results for correctness
4. Measure performance differences
5. Test on different image sizes
6. Prepare report and presentation
