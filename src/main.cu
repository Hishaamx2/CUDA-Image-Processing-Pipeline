#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

void cpuGaussianBlur(const unsigned char* input, unsigned char* output,
                     int width, int height, int channels) {
    float kernel[3][3] = {
        {1.0f / 16, 2.0f / 16, 1.0f / 16},
        {2.0f / 16, 4.0f / 16, 2.0f / 16},
        {1.0f / 16, 2.0f / 16, 1.0f / 16}
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                if (c == 3) {
                    output[(y * width + x) * channels + c] =
                        input[(y * width + x) * channels + c];
                    continue;
                }

                float sum = 0.0f;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int py = std::min(std::max(y + ky, 0), height - 1);
                        int idx = (py * width + px) * channels + c;
                        sum += input[idx] * kernel[ky + 1][kx + 1];
                    }
                }

                output[(y * width + x) * channels + c] =
                    static_cast<unsigned char>(sum);
            }
        }
    }
}

__global__ void gpuGaussianBlurKernel(const unsigned char* input,
                                      unsigned char* output,
                                      int width,
                                      int height,
                                      int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float kernel[3][3] = {
        {1.0f / 16, 2.0f / 16, 1.0f / 16},
        {2.0f / 16, 4.0f / 16, 2.0f / 16},
        {1.0f / 16, 2.0f / 16, 1.0f / 16}
    };

    for (int c = 0; c < channels; c++) {
        int outIdx = (y * width + x) * channels + c;

        if (c == 3) {
            output[outIdx] = input[outIdx];
            continue;
        }

        float sum = 0.0f;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                int idx = (py * width + px) * channels + c;

                sum += input[idx] * kernel[ky + 1][kx + 1];
            }
        }

        output[outIdx] = static_cast<unsigned char>(sum);
    }
}

void gpuGaussianBlur(const unsigned char* input, unsigned char* output,
                     int width, int height, int channels) {
    int imageSize = width * height * channels;

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    gpuGaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output,
                                                   width, height, channels);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void cpuSobel(const unsigned char* input, unsigned char* output,
              int width, int height, int channels) {
    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int outIdx = (y * width + x) * channels + c;

                if (c == 3) {
                    output[outIdx] = input[outIdx];
                    continue;
                }

                float sumX = 0.0f;
                float sumY = 0.0f;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int py = std::min(std::max(y + ky, 0), height - 1);
                        int idx = (py * width + px) * channels + c;

                        sumX += input[idx] * gx[ky + 1][kx + 1];
                        sumY += input[idx] * gy[ky + 1][kx + 1];
                    }
                }

                int magnitude = static_cast<int>(sqrtf(sumX * sumX + sumY * sumY));
                output[outIdx] = static_cast<unsigned char>(std::min(magnitude, 255));
            }
        }
    }
}

__global__ void gpuSobelKernel(const unsigned char* input,
                               unsigned char* output,
                               int width,
                               int height,
                               int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int c = 0; c < channels; c++) {
        int outIdx = (y * width + x) * channels + c;

        if (c == 3) {
            output[outIdx] = input[outIdx];
            continue;
        }

        float sumX = 0.0f;
        float sumY = 0.0f;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                int idx = (py * width + px) * channels + c;

                sumX += input[idx] * gx[ky + 1][kx + 1];
                sumY += input[idx] * gy[ky + 1][kx + 1];
            }
        }

        int magnitude = static_cast<int>(sqrtf(sumX * sumX + sumY * sumY));
        output[outIdx] = static_cast<unsigned char>(min(magnitude, 255));
    }
}

void gpuSobel(const unsigned char* input, unsigned char* output,
              int width, int height, int channels) {
    int imageSize = width * height * channels;

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    gpuSobelKernel<<<gridSize, blockSize>>>(d_input, d_output,
                                            width, height, channels);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int width, height, channels;

    unsigned char* img = stbi_load("images/test.png",
                                   &width,
                                   &height,
                                   &channels,
                                   0);

    if (!img) {
        std::cout << "Failed to load image\n";
        return 1;
    }

    std::cout << "Width: " << width << std::endl;
    std::cout << "Height: " << height << std::endl;
    std::cout << "Channels: " << channels << std::endl;

    int imageSize = width * height * channels;

    std::vector<unsigned char> cpuBlur(imageSize);
    std::vector<unsigned char> gpuBlur(imageSize);

    std::vector<unsigned char> cpuSobelOutput(imageSize);
    std::vector<unsigned char> gpuSobelOutput(imageSize);


    auto cpuStart = std::chrono::high_resolution_clock::now();

    cpuGaussianBlur(img, cpuBlur.data(), width, height, channels);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    auto gpuStart = std::chrono::high_resolution_clock::now();

    gpuGaussianBlur(img, gpuBlur.data(), width, height, channels);

    auto gpuEnd = std::chrono::high_resolution_clock::now();

    double gpuTime =
    std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();

    std::cout << "CPU Blur Time: " << cpuTime << " ms" << std::endl;
    std::cout << "GPU Blur Time: " << gpuTime << " ms" << std::endl;
    std::cout << "Speedup: " << cpuTime / gpuTime << "x" << std::endl;

    auto cpuSobelStart = std::chrono::high_resolution_clock::now();

    cpuSobel(cpuBlur.data(), cpuSobelOutput.data(), width, height, channels);

    auto cpuSobelEnd = std::chrono::high_resolution_clock::now();

    double cpuSobelTime =
        std::chrono::duration<double, std::milli>(cpuSobelEnd - cpuSobelStart).count();

    auto gpuSobelStart = std::chrono::high_resolution_clock::now();

    gpuSobel(gpuBlur.data(), gpuSobelOutput.data(), width, height, channels);

    auto gpuSobelEnd = std::chrono::high_resolution_clock::now();

    double gpuSobelTime =
        std::chrono::duration<double, std::milli>(gpuSobelEnd - gpuSobelStart).count();

    std::cout << "CPU Sobel Time: " << cpuSobelTime << " ms" << std::endl;
    std::cout << "GPU Sobel Time: " << gpuSobelTime << " ms" << std::endl;
    std::cout << "Sobel Speedup: " << cpuSobelTime / gpuSobelTime << "x" << std::endl;

    stbi_write_png("results/cpu_blur.png",
                   width, height, channels,
                   cpuBlur.data(),
                   width * channels);

    stbi_write_png("results/gpu_blur.png",
                   width, height, channels,
                   gpuBlur.data(),
                   width * channels);
    
    stbi_write_png("results/cpu_sobel.png",
                   width, height, channels,
                   cpuSobelOutput.data(),
                   width * channels);

    stbi_write_png("results/gpu_sobel.png",
                   width, height, channels,
                   gpuSobelOutput.data(),
                   width * channels);
   
    std::cout << "Saved CPU blur to results/cpu_blur.png" << std::endl;
    std::cout << "Saved GPU blur to results/gpu_blur.png" << std::endl;

    stbi_image_free(img);

    return 0;
}
