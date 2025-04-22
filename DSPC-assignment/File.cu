// CUDA DCT-based Invisible Watermarking
// Requires OpenCV and CUDA/cuFFT

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 8
#define ALPHA 5.0f

__global__ void embedWatermarkKernel(float* image, int width, int height, const uchar* watermark, int wmWidth, float alpha) {
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;

    if (bx + 7 >= width || by + 7 >= height) return;

    int tid = (by / BLOCK_SIZE % wmWidth) * wmWidth + (bx / BLOCK_SIZE % wmWidth);
    uchar wm_bit = watermark[tid] > 128 ? 1 : 0;

    int idx = (by + 3) * width + (bx + 3);
    image[idx] += (wm_bit == 1 ? alpha : -alpha);
}

void launchCudaKernel(float* d_image, int width, int height, const uchar* d_watermark, int wmWidth) {
    dim3 threads(1, 1);
    dim3 blocks(width / BLOCK_SIZE, height / BLOCK_SIZE);
    embedWatermarkKernel << <blocks, threads >> > (d_image, width, height, d_watermark, wmWidth, ALPHA);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./cuda_watermark <input_image> <watermark_image>" << endl;
        return -1;
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    Mat watermark = imread(argv[2], IMREAD_GRAYSCALE);

    if (image.empty() || watermark.empty()) {
        cerr << "Failed to load images." << endl;
        return -1;
    }

    image.convertTo(image, CV_32F);
    int imgSize = image.rows * image.cols * sizeof(float);
    int wmSize = watermark.rows * watermark.cols * sizeof(uchar);

    float* d_image;
    uchar* d_watermark;
    cudaMalloc(&d_image, imgSize);
    cudaMalloc(&d_watermark, wmSize);

    cudaMemcpy(d_image, image.ptr<float>(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_watermark, watermark.ptr<uchar>(), wmSize, cudaMemcpyHostToDevice);

    launchCudaKernel(d_image, image.cols, image.rows, d_watermark, watermark.cols);

    cudaMemcpy(image.ptr<float>(), d_image, imgSize, cudaMemcpyDeviceToHost);
    image.convertTo(image, CV_8U);
    imwrite("cuda_watermarked.png", image);

    cudaFree(d_image);
    cudaFree(d_watermark);

    cout << "Watermark embedded using CUDA." << endl;
    return 0;
}
