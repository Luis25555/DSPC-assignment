#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

const int BLOCK_SIZE = 8;   
const double ALPHA = 5.0; // Embedding strength

// Embed watermark into the DCT domain  
void embedWatermark(Mat& image, const Mat& watermark) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    int rows = gray.rows - gray.rows % BLOCK_SIZE;  
    int cols = gray.cols - gray.cols % BLOCK_SIZE;  

#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            Mat block = gray(Rect(j, i, BLOCK_SIZE, BLOCK_SIZE));
            Mat dctBlock;
            dct(block, dctBlock);

            // Embed watermark bit
            int wm_x = i / BLOCK_SIZE % watermark.rows;
            int wm_y = j / BLOCK_SIZE % watermark.cols;
            uchar wm_bit = watermark.at<uchar>(wm_x, wm_y) > 128 ? 1 : 0;

            dctBlock.at<float>(3, 3) += (wm_bit == 1 ? ALPHA : -ALPHA);
            idct(dctBlock, block); // Modify original block with embedded watermark
        }
    }

    gray.convertTo(image, CV_8U);
    cvtColor(image, image, COLOR_GRAY2BGR);
}

void extractWatermark(const Mat& image, Mat& extracted, Size wmSize) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    int rows = gray.rows - gray.rows % BLOCK_SIZE;
    int cols = gray.cols - gray.cols % BLOCK_SIZE;

    extracted = Mat::zeros(wmSize, CV_8UC1);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            Mat block = gray(Rect(j, i, BLOCK_SIZE, BLOCK_SIZE));
            Mat dctBlock;
            dct(block, dctBlock);
            
            float coef = dctBlock.at<float>(3, 3);
            int wm_x = i / BLOCK_SIZE % wmSize.height;
            int wm_y = j / BLOCK_SIZE % wmSize.width;

            extracted.at<uchar>(wm_x, wm_y) += (coef > 0 ? 255 : 0);
        }
    }

    // Normalize watermark
    threshold(extracted, extracted, 127, 255, THRESH_BINARY);
}   

int main() {
    Mat image = imread("input.jpg");
    Mat watermark = imread("watermark.png", IMREAD_GRAYSCALE);
    resize(watermark, watermark, Size(32, 32));

    embedWatermark(image, watermark);
    imwrite("watermarked.jpg", image);

    Mat extracted;
    extractWatermark(image, extracted, watermark.size());
    imwrite("extracted.png", extracted);

    cout << "Watermark embedded and extracted successfully!" << endl;
    return 0;
}