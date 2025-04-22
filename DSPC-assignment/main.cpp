// OpenMP DCT-based Invisible Watermarking
// Requires OpenCV and OpenMP

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 8
#define ALPHA 5.0f

void embedWatermark(Mat& image, Mat& watermark) {
    int wmWidth = watermark.cols;
    int imgRows = image.rows;
    int imgCols = image.cols;

#pragma omp parallel for collapse(2)
    for (int y = 0; y < imgRows; y += BLOCK_SIZE) {
        for (int x = 0; x < imgCols; x += BLOCK_SIZE) {
            if (y + 7 >= imgRows || x + 7 >= imgCols) continue;

            Rect blockRegion(x, y, BLOCK_SIZE, BLOCK_SIZE);
            Mat block = image(blockRegion);
            Mat dctBlock;
            dct(block, dctBlock);

            int wmX = (y / BLOCK_SIZE) % wmWidth;
            int wmY = (x / BLOCK_SIZE) % wmWidth;
            uchar wmBit = watermark.at<uchar>(wmX, wmY) > 128 ? 1 : 0;

            dctBlock.at<float>(3, 3) += (wmBit == 1 ? ALPHA : -ALPHA);
            idct(dctBlock, block);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./omp_watermark <input_image> <watermark_image>" << endl;
        return -1;
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    Mat watermark = imread(argv[2], IMREAD_GRAYSCALE);

    if (image.empty() || watermark.empty()) {
        cerr << "Failed to load images." << endl;
        return -1;
    }

    image.convertTo(image, CV_32F);

    double start = omp_get_wtime();
    embedWatermark(image, watermark);
    double end = omp_get_wtime();
    double runtime = end - start;

    cout << "OpenMP Watermark Embedding Time: " << runtime << " seconds" << endl;

    image.convertTo(image, CV_8U);
    imwrite("omp_watermarked.png", image);

    return 0;
}

