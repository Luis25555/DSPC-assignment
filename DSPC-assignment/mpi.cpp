// MPI DCT-based Invisible Watermarking
// Requires OpenCV and MPI

#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
        
#define BLOCK_SIZE 8
#define ALPHA 5.0f

void embedWatermark(Mat& chunk, Mat& watermark, int offsetY) {
    int wmWidth = watermark.cols;

    for (int y = 0; y < chunk.rows; y += BLOCK_SIZE) {
        for (int x = 0; x < chunk.cols; x += BLOCK_SIZE) {
            if (y + 7 >= chunk.rows || x + 7 >= chunk.cols) continue;

            Rect blockRegion(x, y, BLOCK_SIZE, BLOCK_SIZE);
            Mat block = chunk(blockRegion);
            Mat dctBlock;
            dct(block, dctBlock);

            int wmX = ((offsetY + y) / BLOCK_SIZE) % wmWidth;
            int wmY = (x / BLOCK_SIZE) % wmWidth;
            uchar wmBit = watermark.at<uchar>(wmX, wmY) > 128 ? 1 : 0;

            dctBlock.at<float>(3, 3) += (wmBit == 1) ? ALPHA : -ALPHA;
            idct(dctBlock, block);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) cout << "Usage: mpirun -np <n> ./mpi_watermark <input_image> <watermark_image>" << endl;
        MPI_Finalize();
        return 0;
    }

    Mat image, watermark;
    if (rank == 0) {
        image = imread(argv[1], IMREAD_GRAYSCALE);
        watermark = imread(argv[2], IMREAD_GRAYSCALE);

        if (image.empty() || watermark.empty()) {
            cerr << "Failed to load images." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        image.convertTo(image, CV_32F);
    }

    int rows, cols;
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunkRows = rows / size;
    Mat localChunk(chunkRows, cols, CV_32F);

    MPI_Scatter(image.ptr<float>(), chunkRows * cols, MPI_FLOAT,
        localChunk.ptr<float>(), chunkRows * cols, MPI_FLOAT,
        0, MPI_COMM_WORLD);

    if (rank != 0) watermark = imread(argv[2], IMREAD_GRAYSCALE);
    if (watermark.empty()) {
        cerr << "Failed to load watermark on rank " << rank << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    embedWatermark(localChunk, watermark, rank * chunkRows);

    Mat result;
    if (rank == 0) result.create(rows, cols, CV_32F);

    MPI_Gather(localChunk.ptr<float>(), chunkRows * cols, MPI_FLOAT,
        result.ptr<float>(), chunkRows * cols, MPI_FLOAT,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        result.convertTo(result, CV_8U);
        imwrite("mpi_watermarked.png", result);
        cout << "Watermark embedded using MPI." << endl;
    }

    MPI_Finalize();
    return 0;
}
