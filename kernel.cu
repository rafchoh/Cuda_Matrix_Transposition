#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define MAT_SIZE 1024
#define BLOCK_COLS 64
#define BLOCK_ROWS 8
#define PADDING_VALUE 0

__global__ void transposeMatrix(int* output_data, const int* input_data, int width, int padding) {
    __shared__ int tile[BLOCK_COLS][BLOCK_COLS];

    int x = blockIdx.x * BLOCK_COLS + threadIdx.x;
    int y = blockIdx.y * BLOCK_COLS + threadIdx.y;
    int index_in = y * width + x;

#pragma unroll
    for (int j = 0; j < BLOCK_COLS; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = (x < width && (y + j) < width) ? input_data[index_in + j * width] : padding;
    }

    __syncthreads();

    x = blockIdx.y * BLOCK_COLS + threadIdx.x;
    y = blockIdx.x * BLOCK_COLS + threadIdx.y;
    int index_out = y * width + x;

#pragma unroll
    for (int j = 0; j < BLOCK_COLS; j += BLOCK_ROWS) {
        output_data[index_out + j * width] = tile[threadIdx.x][threadIdx.y + j];
    }
}

void printMatrix(const int* matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    const dim3 blockSize(BLOCK_COLS, BLOCK_ROWS);
    const dim3 gridSize(MAT_SIZE / BLOCK_COLS, MAT_SIZE / BLOCK_COLS);

    int* h_inputMatrix = new int[MAT_SIZE * MAT_SIZE];
    int* h_transMatrix = new int[MAT_SIZE * MAT_SIZE];

    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; ++i) {
        h_inputMatrix[i] = rand() % 101; 
    }

    int* d_inputMatrix, * d_transMatrix;
    cudaMalloc((void**)&d_inputMatrix, MAT_SIZE * MAT_SIZE * sizeof(int));
    cudaMalloc((void**)&d_transMatrix, MAT_SIZE * MAT_SIZE * sizeof(int));

    cudaMemcpy(d_inputMatrix, h_inputMatrix, MAT_SIZE * MAT_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    transposeMatrix << <gridSize, blockSize >> > (d_transMatrix, d_inputMatrix, MAT_SIZE, PADDING_VALUE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_transMatrix, d_transMatrix, MAT_SIZE * MAT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Input Matrix:" << std::endl;
    //printMatrix(h_inputMatrix, MAT_SIZE, MAT_SIZE);

    std::cout << "\n \n";

    std::cout << "\nTransposed Matrix:" << std::endl;
   // printMatrix(h_transMatrix, MAT_SIZE, MAT_SIZE);

    std::cout << "\nExecution Time is: " << milliseconds << " ms" << std::endl;

    delete[] h_inputMatrix;
    delete[] h_transMatrix;
    cudaDeviceReset();
    cudaFree(d_inputMatrix);
    cudaFree(d_transMatrix);

    return 0;
}
