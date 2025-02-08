#include <cuda_runtime.h>
#include <stdio.h>

// Kernel to flip an image horizontally
__global__ void flip_horizontal_kernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;  // Calculate the x coordinate of the pixel
    int y = threadIdx.y + blockIdx.y * blockDim.y;  // Calculate the y coordinate of the pixel
    
    // Check if the pixel is within the image boundaries
    if (x >= width || y >= height) return;
    
    int index = y * width + x;  // Compute the 1D index of the pixel
    int flip_index = y * width + (width - x - 1);  // Compute the 1D index of the horizontally flipped pixel
    
    output[index] = input[flip_index];  // Assign the value of the flipped pixel to the output
}

// Kernel to apply a box blur to an image
__global__ void box_blur_kernel(unsigned char *input_image, unsigned char *output_image, int rows, int cols, int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Calculate the row of the pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate the column of the pixel
    int kernel_radius = kernel_size / 2;
    int kernel_area = kernel_size * kernel_size;
    int sum = 0;
    int count = 0;

    // Check if the pixel is within the image boundaries
    if (row < rows && col < cols) {
        // Iterate over the kernel area
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            for (int j = -kernel_radius; j <= kernel_radius; j++) {
                int curr_row = row + i;
                int curr_col = col + j;
                // Check if the neighboring pixel is within the image boundaries
                if (curr_row >= 0 && curr_row < rows && curr_col >= 0 && curr_col < cols) {
                    sum += input_image[curr_row * cols + curr_col];  // Accumulate the pixel values
                    count++;
                }
            }
        }
        output_image[row * cols + col] = sum / count;  // Compute the average and assign to the output
    }
}

// Function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Function to read an image from a file
void read_image(const char *filename, unsigned char *image, int width, int height) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    fread(image, sizeof(unsigned char), width * height, file);
    fclose(file);
}

// Function to save an image to a file
void save_image(const char *filename, unsigned char *image, int width, int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    fwrite(image, sizeof(unsigned char), width * height, file);
    fclose(file);
}

int main() {
    int width = 1024;
    int height = 768;
    int kernel_size = 5;

    size_t size = width * height * sizeof(unsigned char);

    // Allocate host memory
    unsigned char *h_input = (unsigned char *)malloc(size);
    unsigned char *h_output_flip = (unsigned char *)malloc(size);
    unsigned char *h_output_blur = (unsigned char *)malloc(size);

    // Initialize input data (example)
    read_image("input_image.jpg", h_input, width, height);

    // Allocate device memory
    unsigned char *d_input, *d_output_flip, *d_output_blur;
    checkCudaError(cudaMalloc(&d_input, size), "Failed to allocate device memory for input");
    checkCudaError(cudaMalloc(&d_output_flip, size), "Failed to allocate device memory for flip output");
    checkCudaError(cudaMalloc(&d_output_blur, size), "Failed to allocate device memory for blur output");

    // Copy input data to device
    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "Failed to copy input data to device");

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the horizontal flip kernel
    flip_horizontal_kernel<<<gridDim, blockDim>>>(d_input, d_output_flip, width, height);
    checkCudaError(cudaGetLastError(), "Failed to launch flip_horizontal_kernel");

    // Launch the box blur kernel
    box_blur_kernel<<<gridDim, blockDim>>>(d_output_flip, d_output_blur, height, width, kernel_size);
    checkCudaError(cudaGetLastError(), "Failed to launch box_blur_kernel");

    // Copy the output data back to host
    checkCudaError(cudaMemcpy(h_output_flip, d_output_flip, size, cudaMemcpyDeviceToHost), "Failed to copy flip output to host");
    checkCudaError(cudaMemcpy(h_output_blur, d_output_blur, size, cudaMemcpyDeviceToHost), "Failed to copy blur output to host");

    // Save the output images to files
    save_image("output_flip.jpg", h_output_flip, width, height);
    save_image("output_blur.jpg", h_output_blur, width, height);

    // Free allocated memory
    free(h_input);
    free(h_output_flip);
    free(h_output_blur);
    cudaFree(d_input);
    cudaFree(d_output_flip);
    cudaFree(d_output_blur);

    return 0;
}
