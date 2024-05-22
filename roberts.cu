%%writefile roberts.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <stdio.h>

__device__ float
clamp(float val, float minVal, float maxVal)
{
    return fmaxf(minVal, fminf(maxVal, val));
}

__global__ void robertsEdgeDetection(unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int Gx[2][2]  = { { 1, 0 }, { 0, -1 } };
    int Gy[2][2] ={ { 0, 1 }, { -1, 0 } };

    float edgeX = 0.0;
    float edgeY = 0.0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        for (int ky = 0; ky <2; ky++)
        {
            for (int kx = 0; kx < 2; kx++)
            {
                float val = (float)input[(y + ky) * width + (x + kx)];
                edgeX += Gx[ky + 1][kx + 1] * val;
                edgeY += Gy[ky + 1][kx + 1] * val;
            }
        }
    }

    float edge = sqrt(edgeX * edgeX + edgeY * edgeY);
    edge = clamp(edge, 0, 255);
    output[y * width + x] = (unsigned char)edge;
}
int main()
{
    int width, height, channels;
    unsigned char *img = stbi_load("Image.jpg", &width, &height, &channels, 1); // Load image and convert to grayscale
    clock_t start, end;
    double cpu_time_used;

    if (img == NULL)
    {
        fprintf(stderr, "Error in loading the image\n");
        exit(1);
    }

    unsigned char *dst = (unsigned char *)malloc(width * height);
    if (dst == NULL)
    {
        fprintf(stderr, "Memory allocation failed for output image\n");
        stbi_image_free(img);
        exit(1);
    }

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);

    cudaMemcpy(d_input, img, width * height, cudaMemcpyHostToDevice);

    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    dim3 threadsPerBlock(16, 16);

    start = clock();

    robertsEdgeDetection<<<blocks, threadsPerBlock>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize(); // Wait for the GPU to finish

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Roberts filter on GPU took %f seconds to execute \n", cpu_time_used);

    cudaMemcpy(dst, d_output, width * height, cudaMemcpyDeviceToHost);

    stbi_write_png("output_roberts.png", width, height, 1, dst, width);

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(img);
    free(dst);

    return 0;
}

