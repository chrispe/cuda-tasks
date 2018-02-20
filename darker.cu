#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

/* 
	The kernel that applies the gray-dark transformation.
*/
__global__
void _darkGray(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * darkGrayImage, const size_t pitch) {
	/* Set the start index to process. */
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	/* Check that x and y are within the bounds of the image. */
	// We multiply by 4 since we read 4 pixels at a time. 
	if (4*x >= width || y >= height) 
		return;

	/* Perform the coalesced access to the array */
	// According to CUDA's documentation, the address of an element in a pitch array
	// is typically given by: (T*)((char*)BaseAddress + Row * pitch) + Column;
	// In our case, the base address is the inputImage, while Row=y and Column=X.
	// We have actually replaced the original row*width by row*pitch, since the  
	// new width of the array is equal to pitch. We use global memory coalescing. 
	// rowImgWidth is the total width of the image represented in a single row of width 'pitch'
	// since we read 4 pixels at a time.
	const unsigned int rowImgWidth = pitch/4;

	/* Load 4 unsigned char elements of each red/green/blue */
	uchar4 r = ((uchar4*)inputImage)[(y * rowImgWidth) + x];
	uchar4 g = ((uchar4*)inputImage)[(rowImgWidth * height) + (y * rowImgWidth) + x];
	uchar4 b = ((uchar4*)inputImage)[(2 * rowImgWidth * height) + (y * rowImgWidth) + x];

	/* Convert them to float4, to make sure that we get a higher precision */
	float4 r_float = {(float)(r.x), (float)(r.y), (float)(r.z), (float)(r.w)};
	float4 g_float = {(float)(g.x), (float)(g.y), (float)(g.z), (float)(g.w)};
	float4 b_float = {(float)(b.x), (float)(b.y), (float)(b.z), (float)(b.w)};

	/* Calculate grey values for 4 pixels */
	float4 grayPix_float4 = {0.0f,0.0f,0.0f,0.0f};
	grayPix_float4.x = 0.5f + ((0.3f * r_float.x) + (0.59f * g_float.x) + (0.11f * b_float.x))* 0.6f;
	grayPix_float4.y = 0.5f + ((0.3f * r_float.y) + (0.59f * g_float.y) + (0.11f * b_float.y))* 0.6f;
	grayPix_float4.z = 0.5f + ((0.3f * r_float.z) + (0.59f * g_float.z) + (0.11f * b_float.z))* 0.6f;
	grayPix_float4.w = 0.5f + ((0.3f * r_float.w) + (0.59f * g_float.w) + (0.11f * b_float.w))* 0.6f;

	/* Store to the uchar4 */
	uchar4 grayPix_uchar4;
	grayPix_uchar4.x = (unsigned int)(grayPix_float4.x);
	grayPix_uchar4.y = (unsigned int)(grayPix_float4.y);
	grayPix_uchar4.z = (unsigned int)(grayPix_float4.z);
	grayPix_uchar4.w = (unsigned int)(grayPix_float4.w);

	/* Save the gray pixels to the output */
	((uchar4*)darkGrayImage)[(y * rowImgWidth) + x] = grayPix_uchar4;
}

/* 
	Second-level wrapper function for calling the kernel.
	It runs the kernel given the allocated space on the device and the specifications of the image. 
	The function first sets its architecture (or design) such as the number of threads, block size e.t.c.
*/
void darkGray(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * darkGrayImage, const size_t pitch){

	/* Define the number of active threads per block. */

	// The GTX 480 contains a total of 448 cores. Each SM is consisted of 32 cores.
	// Therefore, we have a total of 14 SMs. Each SM can handle 48 resident warps. 
	// Each warp is consisted of 32 threads. Therefore each SM can handle 1536 resident threads.
	// The total number of resident threads that the GPU can handle is 14*1536 = 21.504 threads. 
	// The maximum number of active threads has to be a multiple of 32.

	// Choosing a 32x32 block size gives a total of 1024 threads per SM. This is the maximum
	// number of threads in a block. By testing it we get a lower speedup than by using a block
	// size of 16x16 instead. Having a 32x32 block size introduces greater delays to start 
	// processing the next block. Also, by using the CUDA occupancy calculator, we get that for
	// a block size of 256 (16x16), we get a maximum warp occupancy. In case that the problem
	// is bottlenecked by computation and not memory accesses, increasing occupancy has no effect though.
	// But, in our problem, the dimensions of the image are important. Finally, since our problem 
	// is focused on image processing we prefer to use a 2D block rather than a 1D or 3D, since 
	// it helps in the indexing. 

  	const unsigned int threads = 16;
  	const dim3 blockSize(threads, threads);

  	/* Define the number of grids to use. */

  	// The grid size depends on the size of the problem, a.k.a the dimensions of the image.
  	// Therefore we need to split the problem into k domains, where k is the number of grids
  	// needed in order to fit the problem by dividing by the block size at each dimension.
  	// However, since we are going to use a more memory-coalesing friendly technique, in 
  	// which the thread reads 4 elements at the X-domain and therefore, the X-domain 
  	// (meaning the width) is divided also by 4. We want to use the upper part of the division
  	// (since we do not want to compute less elements) and for that case we use ceil instead of 
  	// the floor function.

  	const dim3 gridSize(ceil(((1.0*width)/4)/blockSize.x), ceil((1.0*height)/blockSize.y));

  	/* Start the kernel given the parameters and the defined architecture. */
  	_darkGray<<<gridSize, blockSize>>>(width, height, inputImage, darkGrayImage, pitch);
  	cudaDeviceSynchronize();
}	