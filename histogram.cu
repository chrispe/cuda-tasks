
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

/* Some histogram constants */
#define HISTOGRAM_SIZE	256		// The size of the histogram.
#define BAR_WIDTH 		4		// The bar width of each value.

/* 
	The kernel that applies the grays-scale transformation and then counts each color value
	in order to create the coressponding histogram.
*/
__global__
void _hist1D (const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * grayImage, unsigned int * histogram, const size_t pitch){ 
	/* Set the start index to process. */
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	/* Create the shared memory for each block. */
	// Each shared memory works as a different counter
	// at each block. At the end of the execution we
	// add the results to the global counter.
	__shared__ unsigned int shared_histogram[HISTOGRAM_SIZE];

	// Each thread initializes one of the array items to zero so that
	// all the items are then set to zero (256 threads - 256 items).
	const size_t thread_hist_index = blockDim.y * threadIdx.y + threadIdx.x;
	shared_histogram[thread_hist_index] = 0;

	// We then wait for all the threads of the block to synchronize. 
	__syncthreads();

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
	grayPix_float4.x = ((0.3f * r_float.x) + (0.59f * g_float.x) + (0.11f * b_float.x)) + 0.5f;
	grayPix_float4.y = ((0.3f * r_float.y) + (0.59f * g_float.y) + (0.11f * b_float.y)) + 0.5f;
	grayPix_float4.z = ((0.3f * r_float.z) + (0.59f * g_float.z) + (0.11f * b_float.z)) + 0.5f;
	grayPix_float4.w = ((0.3f * r_float.w) + (0.59f * g_float.w) + (0.11f * b_float.w)) + 0.5f;

	/* Store to the uchar4 */
	uchar4 grayPix_uchar4;
	grayPix_uchar4.x = (unsigned int)(grayPix_float4.x);
	grayPix_uchar4.y = (unsigned int)(grayPix_float4.y);
	grayPix_uchar4.z = (unsigned int)(grayPix_float4.z);
	grayPix_uchar4.w = (unsigned int)(grayPix_float4.w);

	/* Save the gray pixels to the output */
	((uchar4*)grayImage)[(y * rowImgWidth) + x] = grayPix_uchar4;

	/* Increase the counter of coressponding color for the 4-read pixels. */
	// We check the condition of the pixel not belonging to the image. 
	// But, instead of using branching, we just evaluate the expression
	// if it is true then it adds 1 else 0. 
	atomicAdd(&shared_histogram[grayPix_uchar4.x], 1);
	atomicAdd(&shared_histogram[grayPix_uchar4.y], x * 4 + 1 < width);
	atomicAdd(&shared_histogram[grayPix_uchar4.z], x * 4 + 2 < width);
	atomicAdd(&shared_histogram[grayPix_uchar4.w], x * 4 + 3 < width);

	/* 
		Synchronize all threads and then each thread adds the shared counter
		of a color to the coressponding global counter.
	*/
	__syncthreads();
	atomicAdd(&histogram[thread_hist_index], shared_histogram[thread_hist_index]);
}

/* 
	Second-level wrapper function for calling the kernel.
	It runs the kernel given the allocated space on the device and the specifications of the image. 
	The function first sets its architecture (or design) such as the number of threads, block size e.t.c.
*/
void histogram1D(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * grayImage, unsigned int * histogram, const size_t pitch){

	/* Define the number of active threads per block. */

	// In this case, we need to count 256 different values of color. 
	// Therefore a number of threads which is equal to 16 is very suitable
	// for our problem, since for a blocksize of 16x16, we have 256 threads
	// which is equal to the number of unique color values. 

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
  	_hist1D<<<gridSize, blockSize>>>(width, height, inputImage, grayImage, histogram, pitch);
  	cudaDeviceSynchronize();
}	