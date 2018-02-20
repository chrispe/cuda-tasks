#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

/* The cuda error checking function imported */
extern void cudaError(cudaError_t e, const char * msg);

/* The total number of spectrums in the image. */
#define SPECTRUMS 3

/* 
	We need to also save 2 lines of pixels
	in the horizontal and vertical domain of
	the image, so that the pixels that belong
	to the borders of the domain that each block
	handles, can also access the two lines of pixels
	that belong to another block. Therefore, we have
	the standard 16x16 but since we need to also include
	those neighbours we have a total of 20x20.
*/
#define SHARED_SLICE 20
#define SHARED_MAIN_SIZE SPECTRUMS*SHARED_SLICE*SHARED_SLICE

/* 
	The filter applied in the triangular smooth kernel. 
	It is used as a constant since all the threads just
	read from that array during the execution of the kernel.
	Also since we declare it as a constant, less cache misses
	will be happening and therefore less accesses to the global
	memory.
*/
#define FILTER_X 5
#define FILTER_Y 5
#define FILTER_LENGTH FILTER_X*FILTER_Y
__constant__ float filter[FILTER_Y][FILTER_X];
 
/* Returns the pixel belonging to the (x+x0,y+y0,z) position of the shared memory. */
__device__ unsigned char 
sharedImage_getPixel(const unsigned int width, const unsigned int height, const unsigned char * inputImageShared, 
					 const unsigned int x, const unsigned int y, const unsigned int z, const int x0, const int y0)
{
	unsigned int idx = (z * SHARED_SLICE * SHARED_SLICE)+ (threadIdx.y+y0)*SHARED_SLICE + (threadIdx.x+x0);
	return inputImageShared[idx];
}

/* Sets a pixel on the shared memory, given the required info. The set is executed only when the variable valid is equal to 1. */
__device__ void
sharedImage_setPixel(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * inputImageShared, 
					 const unsigned int x, const unsigned int y, const unsigned int z, const int x0, const int y0, const unsigned char valid)
{
	if(valid){
		unsigned int idx = (z * SHARED_SLICE * SHARED_SLICE) + (threadIdx.y+y0+2)*SHARED_SLICE + (threadIdx.x+x0+2);
		inputImageShared[idx] = inputImage[(z * width * height) + ((y0+y) * width) + (x0+x)];
	}
}

/* 
	Sets the pixel of each thread in the shared memory.
	In case the pixel is near the borders then we also save
	its neighbours so that we can get them later during the execution
	of the kernel from the shared memory instead of the global memory.
*/
__device__ void
sharedImage_setPixels(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * inputImageShared, 
					  const unsigned int x, const unsigned int y)
{
	/* 
		Check that x and y are within the bounds of the image.
	*/
	if (x >= width || y >= height) 
		return;

	/* 
		For each spectrum set the pixel of the thread 
		plus any possible border neighbours... 
	*/
	for(unsigned int z=0; z<SPECTRUMS; z++){									 // x-axis step  y-axis step	perform steps under case
		sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		0, 				0, 					  1);						 // original case
		sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		-2, 			0, 		threadIdx.x < 2);						 // left border
		sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		0, 				-2, 	threadIdx.y < 2);						 // top border
		sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		2, 				0, 		threadIdx.x >= 14);						 // right border
 		sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		0, 				2, 		threadIdx.y >= 14);						 // bottom border
 		sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		-2, 			-2, 	threadIdx.x < 2 && threadIdx.y < 2);	 // top-left border
	 	sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		2, 				-2, 	threadIdx.x >= 14 && threadIdx.y < 2);	 // top-right border
	 	sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		-2, 			2, 		threadIdx.x < 2 && threadIdx.y >= 14);	 // bottom-left border
	 	sharedImage_setPixel(width, height, inputImage, inputImageShared, x, y, z, 		2, 				2, 		threadIdx.x >= 14 && threadIdx.y >= 14); // bottom-right border
	}
}

/* 
	The kernel that applies the triangular smooth filter for the 
	area of the image that is inside the borders ([2,width-2] [2,height-2]). 
*/
__global__
void _triangularSmooth(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * smoothImage){
	/* 
		Set the start index to process. 
		Always +2, so that we do not hit any pixel on the borders. 
	*/
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + 2;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + 2;

	/* 
		Create the shared memory used to read pixel values.
		This is useful since multiple reads are done for the 
		same pixels since every pixel also reads its neighbours.
	*/
	__shared__ unsigned char inputImageShared[SHARED_MAIN_SIZE];	
	sharedImage_setPixels(width,height,inputImage,inputImageShared,x,y);
	__syncthreads();

	/* 
		Check that x and y are within the bounds of the image
	   	(in this case inside the borders too). 
	*/
	if (x >= width - 2 || y >= height - 2) 
		return;

	/* Run the covolution filter for each spectrum. */
	for (unsigned int z = 0; z<SPECTRUMS; z++){
		/* Initialize the result variables */
		float filterSum = 0.0f;
		float smoothPix = 0.0f;	

		/* Run all over the neighbour pixels */
		for (unsigned int y0 = 0; y0 <= 4; y0++) {
			for (unsigned int x0 = 0; x0 <= 4; x0++) {
				unsigned char pixel = sharedImage_getPixel(width, height, inputImageShared, x, y, z, x0, y0);
				smoothPix += static_cast< float >(pixel) * filter[y0][x0];
				filterSum += filter[y0][x0];
			}
		}

		/* Save the computed value. */
		smoothPix /= filterSum;
		smoothImage[(z * width * height) + (y * width) + x] = static_cast< unsigned char >(smoothPix+0.5f); 
	}
}

/* 
	The kernel that applies the triangular smooth filter and can be used 
	for either the left or top border of the image. 
*/
__global__
void _triangularSmoothTopLeft(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * smoothImage){
	/* Set the start index to process. */
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	/* Check that x and y are within the bounds of the image. */
	if (x >= width || y >= height) 
		return;

	/* 
		Set the limits of pixels to use. This is done 
		so that we do not use an index which is out of bounds.
	*/
	const unsigned int xstart = -1*(x>0)-1*(x>1);
	const unsigned int ystart = -1*(y>0)-1*(y>1);
	const unsigned int xsteps = (x<(width-1))+(x<(width-2))+(x>0)+(x>1);
	const unsigned int ysteps = (y<(height-1))+(y<(height-2))+(height>0)+(height>1);

	/* Run the covolution for each spectrum. */
	for (unsigned int z = 0; z<SPECTRUMS; z++){
		/* Initialize the result variables */
		float filterSum = 0.0f;
		float smoothPix = 0.0f;	

		/* Run all over the neighbour pixels */
		for (unsigned int y0 = 0; y0 <= ysteps; y0++) {
			for (unsigned int x0 = 0; x0 <= xsteps; x0++) {
				smoothPix += static_cast< float >(inputImage[(z * width * height) + ((y0+ystart+y) * width) + (x0+xstart+x)]) * filter[y0][x0];
				filterSum += filter[y0][x0];
			}
		}

		/* Save the computed value. */
		smoothPix /= filterSum;
		smoothImage[(z * width * height) + (y * width) + x] = static_cast< unsigned char >(smoothPix+0.5f); 
	}
}

/* 
	The kernel that applies the triangular smooth filter 
	for the right border.
*/
__global__
void _triangularSmoothRight(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * smoothImage){
	/* 
		Set the start index to process. 
		We need to start from the right, that is why we set x to the following.
	*/
	const unsigned int x = width - blockDim.x * blockIdx.x - threadIdx.x - 1;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	/* Check that x and y are within the bounds of the image. */
	if (x >= width || y >= height) 
		return;

	/* 
		Set the limits of pixels to use. This is done 
		so that we do not use an index which is out of bounds.
	*/
	const unsigned int xstart = -1*(x>0)-1*(x>1);
	const unsigned int ystart = -1*(y>0)-1*(y>1);
	const unsigned int xsteps = (x<(width-1))+(x<(width-2))+(x>0)+(x>1);
	const unsigned int ysteps = (y<(height-1))+(y<(height-2))+(height>0)+(height>1);

	/* Run the covolution for each spectrum. */
	for (unsigned int z = 0; z<SPECTRUMS; z++){
		/* Initialize the result variables */
		float filterSum = 0.0f;
		float smoothPix = 0.0f;	

		/* Run all over the neighbour pixels */
		for (unsigned int y0 = 0; y0 <= ysteps; y0++) {
			for (unsigned int x0 = 0; x0 <= xsteps; x0++) {
				smoothPix += static_cast< float >(inputImage[(z * width * height) + ((y0+ystart+y) * width) + (x0+xstart+x)]) * filter[y0][x0];
				filterSum += filter[y0][x0];
			}
		}

		/* Save the computed value. */
		smoothPix /= filterSum;
		smoothImage[(z * width * height) + (y * width) + x] = static_cast< unsigned char >(smoothPix+0.5f); 
	}
}

/* 
	The kernel that applies the triangular smooth filter 
	for the bottom border.
*/
__global__
void _triangularSmoothBottom(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * smoothImage){
	/* 
		Set the start index to process. 
		We need to start from the bottom, that is why we set y to the following.
	*/
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = height - blockDim.y * blockIdx.y - threadIdx.y - 1;

	/* Check that x and y are within the bounds of the image. */
	if (x >= width || y >= height) 
		return;

	/* 
		Set the limits of pixels to use. This is done 
		so that we do not use an index which is out of bounds.
	*/
	const unsigned int xstart = -1*(x>0)-1*(x>1);
	const unsigned int ystart = -1*(y>0)-1*(y>1);
	const unsigned int xsteps = (x<(width-1))+(x<(width-2))+(x>0)+(x>1);
	const unsigned int ysteps = (y<(height-1))+(y<(height-2))+(height>0)+(height>1);

	/* Run the covolution for each spectrum. */
	for (int z = 0; z<SPECTRUMS; z++){
		/* Initialize the result variables */
		float filterSum = 0.0f;
		float smoothPix = 0.0f;	

		/* Run all over the neighbour pixels */
		for (unsigned int y0 = 0; y0 <= ysteps; y0++) {
			for (unsigned int x0 = 0; x0 <= xsteps; x0++) {
				smoothPix += static_cast< float >(inputImage[(z * width * height) + ((y0+ystart+y) * width) + (x0+xstart+x)]) * filter[y0][x0];
				filterSum += filter[y0][x0];
			}
		}

		/* Save the computed value. */
		smoothPix /= filterSum;
		smoothImage[(z * width * height) + (y * width) + x] = static_cast< unsigned char >(smoothPix+0.5f); 
	}
}

/* 
	Second-level wrapper function for calling the kernel.
	It runs the kernel given the allocated space on the device and the specifications of the image. 
	The function first sets its architecture (or design) such as the number of threads, block size e.t.c.
*/
void triangularSmooth(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * smoothImage, const float * hostFilter){
	/* Copy the filter to the global constant memory */
	cudaError_t err;
  	err = cudaMemcpyToSymbol(filter, hostFilter, FILTER_LENGTH*sizeof(float));
  	cudaError(err, "memcpy for filter failed");	

	/* 
		Define the number of active threads per block and the block size
		for the kernel that transforms the pixels inside the borders. 
		Then start the kernel. Finally, check if any error occured in
		the execution of the kernel. We have a total of 512 threads.
	*/
  	const dim3 blockSize(16, 16);
  	const dim3 gridSize(ceil(((1.0*(width-4)))/blockSize.x), ceil((1.0*(height-4))/blockSize.y));
  	_triangularSmooth<<<gridSize, blockSize>>>(width, height, inputImage, smoothImage);
	err = cudaGetLastError();
	cudaError(err, "kernel [0] problem occured");

	/* 
		Define the number of active threads per block and the block size
		for the kernel that transforms the pixels on the left and top border.
		Then start the kernel for both cases. Finally, check if any error occured in
		the execution of the kernels. We have a total of 512 threads.
	*/

	// Use kernel for the left border. 
	const dim3 blockSizeLeft(2, 256);
	const dim3 gridSizeLeft(1, ceil((float)height / blockSizeLeft.y));
  	_triangularSmoothTopLeft<<<gridSizeLeft, blockSizeLeft>>>(width, height, inputImage, smoothImage);
	err = cudaGetLastError();
	cudaError(err, "kernel [1] problem occured");

	// Use kernel for the top border.
	const dim3 blockSizeTop(256, 2);
	const dim3 gridSizeTop(ceil((float)width / blockSizeTop.x), 1);
  	_triangularSmoothTopLeft<<<gridSizeTop, blockSizeTop>>>(width, height, inputImage, smoothImage);
	err = cudaGetLastError();
	cudaError(err, "kernel [2] problem occured");

	/* 
		Define the number of active threads per block and the block size
		for the kernel that transforms the pixels on the right border.
		Then start the kernel. Finally, check if any error occured in
		the execution of the kernels. We have a total of 512 threads.
	*/
  	_triangularSmoothRight<<<gridSizeLeft, blockSizeLeft>>>(width, height, inputImage, smoothImage);
	err = cudaGetLastError();
	cudaError(err, "kernel [3] problem occured");
 
	/* 
		Define the number of active threads per block and the block size
		for the kernel that transforms the pixels on the bottom border.
		Then start the kernel. Finally, check if any error occured in
		the execution of the kernels. We have a total of 512 threads.
	*/	
  	_triangularSmoothBottom<<<gridSizeTop, blockSizeTop>>>(width, height, inputImage, smoothImage);
	err = cudaGetLastError();
	cudaError(err, "kernel [4] problem occured");
	cudaDeviceSynchronize();
}	