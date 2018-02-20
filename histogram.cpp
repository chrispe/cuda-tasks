#include <CImg.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <libgen.h>
#include <Timer.hpp>

using cimg_library::CImg;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using std::string;
using LOFAR::NSTimer;

/* Running options */
#define cimg_display_type 	0	// Sets the type of error displaying.
#define cimg_debug 			0 	// Disables the debugging errors of CImg.
#define hist_debug			1 	// Indicates if debugging should be turned on (use timers).

/* Some histogram constants */
#define HISTOGRAM_SIZE	256		// The size of the histogram.
#define BAR_WIDTH 		4		// The bar width of each value.

/* Define the specs of the testing GPU so that we compute the utilization. */
#define GTX_480_MEM_BANDWIDTH 	133.9	// GB per sec
#define GTX_480_PERFORMANCE 	1088.6  // GFlops

/* Define the index of each timer. */
#define TOTAL_EXEC_TIME			0
#define KERNEL_TIME				1
#define TIME_HOST_TO_DEVICE		2
#define TIME_DEVICE_TO_HOST		3
#define ALLOCATION_TIME			4
#define TOTAL_TIMERS			5 

/* The external function which is responsible for calling the kernel. */
extern void histogram1D(const unsigned int width, const unsigned int height, const unsigned char * inputImage, unsigned char * grayImage, unsigned int * histogram, const size_t pitch);

/* Define the timers. */
NSTimer timer[TOTAL_TIMERS];

/* Define the description of each timer. */
const char * timer_label[TOTAL_TIMERS] = {"Total execution time", "Kernel time", \
						 	   			  "Host to device (memory transfer) time",\
						       			  "Device to host (memory transfer) time",\
						   	   			  "Device allocation time"};
/* 	
	Initializes the timers used for benchmarking the times for each
	process. 
*/
void init_timers(){
	unsigned int i;
	for(i=0;i<TOTAL_TIMERS;i++)
		timer[i] = NSTimer(timer_label[i]);
}

/* 
	Saving the time results to a file using stdout. 
	File format: name, width, height, [times], throughput, bandwidth, [utilizations].
*/
void 
write_results(char * filename, const unsigned int width, const unsigned int height,
			  	const double throughput, const double bandwidth, const double comp_utl,
			  		const double mem_util){
	unsigned int i;
	cout << filename << " ";
	cout << width << " "; 
	cout << height << " "; 
	for(i=0;i<TOTAL_TIMERS;i++)
		cout << 1000.0 * timer[i].getElapsed() << " ";
	cout << throughput << " "; 
	cout << bandwidth << " "; 
	cout << comp_utl << " "; 
	cout << mem_util << endl; 
}

/* 
	Prints the result of each timer, utilizations and more.  
*/
void print_results(char * path, const unsigned int width, const unsigned int height){
	/* Print the timer results. */
	unsigned int i;
	cerr << "______________________" << endl;
	for(i=0;i<TOTAL_TIMERS;i++)
		cerr << fixed << timer_label[i] << ": " << 1000.0 * timer[i].getElapsed() << " ms" << endl;

	/* Print the throughput, bandwidth and their utilization. */
	double throughput = (static_cast< long long unsigned int >(width) * height * 4 * 6) / 1000000000.0 / timer[KERNEL_TIME].getElapsed();
	double bandwidth = (static_cast< long long unsigned int >(width) * height * (4 * sizeof(unsigned char))) / 1000000000.0 / timer[KERNEL_TIME].getElapsed();	
	double comp_util = 100.0*(throughput/GTX_480_PERFORMANCE);
	double mem_util = 100.0*(bandwidth/GTX_480_MEM_BANDWIDTH);
	cerr << "Computational throughput: " << throughput << " GFLOPS/s" << endl;
	cerr << "Bandwidth: " << bandwidth << " GB/s" << endl;
	cerr << "Computational utilization: " << setprecision(2) <<  comp_util << '%' << endl;
	cerr << "Memory utilization: " << setprecision(2) << mem_util << '%' << endl;

	/* Also write results to stdout. */
	write_results(basename(path), width,height,throughput,bandwidth,comp_util,mem_util);
}

/* 
	Checks for any CUDA error. If any is detected, 
	an error message is printed and then it terminates. 
*/
void cudaError(cudaError_t e, const char * msg) {
	if (e != cudaSuccess) {
		cerr << "CUDA runtime error: " << msg << " (" << cudaGetErrorString(e) << ")" << endl;
		assert(e == cudaSuccess);
	}
}

/* 
	Checks for memory allocation error. If detected, 
	an error message is printed and then it terminates. 
*/
void memError(void * p, const char * msg) {
	if (!p) {
		cerr << "Memory allocation error: " << msg << endl;
		assert(p);
	}
}

/* 	
	Makes the appropriate allocations for memory use on the device.
	The allocations include memory for input and output. The input
	is set equal to the image while the output is set to plain zeros.
*/
void 
cuda_alloc_mem(unsigned char ** deviceImageIn, unsigned char ** deviceImageOut, unsigned int ** deviceHistogram, 
				const unsigned char * hostImageIn, const unsigned int width, const unsigned int height, size_t * pitch){
	cudaError_t err;

	/* Make allocations for input and output on the device. */
	// We want to use a pitched malloc since the data represents a 2D array.
	// Therefore, it easies the access to the data. We read the array row
	// by row so that we make less memory accesses (coalesing).
	// Reading row by row also reduces the bank conflicts, since the
	// threads access different memory banks. The value of pitch indicates
	// the number of elements that each row has. That may differ to the 
	// original number of elements in the original 2D array (which represents
	// the image), but it is done for optimal aligning in order to achieve
	// what it was just introduced. Each pixel is represented by 3 elements
	// on the y-axis. That is why we allocate 3*height elements. We also tried
	// using pinned memory in order to decrease data transfer timers, but since
	// the image is used only once, loading it to pinned memory (using cudaMallocHost)
	// takes more time than to allocating it to paged memory.  

	// Example:
	// Row 1 represents the red color of the first pixels.
	// Row 2 represents the green color of the first pixels.
	// Row 3 represents the blue color of the first pixels.
	// and so on...
	//
	// For more info: 
	// http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/
	// online/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html
	//
	// Finally, we also need to allocate an array to save the number of times 
	// each value occures (histogram counter). 

	timer[ALLOCATION_TIME].start();
  	err = cudaMallocPitch((void **)(deviceImageOut), pitch, width * sizeof(unsigned char), height); 
  	cudaError(err, "allocation for deviceImageOut failed");			
  	err = cudaMallocPitch((void **)(deviceImageIn), pitch, width * sizeof(unsigned char), 3*height);									 
  	cudaError(err, "allocation for deviceImageIn failed");
  	err = cudaMalloc((void **)(deviceHistogram), HISTOGRAM_SIZE*sizeof(unsigned int));
	cudaError(err, "allocation for deviceHistogram failed");
  	timer[ALLOCATION_TIME].stop();

  	/* Set the values of the device input array. */
  	timer[TIME_HOST_TO_DEVICE].start();												 
	err = cudaMemcpy2D((void *)(*deviceImageIn), *pitch, hostImageIn,\
	 			width * sizeof(unsigned char), width * sizeof(unsigned char),\
				3*height, cudaMemcpyHostToDevice);
  	cudaError(err, "memcpy for deviceImageIn failed");		
  	timer[TIME_HOST_TO_DEVICE].stop();	

  	/* Set the default values of the histogram - all set to zero. */
  	err = cudaMemset((void *)(*deviceHistogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));
  	cudaError(err, "memcpy for deviceHistogram failed");

  	/* Set the default values of the device output. */
  	err = cudaMemset2D((void *)(*deviceImageOut), *pitch, 0, width * sizeof(unsigned char), height);
  	cudaError(err, "memset2D for deviceImageOut failed");	
}

/* 
	Releases the use of the device. 
	- Free any allocated memory on the device..
	- Apply a reset on the device.
*/
void cuda_release(unsigned char * cp1, unsigned char * cp2, unsigned int * cp3){
	cudaFree(cp1);
	cudaFree(cp2);
	cudaFree(cp3);
	cudaDeviceReset();	
}

/* Generates the histogram image given the counter of each value. */
CImg< unsigned char > * generate_hist_img(const unsigned int * histogramCounters){

	/* Create a new image object with the given dimensions of the histogram */
	CImg< unsigned char > * histImage = new CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	unsigned char * imageData = histImage->data();

	/* Maximum value of histogram */
	unsigned int max = 0;

	/* Get the maximum value of the histogram in order to scale. */
	for (unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) {
		if ( histogramCounters[i] > max ) {
			max = histogramCounters[i];
		}
	}

	/* Set the image data. */
	for (int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH) {
		unsigned int value = HISTOGRAM_SIZE - ((histogramCounters[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);
		for (unsigned int y = 0; y < value; y++ ) {
			for ( int i = 0; i < BAR_WIDTH; i++ ) {
				imageData[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( int y = value; y < HISTOGRAM_SIZE; y++ ) {
			for ( int i = 0; i < BAR_WIDTH; i++ ) {
				imageData[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}

	/* Return a pointer to the image. */
	return histImage;
}

/* 
	This is the caller function (first level wrapper). 
	This function has to be called in order to execute the kernel, 
	as it makes the required initializations. It actually makes the
	required memory allocations and transfers as well. Then we execute 
	the histogram function and return its results. 
*/
CImg< unsigned char > * cuda_histogram(CImg< unsigned char > * inputImage, CImg< unsigned char > * grayImage){
	/* Some useful variables. */
	size_t pitch;
	cudaError_t err;
	unsigned int *deviceHistogram;
	unsigned int width = inputImage->width();
	unsigned int height = inputImage->height();
	unsigned char *deviceImageIn, *deviceImageOut;

	/* Allocate the input/output memory on GPU. */							
	timer[TOTAL_EXEC_TIME].start();
	cuda_alloc_mem(&deviceImageIn, &deviceImageOut, &deviceHistogram, inputImage->data(), width, height, &pitch);

	/* Start the gray-histogram process. We call the wrapper 
	   which in turn then executes the kernel. */
	timer[KERNEL_TIME].start();
	histogram1D(width, height, deviceImageIn, deviceImageOut, deviceHistogram, pitch);
	timer[KERNEL_TIME].stop();

	/* Check if the kernel return an error. */
	err = cudaGetLastError();
	cudaError(err, "kernel problem occured");

	/* Copy results to host. */
	// We need to copy the gray image and the histogram counters as well.
	unsigned int * histogramCounters = (unsigned int * )malloc(HISTOGRAM_SIZE*sizeof(unsigned int));
	memError(histogramCounters, "allocation for histogramCounters failed");
	unsigned char * grayData = (unsigned char*)(grayImage->data());
	timer[TIME_DEVICE_TO_HOST].start();
	err = cudaMemcpy2D(grayData, width, deviceImageOut, pitch, width * sizeof(unsigned char),\
					   height, cudaMemcpyDeviceToHost);
	cudaError(err, "memcpy for grayData failed");
	err = cudaMemcpy(histogramCounters, deviceHistogram, HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaError(err, "memcpy for histogramCounters failed");
	timer[TIME_DEVICE_TO_HOST].stop();
	timer[TOTAL_EXEC_TIME].stop();

	/* Release the device. */
	cuda_release(deviceImageIn, deviceImageOut, deviceHistogram);

	/* Create the histogram image data given the counters and return it. */
	return generate_hist_img(histogramCounters);
}

int main(int argc, char *argv[]) {
	/* Check the given parameters. */
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

	/* Load the image. */
	CImg< unsigned char > inputImage;
	try{
		inputImage = CImg< unsigned char >(argv[1]);
		cerr << "Image '" << argv[1] << "' has been loaded." << endl;
	}
	catch (cimg_library::CImgIOException&) {
		cerr << "Unable to read image file '" << argv[1] << "'. Exiting. " << endl;
		return 1;
	}

	/* Check the validity of the image. */
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

	/* Convert the input image to grayscale. */
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	/* Initialize the timers (used for benchmarking). */
	init_timers();

	/* Apply the transformation and get it on a new CImg object. */
	cerr << "Creating the histogram of the gray-scaled image..." << endl;
	CImg< unsigned char > * histogram = cuda_histogram(&inputImage, &grayImage);

	/* Save the histogram image and the gray-scale image. */
	cerr << "Saving image histogram to '" << ("./" + string(argv[1]) + ".hist.cuda.bmp").c_str() << "'..." << endl;
	histogram->save(("./" + string(argv[1]) + ".hist.cuda.bmp").c_str());
	cerr << "Saving gray-scaled image to '" << ("./" + string(argv[1]) + ".gray.cuda.bmp").c_str() << "'..." << endl;
	grayImage.save(("./" + string(argv[1]) + ".gray.cuda.bmp").c_str());

	/* Print the timer results. */
	#if hist_debug==1
		print_results(argv[1], inputImage.width(), inputImage.height());
	#endif

	/* Free the allocated image. */
	delete(histogram);

	return 0;
}