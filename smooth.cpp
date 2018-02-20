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
#define smooth_debug		1 	// Indicates if debugging should be turned on (use timers).

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
extern void triangularSmooth(const unsigned int width, const unsigned int height, 
					 		 const unsigned char * inputImage, unsigned char * smoothImage, 
					 		 const float * hostFilter);

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
	double throughput = (static_cast< long long unsigned int >(width) * height * 321) / 1000000000.0 / timer[KERNEL_TIME].getElapsed();
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
cuda_alloc_mem(unsigned char ** deviceImageIn, unsigned char ** deviceImageOut, const unsigned char * hostImageIn,
				    const unsigned int width, const unsigned int height, const unsigned int spectrum){
	cudaError_t err;

	/* Make the allocations on the device */
	timer[ALLOCATION_TIME].start();
  	err = cudaMalloc((void **)(deviceImageOut), spectrum * width * height * sizeof(unsigned char)); 
  	cudaError(err, "allocation for deviceImageOut failed");			
  	err = cudaMalloc((void **)(deviceImageIn), spectrum * width * height * sizeof(unsigned char));									 
  	cudaError(err, "allocation for deviceImageIn failed");
  	timer[ALLOCATION_TIME].stop();

  	/* Set the values of the device input array and copy the constant filter to the device's global memory. */
  	timer[TIME_HOST_TO_DEVICE].start();												 
	err = cudaMemcpy((void *)(*deviceImageIn), hostImageIn, spectrum * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  	cudaError(err, "memcpy for deviceImageIn failed");	
  	timer[TIME_HOST_TO_DEVICE].stop();	

  	/* Set the default values of the device output. */
  	err = cudaMemset((void *)(*deviceImageOut), 0, spectrum * width * height * sizeof(unsigned char));
  	cudaError(err, "memset for deviceImageOut failed");	
}

/* 
	Releases the use of the device. 
	- Free any allocated memory on the device..
	- Apply a reset on the device.
*/
void cuda_release(unsigned char * cp1, unsigned char * cp2){
	cudaFree(cp1);
	cudaFree(cp2);
	cudaDeviceReset();	
}

/* 
	This is the caller function (first level wrapper). 
	This function has to be called in order to execute the kernel, 
	as it makes the required initializations. It actually makes the
	required memory allocations and transfers as well. Then we execute 
	the triangular smooth kernel and return its results. 
*/
CImg< unsigned char > * cuda_triangularSmooth(CImg< unsigned char > * inputImage){
	/* Some useful variables. */
	cudaError_t err;
	unsigned int width = inputImage->width();
	unsigned int height = inputImage->height();
	unsigned int spectrum = inputImage->spectrum();
	unsigned char *deviceImageIn, *deviceImageOut;

	/* Define the used filter in triangular smooth. */
	const float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,\
	 				  		1.0f, 2.0f, 2.0f, 2.0f, 1.0f,\
	 				  		1.0f, 2.0f, 3.0f, 2.0f, 1.0f,\
	 				  		1.0f, 2.0f, 2.0f, 2.0f, 1.0f,\
	 				  		1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	/* Allocate the input/output memory on GPU. */							
	timer[TOTAL_EXEC_TIME].start();
	cuda_alloc_mem(&deviceImageIn, &deviceImageOut, inputImage->data(), width, height, spectrum);

	/* Start the triangular smooth process. We call the wrapper 
	   which in turn then executes the kernel. */
	timer[KERNEL_TIME].start();
	triangularSmooth(width, height, deviceImageIn, deviceImageOut, filter);
	timer[KERNEL_TIME].stop();

	/* Copy results to host. */
	unsigned char * smoothedData = (unsigned char*)malloc(spectrum * width * height * sizeof(unsigned char));
	memError(smoothedData, "allocation for smoothedData failed");
	timer[TIME_DEVICE_TO_HOST].start();
	err = cudaMemcpy(smoothedData, deviceImageOut, spectrum * width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaError(err, "memcpy for smoothedData failed");
	timer[TIME_DEVICE_TO_HOST].stop();
	timer[TOTAL_EXEC_TIME].stop();

	/* Release the device. */
	cuda_release(deviceImageIn, deviceImageOut);

	/* Create the new CImg object containing the transformed image */
	CImg< unsigned char > * smoothedImage = new CImg< unsigned char >(smoothedData, width, height, 1, spectrum, true);	
	return smoothedImage;
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

	/* Initialize the timers (used for benchmarking). */
	init_timers();

	/* Apply the transformation and get it on a new CImg object. */
	cerr << "Applying the triangular smooth filter..." << endl;
	CImg< unsigned char > * smoothedImage = cuda_triangularSmooth(&inputImage);

	/* Save the transformed image. */
	cerr << "Saving image to '" << ("./" + string(argv[1]) + ".smooth.cuda.bmp").c_str() << "'..." << endl;
	smoothedImage->save(("./" + string(argv[1]) + ".smooth.cuda.bmp").c_str());

	/* Print the timer results. */
	#if smooth_debug==1
		print_results(argv[1], inputImage.width(), inputImage.height());
	#endif

	/* Free the allocated image. */
	delete(smoothedImage);

	return 0;
}