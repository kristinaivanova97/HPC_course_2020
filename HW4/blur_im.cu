//
// nvcc Blur.cu
// ./a.out Clown.256.ppm - original imge
// other Clown images are blurred with 2 different kenrel values

#include <stdio.h>

#define CHANNEL 3
#define N 1000

struct PPMImage {
    int width;
    int height;
    unsigned int bytes; //amount if bytes, as each pixel in 3 colors
    unsigned char *image;
    unsigned char *dev_img;
};
// Reads a color PPM image fname
int readImg(const char * fname, PPMImage & img, int & rgb_comp_color) {

    char p, s;
    FILE *file;

    if (!(file = fopen(fname, "rb")))
    {
        printf("Couldn't open file %s for reading.\n", fname);
        return 1;
    }

    fscanf(file, "%c%c\n", &p, &s);
    if (p != 'P' || (s != '6' && s!='3')) // P6 or P3
    {
        printf("Not a valid PPM file (%c %c)\n", p, s);
        exit(1);
    }

    fscanf(file, "%d %d\n", &img.width, &img.height);
    fscanf(file, "%d\n", &rgb_comp_color); //максимальное значения цвета

    int pixels = img.width * img.height; //total num of pixels
    img.bytes = pixels * 3;  // colored image with r, g, and b channels
    img.image = (unsigned char *)malloc(img.bytes);
    
    fread(img.image, sizeof(unsigned char), img.bytes, file);
    fclose(file);
    return 0;
}

// Write a color image into a file using PPM file format.
int writeOutImg(unsigned char * image, const char * fname, const PPMImage & blurred, const int  rgb_comp_color) {

    FILE *out;
    if (!(out = fopen(fname, "wb")))
    {
        printf("Couldn't open file for output.\n");
        return 1;
    }
    fprintf(out, "P6\n%d %d\n%d\n", blurred.width, blurred.height,  rgb_comp_color);
    fwrite(image, sizeof(unsigned char), blurred.bytes, out);
    fclose(out);
    return 0;
}

//Array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.
__global__ void blur(unsigned char* input_image, unsigned char* output_image, int width, int height, int filterWidth, double * filter, const int s) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) {
    	
        return;
    }

    int r = 0;
    int g = 0;
    int b = 0;
    for (int fx = 0; fx < filterWidth; fx++) {
      for (int fy = 0; fy < filterWidth; fy++) {
        int imagex = px + fx - filterWidth / 2;
        int imagey = py + fy - filterWidth / 2;
        //imagex = min(max(imagex,0),width-1);
        //imagey = min(max(imagey,0),height-1);
        if (imagex >=0 && imagex < width && imagey >= 0 && imagey < height)
        {
         	r += (input_image[imagey*width*3+imagex*3]);
         	g += (input_image[imagey*width*3+imagex*3+1]);
         	b += (input_image[imagey*width*3+imagex*3+2]);
	 }
      }
    }


    int index = width * 3 * py + px*3; //position of pixel
    output_image[index] = r/s;//output image is average of 9 pixels
    output_image[index+1] = g/s;
    output_image[index+2] = b/s;
}
__global__ void Init(int size, double *filter, double s) //size = kernelwidth**2
{
    int globalidx = threadIdx.x;


    if((globalidx<size)&&(globalidx%2 == 0)) filter[globalidx]=-1.0;
	else if((globalidx<size)&&(globalidx%2 != 0)) filter[globalidx]=1.0; 
    // if(globalidx<size) filter[globalidx] = globalidx;
	//if(globalidx<size) filter[globalidx]=globalidx/25;
    __syncthreads();
    s+=filter[globalidx];
}


int main(int argc, char **argv)
{

    unsigned char * img;
    unsigned char * output;
    unsigned char * output_image;
    int kernelw = 5;
    double * filter;
    double * filter_host;
    filter_host = (double *)malloc(kernelw*kernelw*sizeof(double));
    
    cudaMalloc((double**)&filter, sizeof(double)*kernelw*kernelw);
    cudaMemcpy(filter, filter_host, kernelw*kernelw*sizeof(double), cudaMemcpyHostToDevice);
    
    int s=25;
    Init<<<1,kernelw*kernelw>>>(kernelw*kernelw, filter, s);
    /*cudaMemcpy(filter_host,filter, sizeof(double)*kernelw*kernelw, cudaMemcpyDeviceToHost);
    
    for (i = 0; i<kernelw*kernelw; i++){
        s+=filter_host[i];
    }*/

    if(argc != 2)
    {
        printf("Usage: exec filename\n");
        exit(1);
    }
    char *fname = argv[1];
    PPMImage input;
    
    int  rgb_comp_color;
    
    if (readImg(fname, input, rgb_comp_color) != 0)  exit(1);
    float grid = (input.width * input.height);
    grid = grid * 3 * sizeof(unsigned char);

    cudaMalloc((void**)&img, grid);

    cudaMemcpy(img, input.image, grid, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&output, grid);

    dim3 block(16, 16);
    dim3 gridsize(input.width/block.x + 1, input.height / block.x + 1);

    printf("%d %d", input.width, input.height);
    blur <<<gridsize, block >>>(img, output,input.width,input.height, kernelw, filter, s);
    cudaDeviceSynchronize();
    //get output image from device
    output_image = (unsigned char *) malloc(grid);
    cudaMemcpy(output_image, output, grid, cudaMemcpyDeviceToHost);

    //use output image from device as image to print
    if (writeOutImg(output_image,"blurr.ppm", input,  rgb_comp_color) != 0)
        exit(1);

    cudaFree(img);//free up memory dedicated to pointers
    cudaFree(output);
    free(output_image);

    exit(0);
}


