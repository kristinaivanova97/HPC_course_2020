// nvcc Hisogram.cu
// ./a.out car.ppm

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

#define RGB_COMPONENT_COLOR 255
#define N 256

struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;

void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename); /*This data type represents the input file stream and is used to read information from files.*/
    if (file){
        std::string s; //only for read
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y; //При этом из входного потока читается последовательность символов до пробела - число колонок и рядов
        file >>rgb_comp_color; //максимального значения цвета
        img.all = img.x*img.y;
        std::cout << s << std::endl;
        std::cout << "x = " << img.x << "y = " << img.y << "all = " <<img.all;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue; //триплеты RGB
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}
/*
void writePPM(const char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out); //can write to file
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}
 */

void changeColorPPM(PPMImage &img,int * arr){
    for (int i=0; i<img.all; i++){
        arr[i] = (img.data[i].red + img.data[i].green + img.data[i].blue)/3;
    }
}
__global__ void Histogram_CUDA(int* Image, int* Histogram);

 void Histogram(int* Image, int Height, int Width, int Channels, int* Histogram){
    int* Imaged;
    int* Histogramd;

    cudaMalloc((void**)&Imaged, Height * Width * Channels * sizeof(int));
    cudaMalloc((void**)&Histogramd, N * sizeof(int));

    cudaMemcpy(Imaged, Image, Height * Width * Channels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Histogramd, Histogram, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(Width, Height);
    Histogram_CUDA << <Grid, 1 >> >(Imaged, Histogramd);
    
    cudaMemcpy(Histogram, Histogramd, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(Histogramd);
    cudaFree(Imaged);
}

__global__ void Histogram_CUDA(int* Image, int* Histogram){
    int x = blockIdx.x;
    int y = blockIdx.y;

    int Image_Idx = x + y * gridDim.x;
    atomicAdd(&Histogram[Image[Image_Idx]], 1);
}
int main()
{
    PPMImage image;
    readPPM("car.ppm", image);
    int * arr = (int*)malloc(image.all*sizeof(int));
    
    changeColorPPM(image, arr);
        
    int Histogram_GrayScale[256] = { 0 };

    Histogram(arr, image.x, image.y, 1, Histogram_GrayScale);
    
    for (int i = 0; i < 256; i++){
        std::cout << "Histogram_GrayScale[" << i << "]: " << Histogram_GrayScale[i] << std::endl;
    }
}
