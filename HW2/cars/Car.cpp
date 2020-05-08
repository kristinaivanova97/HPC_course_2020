#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#define RGB_COMPONENT_COLOR 255

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

void changeColorPPM(PPMImage &img){
    for (int i=0; i<img.all; i++){
        img.data[i].red /= 2; 
    }
}

void movePPM(PPMImage &img)
{
    
    int num_shifts = 303;
    int i,j;
    int counter = 0;
    char filename[sizeof "car.ppm"];
    
    for (int shift = 0; shift < num_shifts; shift ++)
    {
        #pragma omp parallel for shared(img) private(i,j) firstprivate(filename) // i -raws, j - columns indices
        
        for(int i = 1; i < img.y; i++)
        {
            int pixel, pixel_new;

            for(int j = img.x; j >= 0; j--)
            {
                pixel = j + i*img.x; //due to raw structure of data
                pixel_new = j + i*img.x - 1;

                img.data[pixel].red = img.data[pixel_new].red;
                img.data[pixel].green = img.data[pixel_new].green;
                img.data[pixel].blue = img.data[pixel_new].blue;
            }
        }
        if (shift % 20 == 0)
        {
            sprintf(filename, "car%d.ppm", counter++);
            writePPM(filename, img);

        }
    }
}

int main(){

    PPMImage image;
    readPPM("car.ppm", image);
    movePPM(image);
    delete(image.data);
    return 0;
}

