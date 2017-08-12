 #include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <assert.h>
#include <sstream>

using namespace std;

#define MAX_IMAGE_WIDTH 256
#define MAX_IMAGE_HEIGHT 64
//Compress 64 * 64 -> 32 * 32, save memory of gpu
#define COMPRESS_RADIO 2

//First char never starts before y=10
#define CHAR_START_X 10

//threshold for segment(char number as output)
#define CHAR_X_THRESHOLD 2
#define CHAR_Y_THRESHOLD 9
#define EMPTY_X_THRESHOLD 2
#define EMPTY_Y_THRESHOLD 1
#define SPARSE_THRESHOLD 6
#define SPLIT_X_OFFSET 1
#define MAX_CHAR_NUM 4
#define MIN_CHAR_SCOPE 7
#define MAX_CHAR_SCOPE 28
#define CHAR_BLOCK_THRESHOLD 12

//threshold for real segment(segmented files as output)
#define REAL_SEG_CHAR_Y 4
#define REAL_SEG_CHAR_X 2
#define REAL_SEG_X_OFFSET 3
#define REAL_SEG_COMPARE_OFFSET 5
#define REAL_SEG_ZERO_OFFSET 10

//original image binarization after translation
 int output[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT];
//image binarization after translation and compression
int* compress_output;
//image width
int img_width;
//image height
int img_height;

int output_buf[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT];  

//The tga header information in file
typedef struct tagTgaHeader
{
   char  ubImageInfoByteCnt;    //The image information length, in byte
   char  ubColorTableExist;       //0-have not color table,1-have 
   char  ubImageType;              //Image type,2-uncompare RGB image,10-compare RGB image
   char  ubColorTableInfo[5];    //Color table information
   char  ubImageInfo[10];          //Image information
}TgaHeader, *LPTGAHEADER;

void remove_noise(){
    
    for(int i = 1; i < img_width - 1; i++){
        for(int j = 1; j < img_height - 1; j++){
            if(output[i + j * img_width] == 0 &&
                output[i + (j + 1) * img_width] == 0 &&
                output[i + (j + 2) * img_width] >= 200 ){
                int flag = 0, sum = 0;
                for(int n = 0; n < img_height; n++){
                    if(output[i + n * img_width] < 200){
                        sum++;
                    }
                }
                if(sum < 4){
                    output[i + (j + 1) * img_width] = 240;
                    output[i + j * img_width] = 240;
                }
            }
        }
    }
     
    for(int i = 1; i < img_width - 1; i++){
        int flag = 0, sum = 0;
        for(int n = 0; n < img_height; n++){
            if(output[i + n * img_width] < 200){
                sum++;
            }
        }
        if(sum < 4){
            for(int n = 0; n < img_height; n++){
                output[i + n * img_width] = 240;
            }
        }
    }    
    // */
    for(int i = 2; i < img_width - 2; i++){
        for(int j = 2; j < img_height - 2; j++){
            if(output[i+ j * img_width] == 0){
                int flag = 0, sum = 0;
                for(int m = i - 2; m < i + 2; m++){
                    for(int n = j - 2; n < j + 2; n++){
                        int value = output[m + n * img_width];                    
                        if(value == 0){
                            sum++;
                        }else if(value == 240){
                            ;
                        }else{
                            flag = 1;
                            break;
                        }
                    }
                    if(sum >= CHAR_BLOCK_THRESHOLD){
                        flag = 1;
                    }
                    if(flag == 1)
                        break;                
                }
                if(flag == 0){
                    output[i+ j * img_width] = 240;
                }
            }
        }
    }
    //cout << "-----------------" << endl;
    
    for(int j = img_height - 1; j >= 0; j--){
        for(int i = 0; i < img_width; i++){
            //cout << output[i + j * img_width] << "\t";
            if(output[i + j * img_width] != 240){
                output[i + j * img_width] = 1;
            }else{
                output[i + j * img_width]  = 0;
            }
        }
        //cout << endl;    
    }
}

/*
* deblur, specific for captGen program
*/
void deblur(){
    for(int j = img_height - 1; j >= 0; j--){
        for(int i = 0; i < img_width; i++){
            //int tmp = output[i + j * img_width];
            //cout <<  tmp << "\t";
        }
        //cout << endl;
    }   
    //cout << "------------------" << endl;
    for(int j = img_height - 3; j >=0; j--){
        for(int i = img_width - 3; i >= 0; i--){
            int tmp = output[i + j * img_width] * 4 - 
                output[i + 1 + j * img_width] - output[i + (j + 1) * img_width] - 
                output[i + 1 + (j + 1) * img_width];   
            if(tmp < 0){
                tmp = 0;
            }else if(tmp == 240 || tmp >= 240){
                tmp = 240;
            }else{
                int flag = tmp % 16;             
                switch(flag){
                case 0:
                    break;
                case 1:
                    tmp++;
                    break;
                case 2:
                    tmp += 2;
                    break;
                case 3:
                    tmp += 3;
                    break;
                default :
                    cout << tmp << "!" << endl;             
                    assert(0);                    
                }
            }
            output[i + j * img_width] = tmp;
        }
    }
    for(int j = img_height - 1; j >= 0; j--){
        for(int i = 0; i < img_width; i++){
            //cout << output[i + j * img_width] << "\t";
        }
        //cout << endl;
    }    
    //cout << "------------------" << endl;    
}

/* 
* This function is used for translating TGA image into binarization
* For the background of the captcha generation program used in CUDA 2012 content  is (240, 240, 240)
* As a result, the pre-processing program leverages the feature
*/
void binarization_translation(string ifilename){
    TgaHeader header;
    memset(&header, 0, sizeof(header));
    string ofilename = ifilename + ".binary";
    ifstream is;
    is.open(ifilename.c_str());
    if(is.is_open()){
        is.read((char*)&header, sizeof(header));
        if(header.ubImageType != 2 && header.ubImageType != 10){
            cout << "TGA TYPE ERROR" << endl;
            assert(0);
        }else{
            int m_iImageWidth = (header.ubImageInfo[5] << 8) + header.ubImageInfo[4];
            int m_iImageHeight = (header.ubImageInfo[7] << 8) + header.ubImageInfo[6];
            int m_iBitsPerPixel = header.ubImageInfo[8];
            //cout << "W: " << m_iImageWidth << endl;
            //cout << "H: " << m_iImageHeight << endl;
            //cout << "Bits Per Pixel: " << m_iBitsPerPixel << endl;
            if(m_iImageWidth <= 0 || m_iImageHeight <= 0 ||
                (m_iBitsPerPixel != 24 && m_iBitsPerPixel != 32)){
                cout << "TGA IMAGE INFO ERROR" << endl;
            }else{
                int m_iImageDataSize = m_iImageWidth * m_iImageHeight * m_iBitsPerPixel / 8;
                char info[255];
                memset((char*)info, 0, 255 * sizeof(char));
                is.read(info, header.ubImageInfoByteCnt);
                if(header.ubImageType == 2){
                    //cout << "UnCompressed Type" << endl;
                    unsigned char* m_pImageData = new unsigned char[m_iImageDataSize];
                    memset(m_pImageData, 0, sizeof(unsigned char) * m_iImageDataSize);
                    is.read((char*)m_pImageData, m_iImageDataSize);
                    int i = 0, j = 0, k = 0;
                    //output = new int[m_iImageWidth * m_iImageHeight];
                    memset(output, 0, sizeof(int) * m_iImageWidth * m_iImageHeight);
                    for(i = 0; i < m_iImageDataSize; i+=3, j++){
                        unsigned int red = m_pImageData[i];
                        unsigned int green = m_pImageData[i + 1];
                        unsigned int blue = m_pImageData[i + 2]; 
                         
                        if(red == green && red == blue){
                            output[j] = red;
                        }else{
                            cout << "RGB are not the same!" << endl;
                            assert(0);
                        }   
                        // */
                    }

                    //deblur, locate and remove noise line before compression
                    img_width = m_iImageWidth;
                    img_height = m_iImageHeight ;  
                    //deblur
                    deblur();       
                    remove_noise();
                    
                    img_width = m_iImageWidth / COMPRESS_RADIO;
                    img_height = m_iImageHeight / COMPRESS_RADIO;
                    compress_output = new int[sizeof(int) * (img_width) * (img_height)];

                    int m = 0, n = 0;
                    for(i = m_iImageHeight - 1, m = 0; i >= 0; i-=COMPRESS_RADIO, m++){
                        for(j = 0, n = 0; j <  m_iImageWidth; j+=COMPRESS_RADIO, n++){
                            //cout << ((output[i *m_iImageWidth + j] == 0) ? " " : "o");
                            //os << ((output[i *m_iImageWidth + j] == 0) ? " " : "o");       
                            float tmp = 0;
                            int ci = 0, cj = 0;
                            for(ci = 0; ci < COMPRESS_RADIO; ci++){
                                for(cj = 0; cj < COMPRESS_RADIO; cj++){
                                    tmp += output[(i - ci) * m_iImageWidth + j + cj];
                                }
                            }
                            //tmp = output[i *m_iImageWidth + j] + output[i *m_iImageWidth + j + 1] 
                            //    + output[(i-1) *m_iImageWidth + j] + output[(i-1) *m_iImageWidth + j + 1];
                            //tmp /= (COMPRESS_RADIO * COMPRESS_RADIO);
                            if(tmp >= COMPRESS_RADIO){
                                compress_output[m * img_width + n] = 1;
                            }else{
                                compress_output[m * img_width + n] = 0;
                            }
                        }
                    }
                    //fill_back();
                    //cout << "Done" << endl;
                }else if(header.ubImageType == 10){
                    cout << "Compressed Type" << endl;
                    cout << "NOT BE PROCESSED" << endl;
                    assert(0);
                }else{
                }
            }
        }
        is.close();
    }else{
        cout << "FILE ERROR" << endl;
    }   
}

/*
* Verify the segments result
*/
void verify_segment(int *cnum_y_array, int *cline_start, int *cline_end, int &total_segs){
    int cscope = 0;
    int i = 0, j = 0;
    for(i = 0; i < total_segs; i++){
        cscope = cline_end[i] - cline_start[i];
        //Merge
        if(cscope < MIN_CHAR_SCOPE && total_segs > 1){
            if((cline_end[i+1] - cline_start[i+1]) < MIN_CHAR_SCOPE){
                //cout << "MERGE ";
                cline_end[i] = cline_end[i + 1];
                total_segs--;
                //cout << total_segs << "!\n";
                for(j = i + 1; j < total_segs - 1; j++){
                    cline_start[j] = cline_start[j + 1];
                    cline_end[j] = cline_end[j + 1];
                }
            }
        }
        //Split
        else if(cscope > MAX_CHAR_SCOPE && total_segs < MAX_CHAR_NUM){
            int split_flag = 0, sparse_num = 0;
            for(j = cline_start[i]; j <= cline_end[i]; j++){
                if(cnum_y_array[j] <= SPARSE_THRESHOLD)
                    sparse_num++;
            }
            if(sparse_num > SPARSE_THRESHOLD){
                split_flag = 1;
            }
            if(split_flag == 1){
                //cout << "SPLIT ";
                total_segs++;
                //cout << total_segs << "!\n";
                for(j = total_segs -1; j > i; j--){
                    cline_start[j] = cline_start[j - 1];
                    cline_end[j] = cline_end[j - 1];
                }     
                
                cline_end[i] = (cline_start[i] + cline_end[i + 1]) / 2;
                cline_start[i + 1] = cline_end[i] + 1;            
            }
        }else{
        }
    }
}

/*
*  Split image 
*  Return the number of segments 
*/
int segment(int *image_buf){
    int total_segs = 0;
    int i = 0, j = 0;
    //Find char location
    int cline_located_flag = 0, sline_located_flag = 0;
    int cnum_y_array[img_width];
    for(i = 0; i < img_width; i++){
        cnum_y_array[i] = -1;
    }
    for(i = CHAR_START_X; i < img_width; i++){
        for(j = 0; j < img_height; j++){
            if(image_buf[j * img_width + i] == 1){
                cnum_y_array[i]++;
            }
        }    
        //cout << cnum_y_array[i] << " ";
    }
    //cout << endl;
    int last_index = 0;
    int seq_num = 0;
    int cline[MAX_CHAR_NUM] = {0};
    int cline_start[MAX_CHAR_NUM] = {0};
    int cline_end[MAX_CHAR_NUM] = {0};
    for(i = CHAR_START_X; i < img_width; i++){
        if(cline_located_flag == 0 && cnum_y_array[i] > CHAR_Y_THRESHOLD){
            if(i - last_index == 1){
                seq_num++;
            }else{
                seq_num = 1;
            }
            last_index = i;
            if(seq_num == CHAR_X_THRESHOLD){
                cline_located_flag = 1;
                cline[total_segs] = i - CHAR_X_THRESHOLD + 1;
                cline_start[total_segs] = cline[total_segs] - SPLIT_X_OFFSET;
                last_index = 0;
                seq_num = 0;
            }
        }
        if(cline_located_flag == 1 && cnum_y_array[i] < EMPTY_Y_THRESHOLD){
            if(i - last_index == 1){
                seq_num++;
            }else{
                seq_num = 1;
            }
            last_index = i;
            if(seq_num == EMPTY_X_THRESHOLD){   
                cline_end[total_segs] = i - EMPTY_X_THRESHOLD + 1;
                last_index = 0;
                seq_num = 0;
                //for next char
                cline_located_flag = 0;
                total_segs++;
                if(total_segs == MAX_CHAR_NUM){
                    break;
                }
            }
        }
    }
    verify_segment(cnum_y_array, cline_start, cline_end, total_segs);
    //modify the segment line
    /*
    for(i = 0; i < total_segs; i++){
        //for(j = 0; j < img_height; j++){
            image_buf[cline_start[i]] = 3;
            image_buf[cline_end[i]] = 3;
        //}
    }
    // */
    return total_segs;
}

int real_segment(int fid, int* image_buf, int seg_num, string ifilename){
    int total_segs = 0;
    int i = 0, j = 0;
    //Find char location
    int cnum_y_array[img_width];
    for(i = 0; i < img_width; i++){
        cnum_y_array[i] = 0;
    }
    for(i = 0; i < img_width; i++){
        for(j = 0; j < img_height; j++){
            if(image_buf[j * img_width + i] == 1){
                cnum_y_array[i]++;
            }
        }    
        //cout << cnum_y_array[i] << " ";
    }    
    //cout << endl << endl;
    int last_index = 0;
    int seq_num = 0;
    int cline[MAX_CHAR_NUM] = {0};
    int cline_start[MAX_CHAR_NUM] = {0};
    int cline_end[MAX_CHAR_NUM] = {0};    
    int mid;
    int min;
    int min_index;
    switch(seg_num){
    case 1:
        for(i = CHAR_START_X; i < img_width; i++){
            if(cnum_y_array[i] >= REAL_SEG_CHAR_Y){
                if(i - last_index == 1){
                    seq_num++;
                }else{
                    seq_num = 1;
                }
                last_index = i;
                if(seq_num == REAL_SEG_CHAR_X){
                    cline[0] = i - REAL_SEG_CHAR_X + 1;
                    cline_start[0] = cline[0] - REAL_SEG_X_OFFSET;
                    cline_end[0] = cline_start[0] + img_width - 1;
                    last_index = 0;
                    seq_num = 0;
                    break;
                }
            }
        }
        //cout << cline[0] << endl;
        break;
    case 2:
        for(i = CHAR_START_X; i < img_width; i++){
            if(cnum_y_array[i] > REAL_SEG_CHAR_Y){
                if(i - last_index == 1){
                    seq_num++;
                }else{
                    seq_num = 1;
                }
                last_index = i;
                if(seq_num == REAL_SEG_CHAR_X){
                    cline[0] = i - REAL_SEG_CHAR_X + 1;
                    cline_start[0] = cline[0] - REAL_SEG_X_OFFSET;
                    //cline_end[0] = cline[0] + REAL_SEG_X_OFFSET * 2;
                    last_index = 0;
                    seq_num = 0;
                    break;
                }
            }
        }        

        for(i = img_width - 1; i > CHAR_START_X; i--){
            if(cnum_y_array[i] > REAL_SEG_CHAR_Y){
                if(last_index- i == 1){
                    seq_num++;
                }else{
                    seq_num = 1;
                }
                last_index = i;
                if(seq_num == REAL_SEG_CHAR_X){
                    cline[1] = i + REAL_SEG_CHAR_X - 1;
                    cline_end[1] = cline[1] + REAL_SEG_X_OFFSET;
                    //cline_end[0] = cline[0] + REAL_SEG_X_OFFSET * 2;
                    last_index = 0;
                    seq_num = 0;
                    break;
                }
            }
        }                
        mid = (cline_start[0] + cline_end[1]) / 2;
        min_index = 0;
        min = 100;
        for(j = mid - REAL_SEG_ZERO_OFFSET; j < mid + REAL_SEG_ZERO_OFFSET; j++){
            if(cnum_y_array[j] == 0){
                min = 0;
                min_index = j;
            }
        }
        if(min != 0){
            min = 100;
            for(j = mid -REAL_SEG_X_OFFSET; j < mid + REAL_SEG_X_OFFSET; j++){
                if(cnum_y_array[j] < min){
                    min = cnum_y_array[j];
                    min_index = j;
                }
            }
        }
        cline_end[0] = min_index;
        cline_start[1] = cline_end[0] + 1;
        //cout << cline_end[0] << " ";
        break;
    case 3:
        for(i = CHAR_START_X; i < img_width; i++){
            if(cnum_y_array[i] > REAL_SEG_CHAR_Y){
                if(i - last_index == 1){
                    seq_num++;
                }else{
                    seq_num = 1;
                }
                last_index = i;
                if(seq_num == REAL_SEG_CHAR_X){
                    cline[0] = i - REAL_SEG_CHAR_X + 1;
                    cline_start[0] = cline[0] - REAL_SEG_X_OFFSET;
                    //cline_end[0] = cline[0] + REAL_SEG_X_OFFSET * 2;
                    last_index = 0;
                    seq_num = 0;
                    break;
                }
            }
        }        

        for(i = img_width - 1; i > CHAR_START_X; i--){
            if(cnum_y_array[i] > REAL_SEG_CHAR_Y){
                if(last_index- i == 1){
                    seq_num++;
                }else{
                    seq_num = 1;
                }
                last_index = i;
                if(seq_num == REAL_SEG_CHAR_X){
                    cline[2] = i + REAL_SEG_CHAR_X - 1;
                    cline_end[2] = cline[2] + REAL_SEG_X_OFFSET;
                    //cline_end[0] = cline[0] + REAL_SEG_X_OFFSET * 2;
                    last_index = 0;
                    seq_num = 0;
                    break;
                }
            }
        }                   
        mid = (cline_end[2] - cline_start[0]) / 3 + cline_start[0];
        min_index = 0;
        min = 100;
        for(j = mid - REAL_SEG_ZERO_OFFSET; j < mid + REAL_SEG_ZERO_OFFSET; j++){
            if(cnum_y_array[j] == 0){
                min = 0;
                min_index = j;
            }
        }     
        if(min != 0){
            min = 100;
            for(j = mid -REAL_SEG_COMPARE_OFFSET; j < mid + REAL_SEG_COMPARE_OFFSET; j++){
                if(cnum_y_array[j] < min){
                    min = cnum_y_array[j];
                    min_index = j;
                }
            }
        }
        cline_end[0] = min_index;
        cline_start[1] = cline_end[0] + 1;        

        mid = (cline_start[1] + cline_end[2]) / 2;
        min_index = 0;
        min = 100;
        for(j = mid - REAL_SEG_ZERO_OFFSET; j < mid + REAL_SEG_ZERO_OFFSET; j++){
            if(cnum_y_array[j] == 0){
                min = 0;
                min_index = j;
            }
        }             
        if(min != 0){
            min = 100;
            for(j = mid -REAL_SEG_COMPARE_OFFSET; j < mid + REAL_SEG_COMPARE_OFFSET; j++){
                if(cnum_y_array[j] < min){
                    min = cnum_y_array[j];
                    min_index = j;
                }
            }
        }
        cline_end[1] = min_index;
        cline_start[2] = cline_end[1] + 1;          
        break;
    case 4:
        for(i = CHAR_START_X; i < img_width; i++){
            if(cnum_y_array[i] > REAL_SEG_CHAR_Y){
                if(i - last_index == 1){
                    seq_num++;
                }else{
                    seq_num = 1;
                }
                last_index = i;
                if(seq_num == REAL_SEG_CHAR_X){
                    cline[0] = i - REAL_SEG_CHAR_X + 1;
                    cline_start[0] = cline[0] - REAL_SEG_X_OFFSET;
                    //cline_end[0] = cline[0] + REAL_SEG_X_OFFSET * 2;
                    last_index = 0;
                    seq_num = 0;
                    break;
                }
            }
        }        

        for(i = img_width - 1; i > CHAR_START_X; i--){
            if(cnum_y_array[i] > REAL_SEG_CHAR_Y){
                if(last_index- i == 1){
                    seq_num++;
                }else{
                    seq_num = 1;
                }
                last_index = i;
                if(seq_num == REAL_SEG_CHAR_X){
                    cline[3] = i + REAL_SEG_CHAR_X - 1;
                    cline_end[3] = cline[3] + REAL_SEG_X_OFFSET;
                    //cline_end[0] = cline[0] + REAL_SEG_X_OFFSET * 2;
                    last_index = 0;
                    seq_num = 0;
                    break;
                }
            }
        }                   
        mid = (cline_end[3] - cline_start[0]) / 4 + cline_start[0];
        min_index = 0;
        min = 100;
        for(j = mid - REAL_SEG_ZERO_OFFSET; j < mid + REAL_SEG_ZERO_OFFSET; j++){
            if(cnum_y_array[j] == 0){
                min = 0;
                min_index = j;
            }
        }     
        if(min != 0){
            min = 100;
            for(j = mid -REAL_SEG_COMPARE_OFFSET; j < mid + REAL_SEG_COMPARE_OFFSET; j++){
                if(cnum_y_array[j] < min){
                    min = cnum_y_array[j];
                    min_index = j;
                }
            }
        }
        cline_end[0] = min_index;
        cline_start[1] = cline_end[0] + 1;        

        mid = (cline_end[3] - cline_start[1]) / 3 + cline_start[1];
        min_index = 0;
        min = 100;
        for(j = mid - REAL_SEG_ZERO_OFFSET; j < mid + REAL_SEG_ZERO_OFFSET; j++){
            if(cnum_y_array[j] == 0){
                min = 0;
                min_index = j;
            }
        }             
        if(min != 0){
            min = 100;
            for(j = mid -REAL_SEG_COMPARE_OFFSET; j < mid + REAL_SEG_COMPARE_OFFSET; j++){
                if(cnum_y_array[j] < min){
                    min = cnum_y_array[j];
                    min_index = j;
                }
            }
        }
        cline_end[1] = min_index;
        cline_start[2] = cline_end[1] + 1; 

        mid = (cline_start[2] + cline_end[3]) / 2;
        min_index = 0;
        min = 100;
        for(j = mid - REAL_SEG_ZERO_OFFSET; j < mid + REAL_SEG_ZERO_OFFSET; j++){
            if(cnum_y_array[j] == 0){
                min = 0;
                min_index = j;
            }
        }             
        if(min != 0){
            min = 100;
            for(j = mid -REAL_SEG_COMPARE_OFFSET; j < mid + REAL_SEG_COMPARE_OFFSET; j++){
                if(cnum_y_array[j] < min){
                    min = cnum_y_array[j];
                    min_index = j;
                }
            }
        }
        cline_end[2] = min_index;
        cline_start[3] = cline_end[2] + 1;       
        
        break;
    default:
        cout << "ERROR SEGMENT NUM: " << seg_num << endl;
    }

    for(i = 0; i < seg_num; i++){
        string ofilename = ifilename;
        stringstream out_file;
        if(fid < 25){
            out_file << fid + i;
        }else if(fid < 50){
            out_file << 25 + (fid - 25) * 2 + i;
        }else if(fid < 75){
            out_file << 75 + (fid - 50) * 3 + i;
        }else{
            out_file << 150 + (fid - 75) * 4 + i;
        }
        ofilename += out_file.str();
        ofilename += ".input";
        ofstream os;
        os.open(ofilename.c_str());
        for(j = 0; j < img_width / 4 * img_height; j++){
            output_buf[j] = 0;
        }                
        int m = 0, n = 0, l = 0, k = 0;

        if((cline_end[i] - cline_start[i]) > (img_width / 4 - 1))
            cline_end[i] = cline_start[i] + (img_width / 4 - 1);
        for(m = 0, l = 0; m < img_height; m++, l++){
            for(n = 0, k = cline_start[i]; k < cline_end[i]; n++, k++){
                output_buf[m  * (img_width / 4) + n] = image_buf[l * img_width + k];
            }
        }

        for(m = 0; m < img_height; m++){
            for(n = 0; n < (img_width / 4); n++){
                os << output_buf[m  * (img_width / 4) + n] << " ";
            }
            os << endl;
        }
        os.close();
    }
    
}

#define NUM_INFER_IMAGE 100
int main(int argc, char *argv[]){
    if(argc != 2){
        cout << "Invalid input!" <<endl<< "Usage: ./pre_image <file_path>" <<endl;
        return -1;
    }
    int wrong_num = 0;

    for(int fid = 0; fid < NUM_INFER_IMAGE; fid++){
        stringstream ss;
        string ifilename = argv[1];            
        ifilename += "/";
        ss << fid;
        ifilename += ss.str();
        ifilename += ".tga";       
        //binarization, compression, and removing nosie
        binarization_translation(ifilename);

        //we don't call the segment function in current version 
        //as we know the char number of each captcha in CUDA 2012 required test
        //(1 char * 25, 2 char * 25, 3 char * 25, 4 char * 25)
        //int total_segs = segment(compress_output);    

        ifilename = argv[1];
        ifilename += "/";
        //(1 char * 25, 2 char * 25, 3 char * 25, 4 char * 25)        
        if(fid < 25){
            real_segment(fid, compress_output, 1, ifilename);
        }else if(fid < 50){
            real_segment(fid, compress_output, 2, ifilename);
        }else if (fid < 75){
            real_segment(fid, compress_output, 3, ifilename);
        }else{
            real_segment(fid, compress_output, 4, ifilename);
        }
        

        //output_segment_images(compress_output, ifilename, 1);
        
        delete compress_output;
    }
}

