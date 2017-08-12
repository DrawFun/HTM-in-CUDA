#ifndef HTM_HEADER
#define HTM_HEADER
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <sdkHelper.h>  // helper for shared that are common to CUDA SDK samples
#include <shrQATest.h>  // This is for automated testing output (--qatest)
#include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, HTM head files
#include "node.h"


// includes, kernels
#include "htm_kernel.cu"

extern "C"

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }

    if (devID < 0)
       devID = 0;
        
    if (devID > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(-1);                                                  
    }
    
    checkCudaErrors( cudaSetDevice(devID) );
    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

    return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_compute_perf   = 0, max_perf_device   = 0;
    int device_count       = 0, best_SM_arch      = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount( &device_count );
    
    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, deviceProp.major);
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
    while( current_device < device_count )
    {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        {
            sm_per_multiproc = 1;
        }
        else
        {
            sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        }
        
        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        
    if( compute_perf  > max_compute_perf )
    {
            // If we find GPU with SM major > 2, search only these
            if ( best_SM_arch > 2 )
            {
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch)
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                 }
            }
            else
            {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
             }
        }
        ++current_device;
    }
    return max_perf_device;
}


// Initialization code to find the best CUDA Device
int findCudaDevice(int argc, const char **argv)
{
    cudaDeviceProp deviceProp;
    int devID = 0;
    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");
        if (devID < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(-1);
        }
        else
        {
            devID = gpuDeviceInit(devID);
            if (devID < 0)
            {
                printf("exiting...\n");
                shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                exit(-1);
            }
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors( cudaSetDevice( devID ) );
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
    return devID;
}

void deviceInit( int argc, char** argv){
    int devID = findCudaDevice(argc, (const char**)argv);
}

void deviceExit(int argc, char** argv){
    cudaDeviceReset();
}
// end of CUDA Helper Functions

/*
** START of main data declaration
*/
/* array to store all to-learn or to-inference images on cpu */
int* images;
/* images on gpu */
int* images_gpu;
/* start up infomation, length of an image that can be shifted vertically */
int* image_x_offset;
/* start up infomation, length of an image that can be shifted horizontally */
int* image_y_offset;
/* images_x_offset on gpu */
int* image_x_offset_gpu;
/* images_y_offset on gpu */
int* image_y_offset_gpu;

/* max number of QCs of each node of HTM level one */
int** max_qc_profile_lv1;
/* max number of QCs of each node of HTM level two */
int** max_qc_profile_lv2;

/* all HTM level one nodes on cpu */
node_t* cpu_level_one_nodes;
/* all HTM level one nodes on gpu */
node_t* gpu_level_one_nodes;

/* all HTM level two nodes on cpu */
node_t* cpu_level_two_nodes;
/* all HTM level two nodes on gpu */
node_t* gpu_level_two_nodes;

/* HTM level three node on cpu, only one node since top level */
node_t* cpu_level_three_nodes;
/* HTM level three node on gpu, only one node since top level */
node_t* gpu_level_three_nodes;

/* index pointing to the start of each image used by level one */
int* gpu_image_from_index;
/* index pointing to the start of each image used by level two */
int* gpu_image_from_index_lv2;

/* stored lv1 inferenced result on gpu for optimization */
float* gpu_inference_result_lv1;
/* index pointing to start of the inferenced result of each lv1 nodes */
int* gpu_infer_from_index_lv1;
/* index pointing to the TG which has the largest prob refering to current input for lv2 nodes */
int* gpu_inference_max_prob_tg_lv1;

/* stored lv2 inferenced result on gpu for optimization */
float* gpu_inference_result_lv2;
/* index pointing to start of the inferenced result of each lv2 nodes */
int* gpu_infer_from_index_lv2;
/* index pointing to the TG which has the largest prob refering to current input for lv2 nodes */
int* gpu_inference_max_prob_tg_lv2;
/* result whether the current QC is considered the same with existing QC for lv1 ndoes */
int* gpu_qc_same_result_lv1;
/* result whether the current QC is considered the same with existing QC for lv2 ndoes */
int* gpu_qc_same_result_lv2;
/* result whether the current QC is considered the same with existing QC for lv3 ndoes */
int* gpu_qc_same_result_lv3;
/*
** END of main data declaration
*/

/*
** START of main function declaration used in cpu
*/
/* initialize level one nodes */
void init_level_one();

/* initialize level one nodes from learned files */
void init_level_one_from_file();

/* initialize level one inference results*/
void init_inference_result_lv1();

/* initialize level two nodes */
void init_level_two();

/* initialize level two nodes from learned files */
void init_level_two_from_file();

/* init inference result of level two, used for optimization on gpu */
void init_inference_result_lv2(int run_mode);

/* initialize level three nodes */
void init_level_three();

/* initialize level three nodes from learned files */
void init_level_three_from_file();

/* make TG for the nodes of the specified level */
void make_TG_level(int level);

/* start up the program */
void start_up(int run_mode);

/*init images */
void init_images(char* file_path);

/* init images in inference mode */
void init_images_infer_mode(char* file_path, int index);

/* free space of level one nodes on gpu */
void free_level_one_gpu();

/* free space of time adjacent matrix of each level two nodes */
void free_level_two_matrix_gpu();

/* cp nodes data from gpu to cpu to make temporal group */
void cpFromDeviceToHost(int level);

/* cp nodes data from cpu to gpu to do inference work */
void cpFromHostToDevice(int level);
/*
** END of main function declaration used in cpu
*/

void start_up(int run_mode){
    /* get the profiled QC info for optimization on gpu */
    max_qc_profile_lv1 = (int**)malloc(LEVEL1_WIDTH * sizeof(int*));
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        max_qc_profile_lv1[i] = (int*)malloc(LEVEL1_WIDTH * sizeof(int));
    }
    FILE* file = fopen("start_up_info/level_one_qc", "r");
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        for(int j = 0; j < LEVEL1_WIDTH; j++){
            fscanf(file, "%d\n", &(max_qc_profile_lv1[i][j]));
        }
    }
    fclose(file);
    
    max_qc_profile_lv2 = (int**)malloc(LEVEL2_WIDTH * sizeof(int*));
    for(int i = 0; i < LEVEL2_WIDTH; i++){
        max_qc_profile_lv2[i] = (int*)malloc(LEVEL2_WIDTH * sizeof(int));
    }
    file = fopen("start_up_info/level_two_qc", "r");
    for(int i = 0; i < LEVEL2_WIDTH; i++){
        for(int j = 0; j < LEVEL2_WIDTH; j++){
            fscanf(file, "%d\n", &(max_qc_profile_lv2[i][j]));
        }
    }
    fclose(file);
    
    /* get the length of shifts both vertically and horizontally of each image to learn */
    image_x_offset = (int*)malloc(num_images * sizeof(int));
    image_y_offset = (int*)malloc(num_images * sizeof(int));
    if(run_mode == TRAINING_MODE){
        file = fopen("start_up_info/image_offset", "r");
        for(int i = 0; i < num_images; i++){
            fscanf(file, "%d", &(image_x_offset[i]));
            fscanf(file, "%d", &(image_y_offset[i]));
        }
        fclose(file);
    }else{
        image_x_offset[0] = 1;
        /* for eye movement inference, all images move NUM_STEP_EYE_MOVE steps */
        image_y_offset[0] = NUM_STEP_EYE_MOVE;
    }
    /* malloc and initizlize shifts info on gpu */
    checkCudaErrors(cudaMalloc((void**)&image_x_offset_gpu, num_images *
                sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&image_y_offset_gpu, num_images *
                sizeof(int)));
    checkCudaErrors(cudaMemcpy(image_x_offset_gpu, image_x_offset, num_images *
                sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(image_y_offset_gpu, image_y_offset, num_images *
                sizeof(int), cudaMemcpyHostToDevice));

    /* malloc and initialize image from index on gpu */
    int* cpu_image_from_index = (int*)malloc(num_images * sizeof(int));
    memset(cpu_image_from_index, 0, num_images * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&gpu_image_from_index, num_images *
                sizeof(int)));
    int from_index = 0;
    for(int i = 0; i < num_images; i++){
        cpu_image_from_index[i] = from_index;
        from_index += image_x_offset[i] * image_y_offset[i] * 2;
    }
    checkCudaErrors(cudaMemcpy(gpu_image_from_index,
                cpu_image_from_index, num_images * sizeof(int),
                cudaMemcpyHostToDevice));
    free(cpu_image_from_index);

    int* cpu_image_from_index_lv2 = (int*)malloc(num_images * sizeof(int));
    memset(cpu_image_from_index_lv2, 0, num_images * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&gpu_image_from_index_lv2, num_images *
                sizeof(int)));
    from_index = 0;
    for(int i = 0; i < num_images; i++){
        cpu_image_from_index_lv2[i] = from_index;
        from_index += image_x_offset[i] * image_y_offset[i] * 2 *
            LEVEL2_WIDTH * LEVEL2_WIDTH;
    }
    checkCudaErrors(cudaMemcpy(gpu_image_from_index_lv2,
                cpu_image_from_index_lv2, num_images * sizeof(int),
                cudaMemcpyHostToDevice));
    free(cpu_image_from_index_lv2);

    /* malloc and initialize QC same info on gpu */
    int* cpu_qc_same_result_lv1 = (int*)malloc(LEVEL1_WIDTH * LEVEL1_WIDTH *
            NUM_THREADS_LEARN_LV1 * sizeof(int));
    memset(cpu_qc_same_result_lv1, 0, LEVEL1_WIDTH * LEVEL1_WIDTH *
            NUM_THREADS_LEARN_LV1 * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&gpu_qc_same_result_lv1, LEVEL1_WIDTH *
                LEVEL1_WIDTH * NUM_THREADS_LEARN_LV1 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_qc_same_result_lv1, cpu_qc_same_result_lv1,
                LEVEL1_WIDTH * LEVEL1_WIDTH * NUM_THREADS_LEARN_LV1 *
                sizeof(int), cudaMemcpyHostToDevice));
    free(cpu_qc_same_result_lv1);

    int* cpu_qc_same_result_lv2 = (int*)malloc(LEVEL2_WIDTH * LEVEL2_WIDTH *
            NUM_THREADS_LEARN_LV2 * sizeof(int));
    memset(cpu_qc_same_result_lv2, 0, LEVEL2_WIDTH * LEVEL2_WIDTH *
            NUM_THREADS_LEARN_LV2 * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&gpu_qc_same_result_lv2, LEVEL2_WIDTH *
                LEVEL2_WIDTH * NUM_THREADS_LEARN_LV2 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_qc_same_result_lv2, cpu_qc_same_result_lv2,
                LEVEL2_WIDTH * LEVEL2_WIDTH * NUM_THREADS_LEARN_LV2 *
                sizeof(int), cudaMemcpyHostToDevice));
    free(cpu_qc_same_result_lv2);

    int* cpu_qc_same_result_lv3 = (int*)malloc(NUM_THREADS_LEARN_LV3 * sizeof(int));
    memset(cpu_qc_same_result_lv3, 0, NUM_THREADS_LEARN_LV3 * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&gpu_qc_same_result_lv3,
                NUM_THREADS_LEARN_LV3 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_qc_same_result_lv3, cpu_qc_same_result_lv3,
                NUM_THREADS_LEARN_LV3 * sizeof(int), cudaMemcpyHostToDevice));
    free(cpu_qc_same_result_lv3);
}

/* init images when run in TRAINING MODE */
void init_images(char* file_path){
    images = (int*)malloc(ST_IMAGE_WIDTH * ST_IMAGE_WIDTH * num_images *
            sizeof(int));
    memset(images, 0, ST_IMAGE_WIDTH * ST_IMAGE_WIDTH * num_images *
            sizeof(int));

    FILE* input_file;
    char* file_name = (char*)malloc(30);
    for(int i = 0; i < num_images; i++){
        sprintf(file_name, "%s/%d.input", file_path, i);
        input_file = fopen(file_name, "r"); 
        for(int k = 0; k < IMAGE_WIDTH; k++){
            for(int l = 0; l < IMAGE_WIDTH; l++){
                fscanf(input_file, "%d", &(images[i * ST_IMAGE_WIDTH *
                            ST_IMAGE_WIDTH + (IMAGE_WIDTH + k) * ST_IMAGE_WIDTH + (IMAGE_WIDTH + l)]));
            }
        }
        fclose(input_file);
    }

    checkCudaErrors(cudaMalloc((void**)&images_gpu, ST_IMAGE_WIDTH *
                ST_IMAGE_WIDTH * num_images * sizeof(int)));
    checkCudaErrors(cudaMemcpy(images_gpu, images, ST_IMAGE_WIDTH *
                ST_IMAGE_WIDTH * num_images * sizeof(int),
                cudaMemcpyHostToDevice));
}

/* init image when run in INFERENCE mod */
void init_images_infer_mode(char* file_path, int index){
    FILE* input_file;
    char* file_name = (char*)malloc(100);
    sprintf(file_name, "%s/%d.input", file_path , index);
    input_file = fopen(file_name, "r"); 
    for(int k = 0; k < IMAGE_WIDTH; k++){
        for(int l = 0; l < IMAGE_WIDTH; l++){
            fscanf(input_file, "%d", &(images[(IMAGE_WIDTH + k) * ST_IMAGE_WIDTH + (IMAGE_WIDTH + l)]));
        }
    }
    fclose(input_file);

    checkCudaErrors(cudaMemcpy(images_gpu, images, ST_IMAGE_WIDTH *
                ST_IMAGE_WIDTH * num_images * sizeof(int),
                cudaMemcpyHostToDevice));
} 

/* init nodes in level one */
void init_level_one(){
    cpu_level_one_nodes = (node_t*)malloc(LEVEL1_WIDTH * LEVEL1_WIDTH * sizeof(node_t));
    checkCudaErrors(cudaMalloc((void**)&gpu_level_one_nodes,(LEVEL1_WIDTH * LEVEL1_WIDTH) * sizeof(node_t)));
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        for(int j = 0; j < LEVEL1_WIDTH; j++){
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].level = 1;
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].row = i;
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].column = j;
            int max_qc = max_qc_profile_lv1[i][j];
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].num_max_QC = max_qc; 
            //init QC
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].QC = (int*)malloc(max_qc *
                    NUM_CHILD_LV1 * sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&(cpu_level_one_nodes[i *
                        LEVEL1_WIDTH + j].QC_gpu),(max_qc * NUM_CHILD_LV1) *
                        sizeof(int)));
            for(int k = 0; k < max_qc; k++){
                for(int l = 0; l < NUM_CHILD_LV1; l++){
                    cpu_level_one_nodes[i * LEVEL1_WIDTH + j].QC[k *
                        NUM_CHILD_LV1 + l] = -1;
                }
            }
            checkCudaErrors(cudaMemcpy(cpu_level_one_nodes[i * LEVEL1_WIDTH +
                        j].QC_gpu, cpu_level_one_nodes[i * LEVEL1_WIDTH + j].QC,
                        sizeof(int) * max_qc * NUM_CHILD_LV1,
                        cudaMemcpyHostToDevice));

            /* init time adjacent matrix */
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].time_adj_matrix =
                (int*)malloc(max_qc * max_qc * sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&cpu_level_one_nodes[i *
                        LEVEL1_WIDTH + j].time_adj_matrix_gpu, max_qc * max_qc *
                        sizeof(int)));
            for(int k = 0; k < max_qc; k++){
                for(int l = 0; l < max_qc; l++){
                    cpu_level_one_nodes[i * LEVEL1_WIDTH + j].time_adj_matrix[k
                        * max_qc + l] = 0;
                }
            }
            checkCudaErrors(cudaMemcpy(cpu_level_one_nodes[i * LEVEL1_WIDTH +
                        j].time_adj_matrix_gpu, cpu_level_one_nodes[i *
                        LEVEL1_WIDTH + j].time_adj_matrix, sizeof(int) * max_qc
                        * max_qc, cudaMemcpyHostToDevice));
            
            /* init group id array */
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].group_id = (int*)malloc(max_qc *
                    sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&cpu_level_one_nodes[i * LEVEL1_WIDTH +
                        j].group_id_gpu, sizeof(int) * max_qc));
            for(int k = 0; k < max_qc; k++){
                cpu_level_one_nodes[i * LEVEL1_WIDTH + j].group_id[k] = -1;
            }
            checkCudaErrors(cudaMemcpy(cpu_level_one_nodes[i * LEVEL1_WIDTH +
                        j].group_id_gpu, cpu_level_one_nodes[i *
                        LEVEL1_WIDTH + j].group_id, sizeof(int) * max_qc, cudaMemcpyHostToDevice));

            /* init prob_QC array */
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].prob_QC = (float*)malloc(max_qc);
            checkCudaErrors(cudaMalloc((void**)&cpu_level_one_nodes[i * LEVEL1_WIDTH +
                        j].prob_QC_gpu, sizeof(float) * max_qc));

            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].num_child = NUM_CHILD_LV1;
            /*init num_child_group array */
            for(int k = 0; k < NUM_CHILD_LV1; k++){
                cpu_level_one_nodes[i * LEVEL1_WIDTH + j].num_child_group[k] = 2;
            }

            /* init input_QC array */
            for(int k = 0; k < MAX_NUM_CHILD; k++){
                cpu_level_one_nodes[i * LEVEL1_WIDTH + j].input_QC[k] = -1;
            }

            /* init counters */
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].num_QC = 0;
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].last_QC = -1;
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].closest_index = -1;
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].num_TG = 0;
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].inference_groupid = -1;
        }
    }

    /* init nodes on gpu */
    checkCudaErrors(cudaMemcpy(gpu_level_one_nodes, cpu_level_one_nodes,
        sizeof(node_t) * LEVEL1_WIDTH * LEVEL1_WIDTH, cudaMemcpyHostToDevice));
}

/* init level one nodes from learned file */
void init_level_one_from_file(){
    FILE* input_file;
    char* file_name = (char*)malloc(20);
    cpu_level_one_nodes = (node_t*)malloc(LEVEL1_WIDTH * LEVEL1_WIDTH * sizeof(node_t));
    checkCudaErrors(cudaMalloc((void**)&gpu_level_one_nodes,(LEVEL1_WIDTH * LEVEL1_WIDTH) * sizeof(node_t)));
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        for(int j = 0; j < LEVEL1_WIDTH; j++){
            node_t* cur_node = &(cpu_level_one_nodes[i * LEVEL1_WIDTH + j]);
            /* find the matching learned file to init the current node */
            sprintf(file_name, "learned/%d_%d_%d", 1, i, j);
            input_file = fopen(file_name, "r"); 

            cur_node->level = 1;
            cur_node->row = i;
            cur_node->column = j;

            /* init num_child_group array */
            fscanf(input_file, "%d\n", &(cur_node->num_child));
            for(int k = 0; k < cur_node->num_child; k++){
                fscanf(input_file, "%d\n", &(cur_node->num_child_group[k]));
            }

            //init QC
            int max_qc;
            fscanf(input_file, "%d\n", &max_qc);
            cur_node->num_max_QC = max_qc; 
            cur_node->QC = (int*)malloc(max_qc * NUM_CHILD_LV1 * sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&(cur_node->QC_gpu),(max_qc * NUM_CHILD_LV1) *
                        sizeof(int)));
            for(int k = 0; k < max_qc; k++){
                for(int l = 0; l < NUM_CHILD_LV1; l++){
                    fscanf(input_file, "%d", &(cur_node->QC[k * NUM_CHILD_LV1 + l]));
                }
            }
            checkCudaErrors(cudaMemcpy(cur_node->QC_gpu, cur_node->QC,
                        sizeof(int) * max_qc * NUM_CHILD_LV1,
                        cudaMemcpyHostToDevice));

            fscanf(input_file, "%d\n", &(cur_node->num_TG));

            /* init group_id array */
            cur_node->group_id = (int*)malloc(max_qc * sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&(cur_node->group_id_gpu), sizeof(int) * max_qc));
            for(int k = 0; k < max_qc; k++){
                fscanf(input_file, "%d ", &(cur_node->group_id[k]));
            }
            checkCudaErrors(cudaMemcpy(cur_node->group_id_gpu, cur_node->group_id, sizeof(int) * max_qc,
                        cudaMemcpyHostToDevice));

            cur_node->prob_QC = (float*)malloc(max_qc);
            checkCudaErrors(cudaMalloc((void**)&(cur_node->prob_QC_gpu), sizeof(float) * max_qc));

            /* init input_QC array */
            for(int k = 0; k < MAX_NUM_CHILD; k++){
                cur_node->input_QC[k] = -1;
            }

            /* init counters */
            cur_node->num_QC = max_qc;
            cur_node->last_QC = -1;
            cur_node->closest_index = -1;
            cur_node->inference_groupid = -1;
            fclose(input_file);
        }
    }

    /* init nodes on gpu */
    checkCudaErrors(cudaMemcpy(gpu_level_one_nodes, cpu_level_one_nodes,
        sizeof(node_t) * LEVEL1_WIDTH * LEVEL1_WIDTH, cudaMemcpyHostToDevice));
}

/* init level two nodes */
void init_level_two(){
    cpu_level_two_nodes = (node_t*)malloc(LEVEL2_WIDTH * LEVEL2_WIDTH * sizeof(node_t));
    checkCudaErrors(cudaMalloc((void**)&gpu_level_two_nodes,(LEVEL2_WIDTH * LEVEL2_WIDTH) * sizeof(node_t)));
    for(int i = 0; i < LEVEL2_WIDTH; i++){
        for(int j = 0; j < LEVEL2_WIDTH; j++){
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].level = 2;
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].row = i;
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].column = j;
            int max_qc = max_qc_profile_lv2[i][j];
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].num_max_QC = max_qc; 

            //init QC
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].QC = (int*)malloc(max_qc *
                    NUM_CHILD_LV2 * sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&(cpu_level_two_nodes[i *
                        LEVEL2_WIDTH + j].QC_gpu),(max_qc * NUM_CHILD_LV2) *
                        sizeof(int)));
            for(int k = 0; k < max_qc; k++){
                for(int l = 0; l < NUM_CHILD_LV2; l++){
                    cpu_level_two_nodes[i * LEVEL2_WIDTH + j].QC[k *
                        NUM_CHILD_LV2 + l] = -1;
                }
            }
            checkCudaErrors(cudaMemcpy(cpu_level_two_nodes[i * LEVEL2_WIDTH +
                        j].QC_gpu, cpu_level_two_nodes[i * LEVEL2_WIDTH + j].QC,
                        sizeof(int) * max_qc * NUM_CHILD_LV2,
                        cudaMemcpyHostToDevice));

            /* init time adjacent matrix */
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].time_adj_matrix =
                (int*)malloc(max_qc * max_qc * sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&cpu_level_two_nodes[i *
                        LEVEL2_WIDTH + j].time_adj_matrix_gpu, max_qc * max_qc *
                        sizeof(int)));
            for(int k = 0; k < max_qc; k++){
                for(int l = 0; l < max_qc; l++){
                    cpu_level_two_nodes[i * LEVEL2_WIDTH + j].time_adj_matrix[k
                        * max_qc + l] = 0;
                }
            }
            checkCudaErrors(cudaMemcpy(cpu_level_two_nodes[i * LEVEL2_WIDTH +
                        j].time_adj_matrix_gpu, cpu_level_two_nodes[i *
                        LEVEL2_WIDTH + j].time_adj_matrix, sizeof(int) * max_qc
                        * max_qc, cudaMemcpyHostToDevice));
            
            /* init group_id array */
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].group_id = (int*)malloc(max_qc *
                    sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&cpu_level_two_nodes[i * LEVEL2_WIDTH +
                        j].group_id_gpu, sizeof(int) * max_qc));
            for(int k = 0; k < max_qc; k++){
                cpu_level_two_nodes[i * LEVEL2_WIDTH + j].group_id[k] = -1;
            }
            checkCudaErrors(cudaMemcpy(cpu_level_two_nodes[i * LEVEL2_WIDTH +
                        j].group_id_gpu, cpu_level_two_nodes[i *
                        LEVEL2_WIDTH + j].group_id, sizeof(int) * max_qc, cudaMemcpyHostToDevice));

            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].prob_QC = (float*)malloc(max_qc);
            checkCudaErrors(cudaMalloc((void**)&cpu_level_two_nodes[i * LEVEL2_WIDTH +
                        j].prob_QC_gpu, sizeof(float) * max_qc));

            /* init num_child_group array */
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].num_child = NUM_CHILD_LV2;
            int sub_width = LEVEL1_WIDTH / LEVEL2_WIDTH;
            for(int k = 0; k < NUM_CHILD_LV2; k++){
                cpu_level_two_nodes[i * LEVEL2_WIDTH + j].num_child_group[k] 
                    = cpu_level_one_nodes[(i * sub_width+ k / sub_width) * LEVEL1_WIDTH + j * sub_width + k % sub_width].num_TG;
            }

            /* init input_QC array */
            for(int k = 0; k < MAX_NUM_CHILD; k++){
                cpu_level_two_nodes[i * LEVEL2_WIDTH + j].input_QC[k] = -1;
            }

            /* init counters */
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].num_QC = 0;
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].last_QC = -1;
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].closest_index = -1;
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].num_TG = 0;
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].inference_groupid = -1;
        }
    }

    /* init nodes on gpu */
    checkCudaErrors(cudaMemcpy(gpu_level_two_nodes, cpu_level_two_nodes,
        sizeof(node_t) * LEVEL2_WIDTH * LEVEL2_WIDTH, cudaMemcpyHostToDevice));
}

/* init level two nodes from learned file */
void init_level_two_from_file(){
    FILE* input_file;
    char* file_name = (char*)malloc(30);
    cpu_level_two_nodes = (node_t*)malloc(LEVEL2_WIDTH * LEVEL2_WIDTH * sizeof(node_t));
    checkCudaErrors(cudaMalloc((void**)&gpu_level_two_nodes,(LEVEL2_WIDTH * LEVEL2_WIDTH) * sizeof(node_t)));
    for(int i = 0; i < LEVEL2_WIDTH; i++){
        for(int j = 0; j < LEVEL2_WIDTH; j++){
            /* find the learned file matching the current node */
            node_t* cur_node = &(cpu_level_two_nodes[i * LEVEL2_WIDTH + j]);
            sprintf(file_name, "learned/%d_%d_%d", 2, i, j);
            input_file = fopen(file_name, "r"); 

            cur_node->level = 2;
            cur_node->row = i;
            cur_node->column = j;

            /* init num_child_group array */
            fscanf(input_file, "%d\n", &(cur_node->num_child));
            for(int k = 0; k < NUM_CHILD_LV2; k++){
                fscanf(input_file, "%d\n", &(cur_node->num_child_group[k]));
            }

            /* init QC */
            int max_qc;
            fscanf(input_file, "%d\n", &max_qc);
            cur_node->num_max_QC = max_qc; 
            cur_node->QC = (int*)malloc(max_qc * NUM_CHILD_LV2 * sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&(cur_node->QC_gpu),(max_qc * NUM_CHILD_LV2) *
                        sizeof(int)));
            for(int k = 0; k < max_qc; k++){
                for(int l = 0; l < NUM_CHILD_LV2; l++){
                    fscanf(input_file, "%d\n", &(cur_node->QC[k * NUM_CHILD_LV2 + l]));
                }
            }
            checkCudaErrors(cudaMemcpy(cur_node->QC_gpu, cur_node->QC,
                        sizeof(int) * max_qc * NUM_CHILD_LV2, cudaMemcpyHostToDevice));

            /* init group_id array */
            fscanf(input_file, "%d\n", &(cur_node->num_TG));
            cur_node->group_id = (int*)malloc(max_qc * sizeof(int));
            checkCudaErrors(cudaMalloc((void**)&(cur_node->group_id_gpu), sizeof(int) * max_qc));
            for(int k = 0; k < max_qc; k++){
                fscanf(input_file, "%d ", &(cur_node->group_id[k]));
            }
            checkCudaErrors(cudaMemcpy(cur_node->group_id_gpu, cur_node->group_id, sizeof(int) * max_qc,
                        cudaMemcpyHostToDevice));

            cur_node->prob_QC = (float*)malloc(max_qc);
            checkCudaErrors(cudaMalloc((void**)&(cur_node->prob_QC_gpu), sizeof(float) * max_qc));

            /* init input_QC */
            for(int k = 0; k < MAX_NUM_CHILD; k++){
                cpu_level_two_nodes[i * LEVEL2_WIDTH + j].input_QC[k] = -1;
            }

            /* init counters */
            cur_node->num_QC = max_qc;
            cur_node->last_QC = -1;
            cur_node->closest_index = -1;
            cur_node->inference_groupid = -1;
            fclose(input_file);
        }
    }

    /* init nodes on gpu */
    checkCudaErrors(cudaMemcpy(gpu_level_two_nodes, cpu_level_two_nodes,
        sizeof(node_t) * LEVEL2_WIDTH * LEVEL2_WIDTH, cudaMemcpyHostToDevice));
}

/* init node of level three(top level) */
void init_level_three(){
    cpu_level_three_nodes = (node_t*)malloc(1 * sizeof(node_t));
    checkCudaErrors(cudaMalloc((void**)&gpu_level_three_nodes, 1 * sizeof(node_t)));
    cpu_level_three_nodes[0].level = 3;
    cpu_level_three_nodes[0].row = 0;
    cpu_level_three_nodes[0].column = 0;
    int max_qc = MAX_NUM_QC_LEVEL3;
    cpu_level_three_nodes[0].num_max_QC = max_qc; 
    
    /* init QC */
    cpu_level_three_nodes[0].QC = (int*)malloc(max_qc *
            NUM_CHILD_LV3 * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&(cpu_level_three_nodes[0].QC_gpu), max_qc * NUM_CHILD_LV3 *
                sizeof(int)));
    for(int k = 0; k < max_qc; k++){
        for(int l = 0; l < NUM_CHILD_LV3; l++){
            cpu_level_three_nodes[0].QC[k *
                NUM_CHILD_LV3 + l] = -1;
        }
    }
    checkCudaErrors(cudaMemcpy(cpu_level_three_nodes[0].QC_gpu, cpu_level_three_nodes[0].QC,
                sizeof(int) * max_qc * NUM_CHILD_LV3,
                cudaMemcpyHostToDevice));

    /* init group_id array */
    cpu_level_three_nodes[0].group_id = (int*)malloc(max_qc *
            sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&cpu_level_three_nodes[0].group_id_gpu, sizeof(int) * max_qc));
    for(int k = 0; k < max_qc; k++){
        cpu_level_three_nodes[0].group_id[k] = -1;
    }
    checkCudaErrors(cudaMemcpy(cpu_level_three_nodes[0].group_id_gpu, cpu_level_three_nodes[0].group_id, 
                sizeof(int) * max_qc, cudaMemcpyHostToDevice));

    cpu_level_three_nodes[0].prob_QC = (float*)malloc(max_qc);
    checkCudaErrors(cudaMalloc((void**)&cpu_level_three_nodes[0].prob_QC_gpu, sizeof(float) * max_qc));

    /* init num_child_group array */
    cpu_level_three_nodes[0].num_child = NUM_CHILD_LV3;
    for(int k = 0; k < NUM_CHILD_LV3; k++){
        cpu_level_three_nodes[0].num_child_group[k] 
            = cpu_level_two_nodes[k].num_TG;
    }

    /* init input_QC array */
    for(int k = 0; k < MAX_NUM_CHILD; k++){
        cpu_level_three_nodes[0].input_QC[k] = -1;
    }

    /* init counters */
    cpu_level_three_nodes[0].num_QC = 0;
    cpu_level_three_nodes[0].last_QC = -1;
    cpu_level_three_nodes[0].closest_index = -1;
    cpu_level_three_nodes[0].num_TG = 0;
    cpu_level_three_nodes[0].inference_groupid = -1;

    /* init level three node on gpu */
    checkCudaErrors(cudaMemcpy(gpu_level_three_nodes, cpu_level_three_nodes,
        sizeof(node_t) * 1, cudaMemcpyHostToDevice));
}

/* init node of level three(top level) from learned file */
void init_level_three_from_file(){
    FILE* input_file;
    char* file_name = (char*)malloc(1);
    sprintf(file_name, "learned/3_0_0");
    input_file = fopen(file_name, "r");
    
    cpu_level_three_nodes = (node_t*)malloc(1 * sizeof(node_t));
    checkCudaErrors(cudaMalloc((void**)&gpu_level_three_nodes, 1 * sizeof(node_t)));

    node_t* cur_node = &(cpu_level_three_nodes[0]);
    cur_node->level = 3;
    cur_node->row = 0;
    cur_node->column = 0;

    /* init num_child_group array */
    fscanf(input_file, "%d\n", &(cur_node->num_child));
    for(int k = 0; k < NUM_CHILD_LV3; k++){
        fscanf(input_file, "%d\n", &(cur_node->num_child_group[k]));
    }

    /* init QC */
    int max_qc;
    fscanf(input_file, "%d\n", &max_qc);
    cur_node->num_max_QC = max_qc; 
    cur_node->QC = (int*)malloc(max_qc * NUM_CHILD_LV3 * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&(cur_node->QC_gpu), max_qc * NUM_CHILD_LV3 *
                sizeof(int)));
    for(int k = 0; k < max_qc; k++){
        for(int l = 0; l < NUM_CHILD_LV3; l++){
            fscanf(input_file, "%d", &(cur_node->QC[k * NUM_CHILD_LV3 + l])); 
        }
    }
    checkCudaErrors(cudaMemcpy(cur_node->QC_gpu, cur_node->QC,
                sizeof(int) * max_qc * NUM_CHILD_LV3,
                cudaMemcpyHostToDevice));

    /* init group_id array */
    fscanf(input_file, "%d\n", &(cur_node->num_TG));
    cur_node->group_id = (int*)malloc(max_qc *
            sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&(cur_node->group_id_gpu), sizeof(int) * max_qc));
    for(int k = 0; k < max_qc; k++){
        fscanf(input_file, "%d", &(cur_node->group_id[k]));
    }
    checkCudaErrors(cudaMemcpy(cur_node->group_id_gpu, cur_node->group_id, sizeof(int) * max_qc,
                cudaMemcpyHostToDevice));

    cur_node->prob_QC = (float*)malloc(max_qc);
    checkCudaErrors(cudaMalloc((void**)&(cur_node->prob_QC_gpu), sizeof(float) * max_qc));

    /* init input_QC array */
    for(int k = 0; k < MAX_NUM_CHILD; k++){
        cpu_level_three_nodes[0].input_QC[k] = -1;
    }

    /* init counters */
    cur_node->num_QC = max_qc;
    cur_node->last_QC = -1;
    cur_node->closest_index = -1;
    cur_node->inference_groupid = -1;

    /* init level three node on gpu */
    checkCudaErrors(cudaMemcpy(gpu_level_three_nodes, cpu_level_three_nodes,
        sizeof(node_t) * 1, cudaMemcpyHostToDevice));
}

/* init inference result of level one, used for optimization on gpu */
void init_inference_result_lv1(){
    /* get the number of all to-learn images */
    int num_position = 0;
    for(int i = 0; i < num_images; i++){
        num_position += (image_x_offset[i] * image_y_offset[i]);
    }
    //double because we learn both horizontally and vertically
    num_position = num_position * 2;

    /* compute the start index of each node pointing to inference result */
    int* cpu_infer_from_index_lv1 = (int*)malloc(LEVEL1_WIDTH * LEVEL1_WIDTH *
            sizeof(int));
    node_t* node;
    int sum_TG = 0;
    for(int i = 0; i < LEVEL1_WIDTH * LEVEL1_WIDTH; i++){
        cpu_infer_from_index_lv1[i] = sum_TG * num_position;

        node = &(cpu_level_one_nodes[i]);
        sum_TG += node->num_TG;
    }
    checkCudaErrors(cudaMalloc((void**)&gpu_infer_from_index_lv1, LEVEL1_WIDTH *
                LEVEL1_WIDTH * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_infer_from_index_lv1,
                cpu_infer_from_index_lv1, LEVEL1_WIDTH * LEVEL1_WIDTH *
                sizeof(int), cudaMemcpyHostToDevice));
    free(cpu_infer_from_index_lv1);

    float* cpu_inference_result_lv1 = (float*)malloc(num_position * sum_TG *
            sizeof(float));
    memset(cpu_inference_result_lv1, 0, num_position * sum_TG * sizeof(float));
    checkCudaErrors(cudaMalloc((void**)&gpu_inference_result_lv1, num_position *
                sum_TG * sizeof(float)));
    checkCudaErrors(cudaMemcpy(gpu_inference_result_lv1,
                cpu_inference_result_lv1, num_position * sum_TG * sizeof(float),
                cudaMemcpyHostToDevice));
    free(cpu_inference_result_lv1);

    /* malloc and init gpu_inference_max_prob_tg_lv1 array */
    int* cpu_inference_max_prob_tg_lv1 = (int*)malloc(num_position *
            LEVEL1_WIDTH * LEVEL1_WIDTH * sizeof(int));
    memset(cpu_inference_max_prob_tg_lv1, 0, num_position * LEVEL1_WIDTH *
            LEVEL1_WIDTH * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&gpu_inference_max_prob_tg_lv1,
                num_position * LEVEL1_WIDTH * LEVEL1_WIDTH * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_inference_max_prob_tg_lv1,
                cpu_inference_max_prob_tg_lv1, num_position * LEVEL1_WIDTH *
                LEVEL1_WIDTH * sizeof(int), cudaMemcpyHostToDevice));
}

/* init inference result of level two, used for optimization on gpu */
void init_inference_result_lv2(int run_mode){
    /* get the number of all to-learn images */
    int num_position = 0;
    for(int i = 0; i < num_images; i++){
        num_position += (image_x_offset[i] * image_y_offset[i]);
    }
    //double because we learn both horizontally and vertically
    num_position = num_position * 2;

    /* compute the start index of each node pointing to inference result */
    int* cpu_infer_from_index_lv2 = (int*)malloc(LEVEL2_WIDTH * LEVEL2_WIDTH *
            sizeof(int));
    node_t* node;
    int sum_TG = 0;
    for(int i = 0; i < LEVEL2_WIDTH * LEVEL2_WIDTH; i++){
        cpu_infer_from_index_lv2[i] = sum_TG * num_position;

        node = &(cpu_level_two_nodes[i]);
        sum_TG += node->num_TG;
    }
    checkCudaErrors(cudaMalloc((void**)&gpu_infer_from_index_lv2, LEVEL2_WIDTH *
                LEVEL2_WIDTH * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_infer_from_index_lv2,
                cpu_infer_from_index_lv2, LEVEL2_WIDTH * LEVEL2_WIDTH *
                sizeof(int), cudaMemcpyHostToDevice));

    if(run_mode == INFERENCE_MODE){
        float* cpu_inference_result_lv2 = (float*)malloc(num_position * sum_TG *
                sizeof(float));
        memset(cpu_inference_result_lv2, 0, num_position * sum_TG * sizeof(float));
        checkCudaErrors(cudaMalloc((void**)&gpu_inference_result_lv2, num_position *
                    sum_TG * sizeof(float)));
        checkCudaErrors(cudaMemcpy(gpu_inference_result_lv2,
                    cpu_inference_result_lv2, num_position * sum_TG * sizeof(float),
                    cudaMemcpyHostToDevice));
        free(cpu_inference_result_lv2);
    }

    /* malloc and init gpu_inference_max_prob_tg_lv1 array */
    int* cpu_inference_max_prob_tg = (int*)malloc(num_position * LEVEL2_WIDTH *
            LEVEL2_WIDTH * sizeof(int));
    memset(cpu_inference_max_prob_tg, 0, num_position * LEVEL2_WIDTH *
            LEVEL2_WIDTH * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&gpu_inference_max_prob_tg_lv2, num_position
                * LEVEL2_WIDTH * LEVEL2_WIDTH * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_inference_max_prob_tg_lv2,
                cpu_inference_max_prob_tg, num_position * LEVEL2_WIDTH *
                LEVEL2_WIDTH * sizeof(int), cudaMemcpyHostToDevice));
}

/* cp nodes data from gpu to cpu to make temporal group */
void cpFromDeviceToHost(int level){
    node_t* target_nodes;
    node_t* target_nodes_gpu;
    int num_nodes;
    switch(level){
        case 1:
            target_nodes = cpu_level_one_nodes;
            target_nodes_gpu = gpu_level_one_nodes;
            num_nodes = LEVEL1_WIDTH * LEVEL1_WIDTH;
            break;
        case 2:
            target_nodes = cpu_level_two_nodes;
            target_nodes_gpu = gpu_level_two_nodes;
            num_nodes = LEVEL2_WIDTH * LEVEL2_WIDTH;
            break;
        case 3:
            target_nodes = cpu_level_three_nodes;
            target_nodes_gpu = gpu_level_three_nodes;
            num_nodes = LEVEL3_WIDTH * LEVEL3_WIDTH;
            break;
        default:
            printf("invalid level!\n");
    }

    checkCudaErrors(cudaMemcpy(target_nodes, target_nodes_gpu, sizeof(node_t) *
                num_nodes, cudaMemcpyDeviceToHost));

    int* qc_ptr;
    int* qc_ptr_gpu;
    int* group_id_ptr;
    int* group_id_ptr_gpu;
    for(int i = 0; i < num_nodes; i++){
        qc_ptr = target_nodes[i].QC;
        qc_ptr_gpu = target_nodes[i].QC_gpu;
        checkCudaErrors(cudaMemcpy(qc_ptr, qc_ptr_gpu, target_nodes[i].num_QC *
                    target_nodes[i].num_child * sizeof(int),
                    cudaMemcpyDeviceToHost));
        group_id_ptr = target_nodes[i].group_id;
        group_id_ptr_gpu = target_nodes[i].group_id_gpu;
        checkCudaErrors(cudaMemcpy(group_id_ptr, group_id_ptr_gpu,
                    target_nodes[i].num_QC * sizeof(int),
                    cudaMemcpyDeviceToHost));
    }

    /* node in level three has no time adjacent matrix since it is supervized
     * and need NOT make TG from the matrix*/
    if(level != 3){
        int* matrix_ptr;
        int* matrix_ptr_gpu;
        int num_matrix_elem;
        for(int i = 0; i < num_nodes; i++){
            matrix_ptr = target_nodes[i].time_adj_matrix;
            matrix_ptr_gpu = target_nodes[i].time_adj_matrix_gpu;
            num_matrix_elem = target_nodes[i].num_max_QC *
                target_nodes[i].num_max_QC;
            checkCudaErrors(cudaMemcpy(matrix_ptr, matrix_ptr_gpu, num_matrix_elem *
                        sizeof(int), cudaMemcpyDeviceToHost));
        }
    }
}

/* cp nodes data from cpu to gpu to do inference work */
void cpFromHostToDevice(int level){
    node_t* target_nodes;
    node_t* target_nodes_gpu;
    int num_nodes;
    switch(level){
        case 1:
            target_nodes = cpu_level_one_nodes;
            target_nodes_gpu = gpu_level_one_nodes;
            num_nodes = LEVEL1_WIDTH * LEVEL1_WIDTH;
            break;
        case 2:
            target_nodes = cpu_level_two_nodes;
            target_nodes_gpu = gpu_level_two_nodes;
            num_nodes = LEVEL2_WIDTH * LEVEL2_WIDTH;
            break;
        case 3:
            target_nodes = cpu_level_three_nodes;
            target_nodes_gpu = gpu_level_three_nodes;
            num_nodes = LEVEL3_WIDTH * LEVEL3_WIDTH;
            break;
        default:
            printf("invalid level!\n");
    }

    checkCudaErrors(cudaMemcpy(target_nodes_gpu, target_nodes, sizeof(node_t) *
                num_nodes, cudaMemcpyHostToDevice));

    /* level three node need not group_id since it is learned in supervised mode */
    if(level != 3){
        for(int i = 0; i < num_nodes; i++){
            checkCudaErrors(cudaMemcpy(target_nodes[i].group_id_gpu,
                        target_nodes[i].group_id,
                        target_nodes[i].num_QC * sizeof(int),
                        cudaMemcpyHostToDevice));
        }
    }
}

/* make tempory group for specified level */
void make_TG_level(int level){
    node_t* target_level_nodes;
    int num_nodes;
    switch(level){
        case 1:
            target_level_nodes = cpu_level_one_nodes;
            num_nodes = LEVEL1_WIDTH * LEVEL1_WIDTH;
            break;
        case 2:
            target_level_nodes = cpu_level_two_nodes;
            num_nodes = LEVEL2_WIDTH * LEVEL2_WIDTH;
            break;
        default:
            printf("invalid level\n");
            return;
    }

    for(int i = 0; i < num_nodes; i++){
        make_TG(&(target_level_nodes[i]));
    }
}

/* write learned information into files */
void write_level_knowledge(int level){
    node_t* target_level_nodes;
    int num_level_width;
    switch(level){
        case 1:
            target_level_nodes = cpu_level_one_nodes;
            num_level_width = LEVEL1_WIDTH;
            break;
        case 2:
            target_level_nodes = cpu_level_two_nodes;
            num_level_width = LEVEL2_WIDTH;
            break;
        case 3:
            target_level_nodes = cpu_level_three_nodes;
            num_level_width = LEVEL3_WIDTH;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }
    for(int i = 0; i < num_level_width; i++){
        for(int j = 0; j < num_level_width; j++){
            write_knowledge(&(target_level_nodes[i * num_level_width + j]));
        }
    }
}

/* free space of level one nodes on gpu */
void free_level_one_gpu(){
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        for(int j = 0; j < LEVEL1_WIDTH; j++){
            checkCudaErrors(cudaFree(cpu_level_one_nodes[i *
                        LEVEL1_WIDTH + j].QC_gpu));
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].QC_gpu = NULL;

            checkCudaErrors(cudaFree(cpu_level_one_nodes[i *
                        LEVEL1_WIDTH + j].time_adj_matrix_gpu));
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].time_adj_matrix_gpu = NULL;

            checkCudaErrors(cudaFree(cpu_level_one_nodes[i * LEVEL1_WIDTH +
                        j].group_id_gpu));
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].group_id_gpu = NULL;

            checkCudaErrors(cudaFree(cpu_level_one_nodes[i * LEVEL1_WIDTH +
                        j].prob_QC_gpu));
            cpu_level_one_nodes[i * LEVEL1_WIDTH + j].prob_QC_gpu = NULL;
        }
    }
    
    checkCudaErrors(cudaFree(gpu_level_one_nodes));
    gpu_level_one_nodes = NULL;
}

/* free space of time adjacent matrix of each level two nodes */
void free_level_two_matrix_gpu(){
    for(int i = 0; i < LEVEL2_WIDTH; i++){
        for(int j = 0; j < LEVEL2_WIDTH; j++){
            checkCudaErrors(cudaFree(cpu_level_two_nodes[i *
                        LEVEL2_WIDTH + j].time_adj_matrix_gpu));
            cpu_level_two_nodes[i * LEVEL2_WIDTH + j].time_adj_matrix_gpu = NULL;

        }
    }
}

int infer_map[NUM_TG_TOP] = {3, 4, 5, 6, 7, 8, 9, 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'h', 'k', 'm', 'n', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'z', 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008,
    };
int answer_map[NUM_TG_TOP] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57
    };
int infer_tg_index[NUM_INFER_IMAGE];
int infer_result[NUM_INFER_IMAGE];
#endif
