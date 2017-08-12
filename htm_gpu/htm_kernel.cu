#ifndef HTM_KERNEL_H
#define HTM_KERNEL_H

#include "node.h"

/* helper function to find the start index of the needed inference result 
   @node_index index of the node
   @num_TG number of temporal groups of the node
   @img_index index of the current image
   @part flag indicating whether is learning vertically or horizontally
   @gpu_infer_from_index index pointing to start of the inferenced result of the node
   @image_x_offset_gpu length of an image that can shift vertically
   @image_y_offset_gpu length of an image that can shift horizontally
*/
__device__ inline int get_from_index(int node_index, int num_TG, int img_index, int
        base_row, int base_column, int part, int* gpu_infer_from_index, int*
        image_x_offset_gpu, int* image_y_offset_gpu){
    int infer_from_index = -1;
    int position_index = 0;
    if(part == 1){
        infer_from_index = gpu_infer_from_index[node_index];
        position_index = 0;
        for(int former_img = 0; former_img < img_index;
                former_img++){
            position_index += image_x_offset_gpu[former_img] *
                image_y_offset_gpu[former_img];
        }
        position_index *= 2;
        position_index += (IMAGE_WIDTH - base_row) *
            image_y_offset_gpu[img_index] + (IMAGE_WIDTH - base_column);
        infer_from_index += position_index * num_TG;
    }else if(part == 2){
        infer_from_index = gpu_infer_from_index[node_index];
        position_index = 0;
        for(int former_img = 0; former_img < img_index;
                former_img++){
            position_index += image_x_offset_gpu[former_img] *
                image_y_offset_gpu[former_img];
        }
        position_index *= 2;
        position_index += image_x_offset_gpu[img_index] *
            image_y_offset_gpu[img_index];
        position_index += (IMAGE_WIDTH - base_column) *
            image_x_offset_gpu[img_index] + (IMAGE_WIDTH - base_row);
        infer_from_index += position_index * num_TG;
    }
    return infer_from_index;
}

/* helper function to find the start index of the needed temporal group of level two
   @node_index index of the node
   @num_TG number of temporal groups of the node
   @img_index index of the current image
   @part flag indicating whether is learning vertically or horizontally
   @gpu_image_from_index_lv2 index pointing to the start of each image used by level two
*/
__device__ inline int get_from_index_tg_lv2(int node_index, int img_index, int
        base_row, int base_column, int part, int* image_x_offset_gpu, int*
        image_y_offset_gpu, int* gpu_image_from_index_lv2){
    int infer_from_index = -1;
    int position_index = 0;
    if(part == 1){
        infer_from_index = gpu_image_from_index_lv2[img_index];
        position_index = (IMAGE_WIDTH - base_row) *
            image_y_offset_gpu[img_index] + (IMAGE_WIDTH - base_column);
        infer_from_index += position_index * LEVEL2_WIDTH * LEVEL2_WIDTH; 
        infer_from_index += node_index;
    }else if(part == 2){
        infer_from_index = gpu_image_from_index_lv2[img_index];
        position_index = image_x_offset_gpu[img_index] *
            image_y_offset_gpu[img_index];
        position_index += (IMAGE_WIDTH - base_column) *
            image_x_offset_gpu[img_index] + (IMAGE_WIDTH - base_row);

        infer_from_index += position_index * LEVEL2_WIDTH * LEVEL2_WIDTH;
        infer_from_index += node_index;
    }
    return infer_from_index; 
}

/* helper function to find the start index of the needed temporal group of level one
   @node_index index of the node
   @num_TG number of temporal groups of the node
   @img_index index of the current image
   @part flag indicating whether is learning vertically or horizontally
   @image_x_offset_gpu length of an image that can shift vertically
   @image_y_offset_gpu length of an image that can shift horizontally
   @gpu_image_from_index index pointing to the start of each image used by level one
*/
__device__ inline int get_from_index_tg_lv1(int num_images, int node_index, int img_index, int
        base_row, int base_column, int part, int* image_x_offset_gpu, int*
        image_y_offset_gpu, int* gpu_image_from_index){
    int infer_from_index = 0;
    int total_position = gpu_image_from_index[num_images - 1] +
        image_x_offset_gpu[num_images - 1] * image_y_offset_gpu[num_images - 1] *
        2;
    int position_index = 0;

    int row = node_index / LEVEL1_WIDTH;
    int column = node_index % LEVEL1_WIDTH;
    int row_farther = row / 2;
    int column_farther = column / 2;
    int farther_node_index = row_farther * LEVEL2_WIDTH + column_farther;

    for(int i = 0; i < farther_node_index; i++){
        infer_from_index += total_position * NUM_CHILD_LV2;
    }

    if(part == 1){
        position_index = gpu_image_from_index[img_index];
        position_index += (IMAGE_WIDTH - base_row) *
            image_y_offset_gpu[img_index] + (IMAGE_WIDTH - base_column);
        infer_from_index += position_index * NUM_CHILD_LV2; 
        infer_from_index += (row % 2) * 2 + column % 2;
    }else if(part == 2){
        position_index = gpu_image_from_index[img_index];
        position_index += image_x_offset_gpu[img_index] *
            image_y_offset_gpu[img_index];
        position_index += (IMAGE_WIDTH - base_column) *
            image_x_offset_gpu[img_index] + (IMAGE_WIDTH - base_row);
        infer_from_index += position_index * NUM_CHILD_LV2; 
        infer_from_index += (row % 2) * 2 + column % 2;
    }
    return infer_from_index;
}

/* kernel to learn level one nodes
 * NUM_THREADS_LEARN_LV1 threads run concurrently to learn one node
 * each thread compares the current input to num_QC / NUM_THREADS_LEARN_LV1 existing QCs
 * thread 0 induces results from all threads
 @node pointer to the current node
 @node_index index of the node
 @images_gpu pointer of the image information
 @gpu_qc_same_result_lv1 array to store whether the input is the same with
 existing QC
*/
__device__ void kernel_learn_node_lv1(node_t* node, int node_index, int* images_gpu, int node_base_row, int
        node_base_column, int img_offset, int* gpu_qc_same_result_lv1){

    int thread_id = threadIdx.x;
    if(thread_id == 0){
        node->closest_index = -1;
    }

    /* get input QC from child nodes */
    for(int l = 0; l < 4; l++){
        for(int m = 0; m < 4; m++){
            node->input_QC[4 * l + m] = images_gpu[img_offset + (node_base_row + l) * ST_IMAGE_WIDTH
                + node_base_column + m];
        }
    }             

    int same;
    gpu_qc_same_result_lv1[node_index * NUM_THREADS_LEARN_LV1 +
        thread_id] = 0;
    /* each thread compares the input QC with num_QC / NUM_THREADS_LEARN_LV1 existing QCs */
    for(int i = 0; i < node->num_QC / NUM_THREADS_LEARN_LV1 + 1; i++){
        int qc_index = i * NUM_THREADS_LEARN_LV1 + thread_id;
        if(qc_index < node->num_QC){
            same = 1;
            for(int j = 0; j < node->num_child; j++){
                if(node->QC_gpu[qc_index * node->num_child + j] != node->input_QC[j]){
                    same = 0;
                    break;
                }
            }
            if(same){
                gpu_qc_same_result_lv1[node_index * NUM_THREADS_LEARN_LV1 +
                    thread_id] = 1;
                node->closest_index = qc_index;
                break;
            }
        }
    }

    __syncthreads();
    /* thread 0 induces results */
    if(thread_id == 0){
        same = 0;
        for(int i = 0; i < NUM_THREADS_LEARN_LV1; i++){
            same |= gpu_qc_same_result_lv1[node_index * NUM_THREADS_LEARN_LV1 + i];
        }

        if(node->num_QC == 0){
            same = 0;
        }

        if(!same){
            if(node->num_QC + 1 > node->num_max_QC){
            }
            for(int i = 0; i < node->num_child; i++){
                node->QC_gpu[node->num_QC * node->num_child + i] = node->input_QC[i];
            }
            node->closest_index = node->num_QC;
            node->num_QC++;
        }

        if(node->last_QC == -1){
            node->last_QC = 0;
        }else{
            node->time_adj_matrix_gpu[node->last_QC * node->num_max_QC + node->closest_index]++; 
            node->last_QC = node->closest_index; 
        }
    }
    __syncthreads();
}

/* learn level one nodes by learning the movie of each image
 * the movie is created by moving the same image horizontally and vertically
 * the possible position of the movie is specified by image_x_offset and image_y_offset
 @num_images number of images
 @node_gpu pointer to nodes on gpu
 @image_x_offset_gpu length of an image that can shift vertically
 @image_y_offset_gpu length of an image that can shift horizontally
*/
__global__ void kernel_learn_lv1(int num_images, node_t* node_gpu, int* images_gpu, int*
        image_x_offset_gpu, int* image_y_offset_gpu, int* gpu_qc_same_result_lv1){
    int node_index = blockIdx.x * 8 + blockIdx.y;
    node_t* node = &(node_gpu[node_index]);

    int dir, from_column, from_row, node_base_row, node_base_column, img_offset;
    for(int img_index = 0; img_index < num_images; img_index++){
	    for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                image_x_offset_gpu[img_index]; base_row--){
	        dir = base_row % 2;
	        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH - image_y_offset_gpu[img_index]; base_column--){
	            if(dir == 0){
	                from_column = base_column;
	            }else{
	                from_column = 2 * IMAGE_WIDTH - base_column -
                        image_y_offset_gpu[img_index] + 1;
	            }

	            node_base_row = base_row + node->row * 4;
	            node_base_column = from_column + node->column * 4;
	            img_offset = img_index * ST_IMAGE_WIDTH * ST_IMAGE_WIDTH;

                kernel_learn_node_lv1(node, node_index, images_gpu, node_base_row, node_base_column,
                        img_offset, gpu_qc_same_result_lv1);
	        }

	    }

        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
                image_y_offset_gpu[img_index]; base_column--){
            dir = base_column % 2;
            for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                    image_x_offset_gpu[img_index]; base_row--){
                if(dir == 0){
                    from_row = base_row;
                }else{
                    from_row = 2 * IMAGE_WIDTH - base_row -
                        image_x_offset_gpu[img_index] + 1;
                }

	            node_base_row = from_row + node->row * 4;
	            node_base_column = base_column + node->column * 4;
	            img_offset = img_index * ST_IMAGE_WIDTH * ST_IMAGE_WIDTH;

                kernel_learn_node_lv1(node, node_index, images_gpu, node_base_row, node_base_column,
                            img_offset, gpu_qc_same_result_lv1);
            }
        }
    }
}

/* kernel to inference level one nodes
 * NUM_THREADS_INFERENCE_LV1 threads run concurrently to inference one node
 * each thread calculate the probability of the current input to be considered matching 
 * num_QC / NUM_THREADS_LEARN_LV1 existing QCs
 * thread 0 induces results from all threads
 @node pointer to the current node
 @images_gpu pointer of the image information
 @inference_result pointer to store the calculate inference result
*/
__device__ void kernel_inference_node_lv1(node_t* node, int* images_gpu, int node_base_row, int
        node_base_column, int img_offset, float* inference_result, int inference_from_index, 
        int inference_from_index_tg, int* gpu_inference_max_prob_tg_lv1, int
        run_mode
        ){

    float sigma_square;
    if(run_mode == TRAINING_MODE){
        sigma_square = 0.1;
    }else{
        sigma_square = 1.5;
    }
    int thread_id = threadIdx.x;
    int input_groupid[NUM_CHILD_LV1];
    for(int l = 0; l < 4; l++){
        for(int m = 0; m < 4; m++){
            input_groupid[4 * l + m] = images_gpu[img_offset + (node_base_row + l) * ST_IMAGE_WIDTH
                + node_base_column + m];
        }
    }      
    /* each thread calculates probability distribution over num_QC / NUM_THREADS_INFERENCE_LV1
       QCs and record it to vector prob_QC */
    int ed = 0;    // euclidean distance
    int QC_value = -1;    // value of the the specific child of the specific QC
    int diff = 0;    
    for(int i = 0; i < node->num_QC / NUM_THREADS_INFERENCE_LV1 + 1; i++){
        int qc_index = i * NUM_THREADS_INFERENCE_LV1 + thread_id;
        if(qc_index < node->num_QC){
            ed = 0;
            for(int j = 0; j < NUM_CHILD_LV1; j++){
                QC_value = node->QC_gpu[qc_index * NUM_CHILD_LV1 + j];
                diff = QC_value - input_groupid[j];
                ed += diff * diff;
            }
            node->prob_QC_gpu[qc_index] = __expf(0 - ed / sigma_square);
        }
    }

    __syncthreads();
    /* thread 0 induces inference result of all threads */
    if(thread_id == 0){
        node->inference_groupid = -1;
        __shared__ float output_inner[MAX_NUM_TG];
        for(int i = 0; i < MAX_NUM_TG; i++){
            output_inner[i] = 0;
        }
        // select the largest QC prob of the TG as the prob of the TG
        for(int i = 0; i < node->num_QC; i++){
            if(output_inner[node->group_id_gpu[i]] < node->prob_QC_gpu[i]){
                output_inner[node->group_id_gpu[i]] = node->prob_QC_gpu[i];
            }
        }
    
        // normalize the output vector
        float sum_prob = 0;
        for(int i = 0; i < node->num_TG; i++){
            sum_prob += output_inner[i];
        }
        for(int i = 0; i < node->num_TG; i++){
            output_inner[i] /= sum_prob;

            inference_result[inference_from_index + i] = output_inner[i];
        }

        float prob_tmp = 0;
        for(int i = 0; i < node->num_TG; i++){
            if(prob_tmp < output_inner[i]){
                prob_tmp = output_inner[i];
                node->inference_groupid = i;
            }
        }
        gpu_inference_max_prob_tg_lv1[inference_from_index_tg] =
            node->inference_groupid;
    }
    __syncthreads();
}

/* inference level one nodes by inference the movie of each image
 * the movie is created by moving the same image horizontally and vertically
 * the possible position of the movie is specified by image_x_offset and image_y_offset
 @num_images number of images
 @node_gpu pointer to nodes on gpu
 @gpu_inference_result_lv1 pointer to start of inference result of level one nodes
 @image_x_offset_gpu length of an image that can shift vertically
 @image_y_offset_gpu length of an image that can shift horizontally
*/
__global__ void kernel_inference_lv1(int num_images, node_t* node_gpu, int* images_gpu, int*
        image_x_offset_gpu, int* image_y_offset_gpu, float*
        gpu_inference_result_lv1, int* gpu_infer_from_index_lv1, int*
        gpu_image_from_index, int* gpu_inference_max_prob_tg_lv1, int run_mode
        ){

    int node_index = blockIdx.x * 8 + blockIdx.y;
    node_t* node = &(node_gpu[node_index]);
    int dir, from_column, from_row, node_base_row, node_base_column, img_offset;
    for(int img_index = 0; img_index < num_images; img_index++){
	    for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                image_x_offset_gpu[img_index]; base_row--){
	        dir = base_row % 2;
	        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH - image_y_offset_gpu[img_index]; base_column--){
	            if(dir == 0){
	                from_column = base_column;
	            }else{
	                from_column = 2 * IMAGE_WIDTH - base_column -
                        image_y_offset_gpu[img_index] + 1;
	            }

	            node_base_row = base_row + node->row * 4;
	            node_base_column = from_column + node->column * 4;
	            img_offset = img_index * ST_IMAGE_WIDTH * ST_IMAGE_WIDTH;

                int infer_from_index = get_from_index(node_index, node->num_TG, img_index, 
                        base_row, base_column, 1,  gpu_infer_from_index_lv1, image_x_offset_gpu, image_y_offset_gpu);
                int infer_from_index_tg = get_from_index_tg_lv1(num_images, node_index,
                        img_index, base_row, base_column, 1,
                        image_x_offset_gpu, image_y_offset_gpu,
                        gpu_image_from_index); 
                kernel_inference_node_lv1(node, images_gpu, node_base_row, node_base_column,
                        img_offset, gpu_inference_result_lv1,
                        infer_from_index, infer_from_index_tg, gpu_inference_max_prob_tg_lv1, run_mode);
	        }

	    }

        if(run_mode == TRAINING_MODE){
        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
                image_y_offset_gpu[img_index]; base_column--){
            dir = base_column % 2;
            for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                    image_x_offset_gpu[img_index]; base_row--){
                if(dir == 0){
                    from_row = base_row;
                }else{
                    from_row = 2 * IMAGE_WIDTH - base_row -
                        image_x_offset_gpu[img_index] + 1;
                }

	            node_base_row = from_row + node->row * 4;
	            node_base_column = base_column + node->column * 4;
	            img_offset = img_index * ST_IMAGE_WIDTH * ST_IMAGE_WIDTH;

                int infer_from_index = get_from_index(node_index, node->num_TG, img_index, 
                        base_row, base_column, 2,  gpu_infer_from_index_lv1, image_x_offset_gpu, image_y_offset_gpu);

                int infer_from_index_tg = get_from_index_tg_lv1(num_images, node_index,
                        img_index, base_row, base_column, 2,
                        image_x_offset_gpu, image_y_offset_gpu,
                        gpu_image_from_index); 

                kernel_inference_node_lv1(node, images_gpu, node_base_row, node_base_column,
                            img_offset, gpu_inference_result_lv1,
                            infer_from_index, infer_from_index_tg,
                            gpu_inference_max_prob_tg_lv1, run_mode);
            }
        }
        }
    }
}

/* kernel to learn level two nodes
 * NUM_THREADS_LEARN_LV2 threads run concurrently to learn one node
 * each thread compares the current input to num_QC / NUM_THREADS_LEARN_LV2 existing QCs
 * thread 0 induces results from all threads
 @num_images number of total images
 @node pointer to the current node
 @node_index index of the node
 @images_gpu pointer of the image information
 @gpu_inference_result_lv1 pointer to the inference result of level one
*/
__device__ void kernel_learn_node_lv2(int num_images, node_t* node, int node_index, int img_index, int
        base_row, int base_column, int part, int* gpu_infer_from_index_lv1, int* image_x_offset_gpu, int* image_y_offset_gpu, 
        float* gpu_inference_result_lv1, int* gpu_qc_same_result_lv2, int*
        gpu_inference_max_prob_tg_lv1, int* gpu_image_from_index){

    int thread_id = threadIdx.x;
    if(thread_id == 0){
        node->closest_index = -1;
    }

    /* get input_QC of the current node */
    int lv2_row = node->row;
    int lv2_col = node->column;
    int sub_width = 8 / 4;
    int lv1_node_index = (lv2_row * sub_width) * 8 + lv2_col * sub_width;
    int infer_from_index = get_from_index_tg_lv1(num_images, lv1_node_index, img_index,
            base_row, base_column, part, image_x_offset_gpu, image_y_offset_gpu,
            gpu_image_from_index);
    for(int i = 0; i < NUM_CHILD_LV2; i++){
        node->input_QC[i] = gpu_inference_max_prob_tg_lv1[infer_from_index + i];
    }

    int same;
    gpu_qc_same_result_lv2[node_index * NUM_THREADS_LEARN_LV2 +
        thread_id] = 0;
    /* each thread compares the input QC with num_QC / NUM_THREADS_LEARN_LV2 existing QCs */
    for(int i = 0; i < node->num_QC / NUM_THREADS_LEARN_LV2 + 1; i++){
        int qc_index = i * NUM_THREADS_LEARN_LV2 + thread_id;
        if(qc_index < node->num_QC){
        same = 1;
        for(int j = 0; j < node->num_child; j++){
            if(node->QC_gpu[qc_index * node->num_child + j] != node->input_QC[j]){
                same = 0;
                break;
            }
        }
        if(same){
            gpu_qc_same_result_lv2[node_index * NUM_THREADS_LEARN_LV2 +
                thread_id] = 1;
            node->closest_index = qc_index;
            break;
        }
        }
    }

    __syncthreads();
    /* thread 0 induces results */
    if(thread_id == 0){
        same = 0;
        for(int i = 0; i < NUM_THREADS_LEARN_LV2; i++){
            same |= gpu_qc_same_result_lv2[node_index * NUM_THREADS_LEARN_LV2 + i];
        }

        if(node->num_QC == 0){
            same = 0;
        }

        if(!same){
            if(node->num_QC + 1 > node->num_max_QC){
            }
            for(int i = 0; i < node->num_child; i++){
                node->QC_gpu[node->num_QC * node->num_child + i] = node->input_QC[i];
            }
            node->closest_index = node->num_QC;
            node->num_QC++;
        }

        if(node->last_QC == -1){
            node->last_QC = 0;
        }else{
            node->time_adj_matrix_gpu[node->last_QC * node->num_max_QC + node->closest_index]++; 
            node->last_QC = node->closest_index; 
        }
    }
    __syncthreads();
}

/* learn level two nodes by learning the movie of each image
 * the movie is created by moving the same image horizontally and vertically
 * the possible position of the movie is specified by image_x_offset and image_y_offset
 @num_images number of images
 @node_gpu pointer to nodes on gpu
 @images_gpu pointer to image information stored in gpu
 @image_x_offset_gpu length of an image that can shift vertically
 @image_y_offset_gpu length of an image that can shift horizontally
 @gpu_inference_result_lv2 pointer to inference result of level one
*/
__global__ void kernel_learn_lv2(int num_images, node_t* node_gpu, int* images_gpu, int*
        image_x_offset_gpu, int* image_y_offset_gpu, float*
        gpu_inference_result_lv1, int* gpu_infer_from_index_lv1,
        int* gpu_qc_same_result_lv2, int*
        gpu_inference_max_prob_tg_lv1, int* gpu_image_from_index){
    int node_index = blockIdx.x * 4 + blockIdx.y;
    node_t* node = &(node_gpu[node_index]);

    for(int img_index = 0; img_index < num_images; img_index++){
	    for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                image_x_offset_gpu[img_index]; base_row--){
	        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH - image_y_offset_gpu[img_index]; base_column--){

                kernel_learn_node_lv2(num_images, node, node_index, img_index, base_row, base_column, 1,
                    gpu_infer_from_index_lv1, image_x_offset_gpu, image_y_offset_gpu, gpu_inference_result_lv1,
                    gpu_qc_same_result_lv2, gpu_inference_max_prob_tg_lv1, gpu_image_from_index);
	        }

	    }

        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
                image_y_offset_gpu[img_index]; base_column--){
            for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                    image_x_offset_gpu[img_index]; base_row--){

                kernel_learn_node_lv2(num_images, node, node_index, img_index, base_row, base_column, 2,
                        gpu_infer_from_index_lv1, image_x_offset_gpu, image_y_offset_gpu, gpu_inference_result_lv1,
                        gpu_qc_same_result_lv2, gpu_inference_max_prob_tg_lv1, gpu_image_from_index);
            }
        }
    }
}

/* kernel to inference level two nodes
 * NUM_THREADS_INFERENCE_LV2 threads run concurrently to inference one node
 * each thread calculate the probability of the current input to be considered matching 
 * num_QC / NUM_THREADS_LEARN_LV2 existing QCs
 * thread 0 induces results from all threads
 @node pointer to the current node
 @node_index index of the current node in the node array
 @gpu_inference_result_lv1 pointer to inference result of level one
 @gpu_inference_result_lv2 pointer to inference result of level two
 @run_mode can be TRAINING_MODE or INFERENCE_MODE
*/
__device__ void kernel_inference_node_lv2(node_t* node, int node_index, int img_index, int
        base_row, int base_column, int part, int* gpu_infer_from_index_lv1, int* image_x_offset_gpu, int* image_y_offset_gpu, 
        float* gpu_inference_result_lv1, float* gpu_inference_result_lv2, int
        inference_from_index, int* gpu_inference_max_prob_tg_lv2, int
        infer_from_index_tg, int run_mode, float* input_inner){

    float sigma_square;
    if(run_mode == TRAINING_MODE){
        sigma_square = 0.1;
    }else{
        sigma_square = 1.5;
    }
    int thread_id = threadIdx.x;
    int lv2_row = node->row;
    int lv2_col = node->column;
    for(int l = 0; l < 2; l++){
        for(int m = 0; m < 2; m++){
            int lv1_node_index = (lv2_row * 2 + l) * 8 + lv2_col * 2 + m;
            int infer_from_index = get_from_index(lv1_node_index, node->num_child_group[l * 2 + m] , img_index, 
                base_row, base_column, part,  gpu_infer_from_index_lv1, image_x_offset_gpu, image_y_offset_gpu);
            for(int n = 0; n < node->num_child_group[l * 2 + m]; n++){
                input_inner[(2 * l + m) * MAX_NUM_TG + n] = gpu_inference_result_lv1[infer_from_index + n];
            }
        }
    }

    float ed = 0;    // euclidean distance
    int QC_value = -1;    // value of the the specific child of the specific QC
    float diff = 0;    
    /* each thread calculates probability distribution over num_QC / NUM_THREADS_INFERENCE_LV2
       QCs and record it to vector prob_QC */
    for(int i = 0; i < node->num_QC / NUM_THREADS_INFERENCE_LV2 + 1; i++){
        int qc_index = i * NUM_THREADS_INFERENCE_LV2 + thread_id;
        if(qc_index < node->num_QC){
            ed = 0;
            for(int j = 0; j < NUM_CHILD_LV2; j++){
                QC_value = node->QC_gpu[qc_index * NUM_CHILD_LV2 + j];
                diff = 1 - input_inner[j * MAX_NUM_TG + QC_value];
                ed += diff * diff;
            }
            node->prob_QC_gpu[qc_index] = __expf(0 - ed / sigma_square);
        }
    }

    __syncthreads();
    /* thread 0 induces inference result of all threads */
    if(thread_id == 0){
        node->inference_groupid = -1;

        __shared__ float output_inner[MAX_NUM_TG];
        for(int i = 0; i < MAX_NUM_TG; i++){
            output_inner[i] = 0;
        }
        for(int i = 0; i < node->num_QC; i++){
            if(output_inner[node->group_id_gpu[i]] < node->prob_QC_gpu[i]){
                output_inner[node->group_id_gpu[i]] = node->prob_QC_gpu[i];
            }
        }

        float sum_prob = 0;
        for(int i = 0; i < node->num_TG; i++){
            sum_prob += output_inner[i];

            if(run_mode == INFERENCE_MODE){
                gpu_inference_result_lv2[inference_from_index + i] = output_inner[i];
            }
        }
        for(int i = 0; i < node->num_TG; i++){
            output_inner[i] /= sum_prob;
        }

        ed = 0;
        for(int i = 0; i < node->num_TG; i++){
            if(ed < output_inner[i]){
                ed = output_inner[i];
                node->inference_groupid = i;
            }
        gpu_inference_max_prob_tg_lv2[infer_from_index_tg] =
            node->inference_groupid;
        }
    }
    __syncthreads();
}

/* inference level two nodes by inference the movie of each image
 * the movie is created by moving the same image horizontally and vertically
 * the possible position of the movie is specified by image_x_offset and image_y_offset
 @num_images number of images
 @node_gpu pointer to nodes on gpu
 @image_gpu pointer to image information stored in gpu
 @gpu_inference_result_lv1 pointer to start of inference result of level one nodes
 @gpu_inference_result_lv2 pointer to start of inference result of level two nodes
 @image_x_offset_gpu length of an image that can shift vertically
 @image_y_offset_gpu length of an image that can shift horizontally
*/
__global__ void kernel_inference_lv2(int num_images, node_t* node_gpu, int* images_gpu, int*
        image_x_offset_gpu, int* image_y_offset_gpu, float*
        gpu_inference_result_lv1, float* gpu_inference_result_lv2, int* gpu_infer_from_index_lv1,
        int* gpu_infer_from_index_lv2, int*
        gpu_inference_max_prob_tg_lv2, int* gpu_image_from_index_lv2, int
        run_mode){
    int node_index = blockIdx.x * 4 + blockIdx.y;
    node_t* node = &(node_gpu[node_index]);

    __shared__ float input_inner[NUM_CHILD_LV2 * MAX_NUM_TG];
    for(int img_index = 0; img_index < num_images; img_index++){
	    for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                image_x_offset_gpu[img_index]; base_row--){
	        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH - image_y_offset_gpu[img_index]; base_column--){

                int infer_from_index = get_from_index(node_index, node->num_TG, img_index, base_row, base_column, 1,
                        gpu_infer_from_index_lv2, image_x_offset_gpu, image_y_offset_gpu);
                int infer_from_index_tg = get_from_index_tg_lv2(node_index,
                        img_index, base_row, base_column, 1,
                        image_x_offset_gpu, image_y_offset_gpu,
                        gpu_image_from_index_lv2);

                kernel_inference_node_lv2(node, node_index, img_index, base_row, base_column, 1,  gpu_infer_from_index_lv1,
                        image_x_offset_gpu, image_y_offset_gpu, gpu_inference_result_lv1, gpu_inference_result_lv2,
                        infer_from_index, gpu_inference_max_prob_tg_lv2,
                        infer_from_index_tg, run_mode, input_inner);
	        }
	    }

        if(run_mode == TRAINING_MODE){
        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
                image_y_offset_gpu[img_index]; base_column--){
            for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                    image_x_offset_gpu[img_index]; base_row--){

                int infer_from_index = get_from_index(node_index, node->num_TG, img_index, base_row, base_column, 2,
                        gpu_infer_from_index_lv2, image_x_offset_gpu, image_y_offset_gpu);
                int infer_from_index_tg = get_from_index_tg_lv2(node_index,
                        img_index, base_row, base_column, 2,
                        image_x_offset_gpu, image_y_offset_gpu,
                        gpu_image_from_index_lv2);
                kernel_inference_node_lv2(node, node_index, img_index, base_row, base_column, 2,  gpu_infer_from_index_lv1, 
                        image_x_offset_gpu, image_y_offset_gpu, gpu_inference_result_lv1, gpu_inference_result_lv2,
                        infer_from_index, gpu_inference_max_prob_tg_lv2,
                        infer_from_index_tg, run_mode, input_inner);
            }
        }
        }
    }
}

/* kernel to learn level three nodes
 * NUM_THREADS_LEARN_LV3 threads run concurrently to learn one node
 * each thread compares the current input to num_QC / NUM_THREADS_LEARN_LV3 existing QCs
 * thread 0 induces results from all threads
 @node pointer to the current node
 @node_index index of the node
 @image_x_offset_gpu length of an image that can shift vertically
 @image_y_offset_gpu length of an image that can shift horizontally
 @gpu_inference_result_lv1 pointer to the inference result of level one
*/
__device__ void kernel_learn_node_lv3(node_t* node, int node_index, int img_index, int
        base_row, int base_column, int part, int* gpu_infer_from_index_lv2, int* image_x_offset_gpu, int* image_y_offset_gpu, 
        int* gpu_image_from_index_lv2, int*
        gpu_inference_max_prob_tg_lv2, int* gpu_qc_same_result_lv3){

    /* get input QC */
    int infer_from_index_tg = get_from_index_tg_lv2(node_index,
        img_index, base_row, base_column, part,
        image_x_offset_gpu, image_y_offset_gpu,
        gpu_image_from_index_lv2);

    for(int i = 0; i < NUM_CHILD_LV3; i++){
        node->input_QC[i] = gpu_inference_max_prob_tg_lv2[infer_from_index_tg + i]; 
    }

    int thread_id = threadIdx.x;
    gpu_qc_same_result_lv3[thread_id] = 0;
    int same;
    /* each thread compares the input QC with num_QC / NUM_THREADS_LEARN_LV2 existing QCs */
    for(int i = 0; i < node->num_QC / NUM_THREADS_LEARN_LV3 + 1; i++){
        int qc_index = i * NUM_THREADS_LEARN_LV3 + thread_id;
        if(qc_index < node->num_QC){
            same = 1;
            for(int j = 0; j < NUM_CHILD_LV3; j++){
                if(node->QC_gpu[qc_index * NUM_CHILD_LV3 + j] != node->input_QC[j]){
                    same = 0;
                    break;
                }
            }
            if(same){
                gpu_qc_same_result_lv3[thread_id] = 1;
                break;
            }
        }
    }

    __syncthreads();
    /* thread 0 induces results */
    if(thread_id == 0){
        same = 0;
        for(int i = 0; i < NUM_THREADS_LEARN_LV3; i++){
            same |= gpu_qc_same_result_lv3[i];
        }
        if(node->num_QC == 0){
            same = 0;
        }

        if(!same){
            for(int i = 0; i < NUM_CHILD_LV3; i++){
                node->QC_gpu[node->num_QC * node->num_child + i] = node->input_QC[i];
            }
            node->num_QC++;
        }
    }
    __syncthreads();
}

/* kernel to learn level three nodes
 * NUM_THREADS_LEARN_LV3 threads run concurrently to learn one node
 * each thread compares the current input to num_QC / NUM_THREADS_LEARN_LV3 existing QCs
 * thread 0 induces results from all threads
 @num_images number of images
 @node_gpu pointer to node array stored in gpu
 @images_gpu pointer to image information stored in gpu
 @images_gpu pointer of the image information
 @image_x_offset_gpu length of an image that can shift vertically
 @image_y_offset_gpu length of an image that can shift horizontally
*/
__global__ void kernel_learn_lv3(int num_images, node_t* node_gpu, int* images_gpu, int*
        image_x_offset_gpu, int* image_y_offset_gpu, int* gpu_infer_from_index_lv2, int*
        gpu_image_from_index_lv2, int* gpu_inference_max_prob_tg_lv2, int*
        gpu_qc_same_result_lv3){
    //printf("inside learn lv3\n");
    node_t* node = &(node_gpu[0]);

    int super_index = 0;
    for(int img_index = 0; img_index < num_images; img_index++){
	    for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                image_x_offset_gpu[img_index]; base_row--){
	        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH - image_y_offset_gpu[img_index]; base_column--){

                kernel_learn_node_lv3(node, 0, img_index, base_row, base_column, 1,
                        gpu_infer_from_index_lv2, image_x_offset_gpu, image_y_offset_gpu,
                        gpu_image_from_index_lv2, gpu_inference_max_prob_tg_lv2, gpu_qc_same_result_lv3);
	        }

	    }

        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
                image_y_offset_gpu[img_index]; base_column--){
            for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                    image_x_offset_gpu[img_index]; base_row--){

                kernel_learn_node_lv3(node, 0, img_index, base_row, base_column, 2, 
                        gpu_infer_from_index_lv2, image_x_offset_gpu, image_y_offset_gpu,
                        gpu_image_from_index_lv2, gpu_inference_max_prob_tg_lv2, gpu_qc_same_result_lv3);
            }
        }
        if(threadIdx.x == 0){
            for(int i = super_index; i < node->num_QC; i++){
                node->group_id_gpu[i] = img_index;
            }
            node->num_TG++;
            super_index = node->num_QC;
        }
    }
}

/* kernel to inference level three nodes
 * NUM_THREADS_INFERENCE_LV3 threads run concurrently to inference one node
 * each thread calculate the probability of the current input to be considered matching 
 * num_QC / NUM_THREADS_LEARN_LV3 existing QCs
 * thread 0 induces results from all threads
 @node pointer to the current node
 @node_index index of the current node in the node array
 @gpu_inference_result_lv2 pointer to inference result of level two
 @image_x_offset_gpu length of an image that can shift vertically
 @image_y_offset_gpu length of an image that can shift horizontally
*/
__device__ void kernel_inference_node_lv3(node_t* node, int node_index, int img_index, int
        base_row, int base_column, int part, int* gpu_infer_from_index_lv2, int* image_x_offset_gpu, int* image_y_offset_gpu, 
        float* gpu_inference_result_lv2, occur_prob_t* gpu_imove, float*
        input_inner){

    float sigma_square = 1.5;
    int thread_id = threadIdx.x;
    for(int i = 0; i < node->num_child; i++){
        int infer_from_index = get_from_index(i, node->num_child_group[i],
                img_index, base_row, base_column, part,
                gpu_infer_from_index_lv2, image_x_offset_gpu,
                image_y_offset_gpu);
        for(int j = 0; j < node->num_child_group[i]; j++){
            input_inner[i * MAX_NUM_TG + j] =
                gpu_inference_result_lv2[infer_from_index + j]; 
        }
    }

    float ed = 0;    // euclidean distance
    int QC_value = -1;    // value of the the specific child of the specific QC
    float diff = 0;    
    /* each thread calculates probability distribution over num_QC / NUM_THREADS_INFERENCE_LV3
       QCs and record it to vector prob_QC */
    for(int i = 0; i < node->num_QC / NUM_THREADS_INFERENCE_LV3 + 1; i++){
        int qc_index = i * NUM_THREADS_INFERENCE_LV3 + thread_id;
        if(qc_index < node->num_QC){
            ed = 0;
            for(int j = 0; j < NUM_CHILD_LV3; j++){
                QC_value = node->QC_gpu[qc_index * NUM_CHILD_LV3 + j];
                diff = 1 - input_inner[j * MAX_NUM_TG + QC_value];
                ed += diff * diff;
            }
            node->prob_QC_gpu[qc_index] = __expf(0 - ed / sigma_square);
        }
    }    

    __syncthreads();
    /* thread 0 induces results from other threads */
    if(thread_id == 0){
        node->inference_groupid = -1;
        __shared__ float output_inner[MAX_NUM_TG];
        for(int i = 0;i < MAX_NUM_TG; i++){
            output_inner[i] = 0;
        }
        for(int i = 0; i < node->num_QC; i++){
            if(output_inner[node->group_id_gpu[i]] < node->prob_QC_gpu[i]){
                output_inner[node->group_id_gpu[i]] = node->prob_QC_gpu[i];
            }
        }

        float sum_prob = 0;
        for(int i = 0; i < node->num_TG; i++){
            sum_prob += output_inner[i];
        }
        for(int i = 0; i < node->num_TG; i++){
            output_inner[i] /= sum_prob;

        }

        ed = 0;
        for(int i = 0; i < node->num_TG; i++){
            if(ed < output_inner[i]){
                ed = output_inner[i];
                node->inference_groupid = i;
            }
        }
        gpu_imove[node->inference_groupid].occur++;
        gpu_imove[node->inference_groupid].prob += ed;

    }

    __syncthreads();
}

/* inference level three nodes by inference the movie of each image
 * the movie is created by moving the same image horizontally and vertically
 * the possible position of the movie is specified by image_x_offset and image_y_offset
 @num_images number of images
 @node_gpu pointer to nodes on gpu
 @image_gpu pointer to image information stored in gpu
 @image_x_offset_gpu length of an image that can shift vertically
 @image_y_offset_gpu length of an image that can shift horizontally
 @gpu_inference_result_lv2 pointer to start of inference result of level two nodes
*/
__global__ void kernel_inference_lv3(int num_images, node_t* node_gpu, int* images_gpu, int*
        image_x_offset_gpu, int* image_y_offset_gpu, int* gpu_infer_from_index_lv2, 
        float* gpu_inference_result_lv2, occur_prob_t* gpu_imove
        ){
    node_t* node = &(node_gpu[0]);

    __shared__ float input_inner[NUM_CHILD_LV3 * MAX_NUM_TG];
    for(int img_index = 0; img_index < num_images; img_index++){
	    for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                image_x_offset_gpu[img_index]; base_row--){
	        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH - image_y_offset_gpu[img_index]; base_column--){

                kernel_inference_node_lv3(node, 0, img_index, base_row, base_column, 1,
                        gpu_infer_from_index_lv2, image_x_offset_gpu, image_y_offset_gpu,
                        gpu_inference_result_lv2, gpu_imove, input_inner
                        );
	        }

	    }
    }
}
#endif
