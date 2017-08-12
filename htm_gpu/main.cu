#include "htm.cu"

/* Train the HTM system
   @file_path path of pre-processed to-learn images
*/
void train(char* file_path){
    /* total trainig time including initiation and communication */
    float total_time = 0;
    dim3 dimGrid(8, 8);

    StopWatchInterface *timer = 0;
    sdkCreateTimer( &timer );

    StopWatchInterface *total_timer = 0;
    sdkCreateTimer( &total_timer );

    sdkResetTimer(&total_timer);
    sdkStartTimer( &total_timer );    

    /* init images and start up the training process */
    init_images(file_path);
    start_up(TRAINING_MODE);

    /* init level one nodes */
    init_level_one();

    sdkResetTimer(&timer);
    sdkStartTimer( &timer );    
    /* learn level one nodes on gpu */
    kernel_learn_lv1<<<dimGrid, NUM_THREADS_LEARN_LV1>>>(num_images, gpu_level_one_nodes, images_gpu,
            image_x_offset_gpu, image_y_offset_gpu, gpu_qc_same_result_lv1);
    checkCudaErrors(cudaThreadSynchronize());
    sdkStopTimer( &timer );
    printf( "END of learning level one, cosumed time: %.2f (ms)\n", sdkGetTimerValue( &timer ) );
    total_time += sdkGetTimerValue( &timer );

    /* copy learned infomation from gpu to cpu to make temporal group */
    cpFromDeviceToHost(1);

    sdkResetTimer(&timer);
    sdkStartTimer( &timer );    
    /* make temporal group for level one nodes */
    make_TG_level(1);
    sdkStopTimer( &timer );
    printf( "END of making temporal group for level one, cosumed time: %.2f (ms)\n", sdkGetTimerValue( &timer ) );
    total_time += sdkGetTimerValue( &timer );

    /* write learned level one information to file */
    write_level_knowledge(1);

    /* init inference result of level one nodes */
    init_inference_result_lv1();

    /* copy level one nodes with temporal group from cpu to gpu */
    cpFromHostToDevice(1); 

    sdkResetTimer(&timer);
    sdkStartTimer( &timer );    
    /* inference level one on gpu and store inferenced result */
    kernel_inference_lv1<<<dimGrid, NUM_THREADS_INFERENCE_LV1>>>(num_images, gpu_level_one_nodes, 
            images_gpu, image_x_offset_gpu, image_y_offset_gpu, 
            gpu_inference_result_lv1, gpu_infer_from_index_lv1,
            gpu_image_from_index, gpu_inference_max_prob_tg_lv1, TRAINING_MODE
            );
    checkCudaErrors(cudaThreadSynchronize());
    sdkStopTimer( &timer );
    printf( "END of inferencing level one, consumed time: %.2f (ms)\n", sdkGetTimerValue( &timer ) );
    total_time += sdkGetTimerValue( &timer );

    /* free space allocated for level one nodes */
    free_level_one_gpu();

    /* init level two nodes */
    init_level_two();  
    
    dim3 dimGrid2(4, 4);
    sdkResetTimer(&timer);
    sdkStartTimer( &timer );    
    /* learn level two nodes on gpu */
    kernel_learn_lv2<<<dimGrid2, NUM_THREADS_LEARN_LV2>>>(num_images, gpu_level_two_nodes, images_gpu,
            image_x_offset_gpu, image_y_offset_gpu, gpu_inference_result_lv1,
            gpu_infer_from_index_lv1, gpu_qc_same_result_lv2,
            gpu_inference_max_prob_tg_lv1, gpu_image_from_index);
    checkCudaErrors(cudaThreadSynchronize());
    sdkStopTimer( &timer );
    printf("END of learning level two, consumed time: %.2f (ms)\n",
            sdkGetTimerValue( &timer ));
    total_time += sdkGetTimerValue( &timer );

    /* copy level two nodes with temporal group from cpu to gpu */
    cpFromDeviceToHost(2); 

    sdkResetTimer(&timer);
    sdkStartTimer( &timer );    
    /* make temporal group for level two nodes */
    make_TG_level(2);
    sdkStopTimer( &timer );
    printf( "END of making temporal group for level two, cosumed time: %.2f (ms)\n", sdkGetTimerValue( &timer ) );
    total_time += sdkGetTimerValue( &timer );

    /* write learned level two information to file */
    write_level_knowledge(2);

    /* copy level two nodes with temporal group from cpu to gpu */
    cpFromHostToDevice(2); 

    /* free gpu space allocated for time adjacent matrix of level two nodes */
    free_level_two_matrix_gpu(); 

    /* init inference result for level two nodes */
    init_inference_result_lv2(TRAINING_MODE);

    sdkResetTimer(&timer);
    sdkStartTimer( &timer );    
    /* inference level two on gpu and store inferenced result */
    kernel_inference_lv2<<<dimGrid2, NUM_THREADS_INFERENCE_LV2>>>(num_images, gpu_level_two_nodes, images_gpu,
            image_x_offset_gpu, image_y_offset_gpu, gpu_inference_result_lv1, gpu_inference_result_lv2,
            gpu_infer_from_index_lv1, 
            gpu_infer_from_index_lv2, gpu_inference_max_prob_tg_lv2, gpu_image_from_index_lv2, TRAINING_MODE);
    checkCudaErrors(cudaThreadSynchronize());
    sdkStopTimer( &timer );
    printf( "END of inferencing level two, consumed time: %.2f (ms)\n", sdkGetTimerValue( &timer ) );
    total_time += sdkGetTimerValue( &timer );

    /* init level three node */
    init_level_three();

    dim3 dimGrid3(1, 1);
    sdkResetTimer(&timer);
    sdkStartTimer( &timer );    
    /* learn level three nodes on gpu */
    kernel_learn_lv3<<<dimGrid3, NUM_THREADS_LEARN_LV3>>>(num_images, gpu_level_three_nodes, images_gpu,
            image_x_offset_gpu, image_y_offset_gpu, gpu_infer_from_index_lv2,
            gpu_image_from_index_lv2, gpu_inference_max_prob_tg_lv2,
            gpu_qc_same_result_lv3);
    checkCudaErrors(cudaThreadSynchronize());
    sdkStopTimer( &timer );
    printf("END of learning level three, consumed time: %.2f (ms)\n",
            sdkGetTimerValue( &timer ));
    total_time += sdkGetTimerValue( &timer );

    /* copy learned node information from gpu to cpu */
    cpFromDeviceToHost(3);

    /* write learned level three node to file */
    write_level_knowledge(3);

    sdkStopTimer( &total_timer );
    printf( "total time: %.2f (ms)\n", sdkGetTimerValue( &total_timer ) );
    printf( "total computing time: %.2f (ms)\n", total_time);
    
    sdkDeleteTimer( &timer );
}

/* inference a specified image */
void do_inference(occur_prob_t* gpu_imove){
    dim3 dimGrid(8, 8);
    /* infernce level onde nodes */
    kernel_inference_lv1<<<dimGrid, NUM_THREADS_INFERENCE_LV1>>>(num_images, gpu_level_one_nodes,
            images_gpu, image_x_offset_gpu, image_y_offset_gpu, 
            gpu_inference_result_lv1, gpu_infer_from_index_lv1,
            gpu_image_from_index, gpu_inference_max_prob_tg_lv1, INFERENCE_MODE);

    checkCudaErrors(cudaThreadSynchronize());

    dim3 dimGrid2(4, 4);
    /* inference level two nodes */
    kernel_inference_lv2<<<dimGrid2, NUM_THREADS_INFERENCE_LV2>>>(num_images, gpu_level_two_nodes, images_gpu,
            image_x_offset_gpu, image_y_offset_gpu, gpu_inference_result_lv1, gpu_inference_result_lv2,
            gpu_infer_from_index_lv1, 
            gpu_infer_from_index_lv2, gpu_inference_max_prob_tg_lv2, gpu_image_from_index_lv2, INFERENCE_MODE);
    checkCudaErrors(cudaThreadSynchronize());

    dim3 dimGrid3(1, 1);
    /* inference level three nodes */
    kernel_inference_lv3<<<dimGrid3,
        NUM_THREADS_INFERENCE_LV3>>>(num_images, gpu_level_three_nodes, images_gpu,
                image_x_offset_gpu, image_y_offset_gpu,
                gpu_infer_from_index_lv2, gpu_inference_result_lv2, gpu_imove
                );
    checkCudaErrors(cudaThreadSynchronize());
}

/* inference images
   @file_path path of pre-processed images to inference
*/
void inference(char* file_path, char* result_path){
    /* create timer */
    StopWatchInterface *total_timer = 0;
    sdkCreateTimer( &total_timer );

    sdkResetTimer(&total_timer);
    sdkStartTimer( &total_timer );    
    printf("Begin Inferencing\n");

    /* allocate space for to-inference image */
    images = (int*)malloc(ST_IMAGE_WIDTH * ST_IMAGE_WIDTH * num_images *
            sizeof(int));
    memset(images, 0, ST_IMAGE_WIDTH * ST_IMAGE_WIDTH * num_images *
            sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&images_gpu, ST_IMAGE_WIDTH *
                ST_IMAGE_WIDTH * num_images * sizeof(int)));

    /* structure for eye-move recognition, stores inference result */
    occur_prob_t* cpu_imove;
    occur_prob_t* gpu_imove;
    cpu_imove = (occur_prob_t*)malloc(NUM_TG_TOP * sizeof(occur_prob_t));
    memset(cpu_imove, 0, NUM_TG_TOP * sizeof(occur_prob_t));
    checkCudaErrors(cudaMalloc((void**)&gpu_imove, NUM_TG_TOP *
                sizeof(occur_prob_t)));
    checkCudaErrors(cudaMemcpy(gpu_imove, cpu_imove, NUM_TG_TOP *
                sizeof(occur_prob_t), cudaMemcpyHostToDevice));

    /* start up inference process */
    start_up(INFERENCE_MODE);
    /* read learned information from file */
    init_level_one_from_file();
    init_level_two_from_file();  
    init_level_three_from_file();

    /* inference each of the to-infer images */
    for(int i = 0; i < NUM_INFER_IMAGE; i++){
        //printf("image %d\n", i);
        memset(cpu_imove, 0, NUM_TG_TOP * sizeof(occur_prob_t));
        checkCudaErrors(cudaMemcpy(gpu_imove, cpu_imove, NUM_TG_TOP *
                sizeof(occur_prob_t), cudaMemcpyHostToDevice));

        /* init inference results of level one and level two */
        init_inference_result_lv1();
        init_inference_result_lv2(INFERENCE_MODE);
        
        /* init image */
        init_images_infer_mode(file_path, i);

        /* inference the image */
        do_inference(gpu_imove);

        /* copy the inference result from gpu to cpu */
        checkCudaErrors(cudaMemcpy(cpu_imove, gpu_imove, NUM_TG_TOP *
                sizeof(occur_prob_t), cudaMemcpyDeviceToHost));

        /* get the final inference result */
        int rec_index = -1;
        occur_prob_t tmp_index;
        tmp_index.occur = -1;
        tmp_index.prob = 0;
        for(int j = 0; j < NUM_TG_TOP; j++){
            if(cpu_imove[j].occur > tmp_index.occur || (cpu_imove[j].occur ==
                        tmp_index.occur && cpu_imove[j].prob > tmp_index.prob)){
                tmp_index.occur = cpu_imove[j].occur;
                tmp_index.prob = cpu_imove[j].prob;
                rec_index = j;
            }
        }
        infer_tg_index[i] = rec_index;
        infer_result[i] = infer_map[rec_index];

        //printf("%d\n", rec_index);

        checkCudaErrors(cudaFree(gpu_inference_result_lv1));
        checkCudaErrors(cudaFree(gpu_inference_result_lv2));
    }

    int infer_result_pointed[NUM_INFER_IMAGE];
    for(int i = 0; i < NUM_INFER_IMAGE; i++){
        infer_result_pointed[i] = infer_result[i];
    }
    char* ori_order_file_name = (char*)malloc(1);
    sprintf(ori_order_file_name, "%s_ori_order", result_path);

    FILE* result_file = fopen(ori_order_file_name, "w+");
    FILE* result_file_pointed = fopen(result_path, "w+");
    for(int i = 0;i < 25; i++){
        if(infer_result[i] < 10){
            fprintf(result_file, "%d\n", infer_result[i]);
            fprintf(result_file_pointed, "%d\n", infer_result_pointed[i]);
        }else if(infer_result[i] < 1000){
            fprintf(result_file, "%c\n", infer_result[i]);
            fprintf(result_file_pointed, "%c\n", infer_result_pointed[i]);
        }else{
            switch(infer_result[i]){
                case 10001:
                    fprintf(result_file, "[le]\n");
                    fprintf(result_file_pointed, "[le]\n");
                    break;
                case 10002:
                    fprintf(result_file, "[kai]\n");
                    fprintf(result_file_pointed, "[kai]\n");
                    break;
                case 10003:
                    fprintf(result_file, "[la]\n");
                    fprintf(result_file_pointed, "[la]\n");
                    break;
                case 10004:
                    fprintf(result_file, "[si]\n");
                    fprintf(result_file_pointed, "[si]\n");
                    break;
                case 10005:
                    fprintf(result_file, "[pu]\n");
                    fprintf(result_file_pointed, "[pu]\n");
                    break;
                case 10006:
                    fprintf(result_file, "[te]\n");
                    fprintf(result_file_pointed, "[te]\n");
                    break;
                case 10007:
                    fprintf(result_file, "[mi]\n");
                    fprintf(result_file_pointed, "[mi]\n");
                    break;
                case 10008:
                    fprintf(result_file, "[fei]\n");
                    fprintf(result_file_pointed, "[fei]\n");
                    break;
            }
        }
    }

    for(int i = 25;i < 75; i++){
        if(infer_result[i] < 10){
            fprintf(result_file, "%d", infer_result[i]);
        }else if(infer_result[i] < 10000){
            fprintf(result_file, "%c", infer_result[i]);
        }else{
            switch(infer_result[i]){
                case 10001:
                    fprintf(result_file, "[le]");
                    break;
                case 10002:
                    fprintf(result_file, "[kai]");
                    break;
                case 10003:
                    fprintf(result_file, "[la]");
                    break;
                case 10004:
                    fprintf(result_file, "[si]");
                    break;
                case 10005:
                    fprintf(result_file, "[pu]");
                    break;
                case 10006:
                    fprintf(result_file, "[te]");
                    break;
                case 10007:
                    fprintf(result_file, "[mi]");
                    break;
                case 10008:
                    fprintf(result_file, "[fei]");
                    break;
            }
        }
        if(i % 2 == 0){

            /* sort to increase order */
            int index = i - 1;
            if(infer_result_pointed[i] < infer_result_pointed[i - 1]){
                int tmp_inner = infer_result_pointed[i];
                infer_result_pointed[i] = infer_result_pointed[i - 1];
                infer_result_pointed[i - 1] = tmp_inner;
            }
            for(;index <= i; index++){
                if(infer_result_pointed[index] < 10){
                    fprintf(result_file_pointed, "%d",
                            infer_result_pointed[index]);
                }else if(infer_result_pointed[index] < 10000){
                    fprintf(result_file_pointed, "%c",
                            infer_result_pointed[index]);
                }else{
                    switch(infer_result_pointed[index]){
                        case 10001:
                            fprintf(result_file_pointed, "[le]");
                            break;
                        case 10002:
                            fprintf(result_file_pointed, "[kai]");
                            break;
                        case 10003:
                            fprintf(result_file_pointed, "[la]");
                            break;
                        case 10004:
                            fprintf(result_file_pointed, "[si]");
                            break;
                        case 10005:
                            fprintf(result_file_pointed, "[pu]");
                            break;
                        case 10006:
                            fprintf(result_file_pointed, "[te]");
                            break;
                        case 10007:
                            fprintf(result_file_pointed, "[mi]");
                            break;
                        case 10008:
                            fprintf(result_file_pointed, "[fei]");
                            break;
                    }
                }
            }
            fprintf(result_file, "\n");
            fprintf(result_file_pointed, "\n");
        }
    }

    for(int i = 75;i < 150; i++){
        if(infer_result[i] < 10){
            fprintf(result_file, "%d", infer_result[i]);
        }else if(infer_result[i] < 10000){
            fprintf(result_file, "%c", infer_result[i]);
        }else{
            switch(infer_result[i]){
                case 10001:
                    fprintf(result_file, "[le]");
                    break;
                case 10002:
                    fprintf(result_file, "[kai]");
                    break;
                case 10003:
                    fprintf(result_file, "[la]");
                    break;
                case 10004:
                    fprintf(result_file, "[si]");
                    break;
                case 10005:
                    fprintf(result_file, "[pu]");
                    break;
                case 10006:
                    fprintf(result_file, "[te]");
                    break;
                case 10007:
                    fprintf(result_file, "[mi]");
                    break;
                case 10008:
                    fprintf(result_file, "[fei]");
                    break;
            }
        }
        if(i % 3 == 2){

            int index = i - 2;
            for(int j = i; j >= index; j--){
                int max_index = j;
                int max_value = infer_result_pointed[j];
                for(int k = index; k <= j; k++){
                    if(max_value < infer_result_pointed[k]){
                        max_value = infer_result_pointed[k];
                        max_index = k;
                    }
                }
                if(max_index != j){
                    infer_result_pointed[max_index] = infer_result_pointed[j];
                    infer_result_pointed[j] = max_value;
                }
            }

            for(int j = index; j <= i; j++){
                if(infer_result_pointed[j] < 10){
                    fprintf(result_file_pointed, "%d", infer_result_pointed[j]);
                }else if(infer_result_pointed[j] < 10000){
                    fprintf(result_file_pointed, "%c", infer_result_pointed[j]);
                }else{
                    switch(infer_result_pointed[j]){
                        case 10001:
                            fprintf(result_file_pointed, "[le]");
                            break;
                        case 10002:
                            fprintf(result_file_pointed, "[kai]");
                            break;
                        case 10003:
                            fprintf(result_file_pointed, "[la]");
                            break;
                        case 10004:
                            fprintf(result_file_pointed, "[si]");
                            break;
                        case 10005:
                            fprintf(result_file_pointed, "[pu]");
                            break;
                        case 10006:
                            fprintf(result_file_pointed, "[te]");
                            break;
                        case 10007:
                            fprintf(result_file_pointed, "[mi]");
                            break;
                        case 10008:
                            fprintf(result_file_pointed, "[fei]");
                            break;
                    }
                }
            }

            fprintf(result_file, "\n");
            fprintf(result_file_pointed, "\n");
        }
    }

    for(int i = 150;i < 250; i++){
        if(infer_result[i] < 10){
            fprintf(result_file, "%d", infer_result[i]);
        }else if(infer_result[i] < 10000){
            fprintf(result_file, "%c", infer_result[i]);
        }else{
            switch(infer_result[i]){
                case 10001:
                    fprintf(result_file, "[le]");
                    break;
                case 10002:
                    fprintf(result_file, "[kai]");
                    break;
                case 10003:
                    fprintf(result_file, "[la]");
                    break;
                case 10004:
                    fprintf(result_file, "[si]");
                    break;
                case 10005:
                    fprintf(result_file, "[pu]");
                    break;
                case 10006:
                    fprintf(result_file, "[te]");
                    break;
                case 10007:
                    fprintf(result_file, "[mi]");
                    break;
                case 10008:
                    fprintf(result_file, "[fei]");
                    break;
            }
        }
        if(i % 4 == 1){

            int index = i - 3;
            for(int j = i; j >= index; j--){
                int max_index = j;
                int max_value = infer_result_pointed[j];
                for(int k = index; k <= j; k++){
                    if(max_value < infer_result_pointed[k]){
                        max_value = infer_result_pointed[k];
                        max_index = k;
                    }
                }
                if(max_index != j){
                    infer_result_pointed[max_index] = infer_result_pointed[j];
                    infer_result_pointed[j] = max_value;
                }
            }

            for(int j = index; j <= i; j++){
                if(infer_result_pointed[j] < 10){
                    fprintf(result_file_pointed, "%d", infer_result_pointed[j]);
                }else if(infer_result_pointed[j] < 10000){
                    fprintf(result_file_pointed, "%c", infer_result_pointed[j]);
                }else{
                    switch(infer_result_pointed[j]){
                        case 10001:
                            fprintf(result_file_pointed, "[le]");
                            break;
                        case 10002:
                            fprintf(result_file_pointed, "[kai]");
                            break;
                        case 10003:
                            fprintf(result_file_pointed, "[la]");
                            break;
                        case 10004:
                            fprintf(result_file_pointed, "[si]");
                            break;
                        case 10005:
                            fprintf(result_file_pointed, "[pu]");
                            break;
                        case 10006:
                            fprintf(result_file_pointed, "[te]");
                            break;
                        case 10007:
                            fprintf(result_file_pointed, "[mi]");
                            break;
                        case 10008:
                            fprintf(result_file_pointed, "[fei]");
                            break;
                    }
                }
            }

            fprintf(result_file, "\n");
            fprintf(result_file_pointed, "\n");
        }
    }
    fclose(result_file);
    fclose(result_file_pointed);

    sdkStopTimer( &total_timer );
    printf( "END Inferencing, total time: %.2f (ms)\n", sdkGetTimerValue( &total_timer ) );

    /* read answer file */
    char* answer_file_name = (char*)malloc(30);
    sprintf(answer_file_name, "%s/answer.txt", file_path);
    FILE* answer_file = fopen(answer_file_name, "r");
    if(answer_file){
        int answer[NUM_INFER_IMAGE];
        for(int i = 0; i < NUM_INFER_IMAGE; i++){
            fscanf(answer_file, "%d\n", &(answer[i]));
        }
        int correct_count = 0;
        for(int i = 0; i < NUM_INFER_IMAGE; i++){
            if(answer_map[infer_tg_index[i]] == answer[i]){
                correct_count++;
            }
        }
        printf("Recognize accuracy is %.2f\n", (1.0 * correct_count) / NUM_INFER_IMAGE);
    }else{
        printf("Success inferencing\n");
    }
}

/* entrance of the HTM program
   @arg0 ./htm
   @arg1 run mode, can be T or M
   @arg2 file path of input images
   @arg3 file path of inference result
*/
int main(int argc, char** argv){
    if(argc < 3){
        printf("Usage: ./htm <T|I> <input filepath> [output filepath]\n");
        return 1;
    }

    shrQAStart(argc, argv);

    deviceInit(argc, argv);

    char mode = argv[1][0];
    char* file_path = argv[2];
    char* result_path;
    if(mode == 'T'){
        num_images = 58;
        train(file_path);
        sigma_square = 0.1;
    }else if(mode == 'I'){
        num_images = 1;
        result_path = argv[3];
        inference(file_path, result_path);
        sigma_square = 1.5;
    }else{
        printf("Invalide execution mode!\n");
        printf("Usage: ./htm <T|I> <filepath> [num of images]\n");
        return 1;
    }
    deviceExit(argc, argv);
}
