#include "htm.h" 
#include "node.h"
#include "stdio.h"
#include "unistd.h"
#include <sys/time.h>

void inference(char* file_path, char* result_path){
    printf("Begin Inference\n");
    struct timeval tv_begin, tv_end;
    gettimeofday(&tv_begin, NULL);
    /* start up and read learned information */
    start_up();
    init_from_file(1);
    init_from_file(2);
    init_level_three();
    read_knowledge(&(level_three_nodes[0][0]));

    FILE* input_file;
    char* file_name = (char*)malloc(30);
    for(int m = 0; m < NUM_INFER_IMAGE; m++){
        sprintf(file_name, "%s/%d.input",file_path, m);
        input_file = fopen(file_name, "r"); 
        for(int i = 0; i < IMAGE_WIDTH; i++){
            for(int j = 0; j < IMAGE_WIDTH; j++){
                fscanf(input_file, "%d", &(pixels[i][j]));
            }
        }
        fclose(input_file);

        for(int imove = 0; imove < NUM_STEP_EYE_MOVE; imove++){
            for(int k = 0; k < IMAGE_WIDTH; k++){
                for(int j = IMAGE_WIDTH - 1; j >= 1; j--){
                    pixels[k][j] = pixels[k][j - 1];
                }
                for(int j = 0; j < 1; j++){
                    pixels[k][j] = 0;
                }
            }

            inference_level_one(0, 0);
            inference_level(2);
            inference_level_three();
        }

        infer_result[m] = infer_map[rec_index];
        //printf("%d\n", rec_index);
        rec_index = -1;
        rec_prob = 0;
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

    gettimeofday(&tv_end, NULL);
    int consumed_time = 1000000 * (tv_end.tv_sec - tv_begin.tv_sec) +
        tv_end.tv_usec - tv_begin.tv_usec;
    printf("END inferencing, consumed time is %d (ms)\n", consumed_time / 1000);
}

void train(char* input_path){
    printf("Begin Training\n");
    struct timeval tv_begin, tv_end;
    gettimeofday(&tv_begin, NULL);
    start_up();
    init_level_one();
    long num_iter = 0;

    FILE* input_file;
    char* file_name = (char*)malloc(30);
    for(int i = 0; i < NUM_IMAGE; i++){
        sprintf(file_name, "%s/%d.input", input_path, i);
        printf("learn level%d, image%d\n", 1, i);
        fflush(stdout);
        input_file = fopen(file_name, "r"); 
        for(int k = 0; k < IMAGE_WIDTH; k++){
            for(int l = 0; l < IMAGE_WIDTH; l++){
                fscanf(input_file, "%d", &(pixels[k + IMAGE_WIDTH][l +
                            IMAGE_WIDTH]));
            }
        }
        fclose(input_file);

        int dir, from_column, from_row;
        for(int base_row = IMAGE_WIDTH; base_row >
                IMAGE_WIDTH - image_x_offset[i]; base_row--){
            dir = base_row % 2;
            for(int base_column = IMAGE_WIDTH; base_column >
                    IMAGE_WIDTH - image_y_offset[i]; base_column--){
                if(dir == 0){
                    from_column = base_column;
                }else{
                    from_column = 2 * IMAGE_WIDTH -
                        base_column - image_y_offset[i] + 1;
                }
                num_iter++;
                learn_level_one(base_row, from_column);
            }
        }

        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
                image_y_offset[i]; base_column--){
            dir = base_column % 2;
            for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                    image_x_offset[i]; base_row--){
                if(dir == 0){
                    from_row = base_row;
                }else{
                    from_row = 2 * IMAGE_WIDTH - base_row -
                        image_x_offset[i] + 1;
                }
                num_iter++;
                learn_level_one(from_row, base_column);
            }
        }
    }
    printf("END level one learning\n");

    make_TG_level(1);

    write_level_knowledge(1);

    /* learn through level two to level five */
    for(int level_id = 2; level_id < 3; level_id++){
        init_level(level_id);
        for(int i = 0; i < NUM_IMAGE; i++){
            printf("learn level%d, image%d\n", level_id, i);
            fflush(stdout);
            sprintf(file_name, "%s/%d.input", input_path, i);
            input_file = fopen(file_name, "r"); 
            for(int k = 0; k < IMAGE_WIDTH; k++){
                for(int l = 0; l < IMAGE_WIDTH; l++){
                    fscanf(input_file, "%d", &(pixels[k + IMAGE_WIDTH][l +
                                IMAGE_WIDTH]));
                }
            }
            fclose(input_file);

            int dir, from_column, from_row;
            for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
            image_x_offset[i]; base_row--){
                dir = base_row % 2;
                for(int base_column = IMAGE_WIDTH; base_column >
                    IMAGE_WIDTH - image_y_offset[i]; base_column--){
                    if(dir == 0){
                        from_column = base_column;
                    }else{
                        from_column = 2 * IMAGE_WIDTH - base_column - image_y_offset[i] + 1;
                    }
                    inference_level_one(base_row, from_column);
                    for(int infer_id = 2; infer_id < level_id; infer_id++){
                        inference_level(infer_id);
                    }
                    learn_level(level_id);
                }
            }

            for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
                image_y_offset[i]; base_column--){
                dir = base_column % 2;
                for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
                image_x_offset[i]; base_row--){
                    if(dir == 0){
                        from_row = base_row;
                    }else{
                        from_row = 2 * IMAGE_WIDTH - base_row - image_x_offset[i] + 1;
                    }
                    inference_level_one(from_row, base_column);
                    for(int infer_id = 2; infer_id < level_id; infer_id++){
                        inference_level(infer_id);
                    }
                    learn_level(level_id);
                }
            }
        }

        printf("END level two learning\n");

        make_TG_level(level_id);
        write_level_knowledge(level_id);
    }

    /* learn level three*/
    init_level_three();
    int super_index =  0;

    for(int i = 0; i < NUM_IMAGE; i++){
        sprintf(file_name, "%s/%d.input", input_path, i);
        printf("learn level%d, image%d\n", 3, i);
        fflush(stdout);
        input_file = fopen(file_name, "r"); 
        for(int k = 0; k < IMAGE_WIDTH; k++){
            for(int l = 0; l < IMAGE_WIDTH; l++){
                fscanf(input_file, "%d", &(pixels[k + IMAGE_WIDTH][l +
                            IMAGE_WIDTH]));
            }
        }
        fclose(input_file);

        int dir, from_column, from_row;
        for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
            image_x_offset[i]; base_row--){
            dir = base_row % 2;
            for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
            image_y_offset[i]; base_column--){
                if(dir == 0){
                    from_column = base_column;
                }else{
                    from_column = 2 * IMAGE_WIDTH - base_column  - image_y_offset[i] + 1;
                }
                inference_level_one(base_row, from_column);
                inference_level(2);
                learn_level_three();
            }
        }

        for(int base_column = IMAGE_WIDTH; base_column > IMAGE_WIDTH -
            image_y_offset[i]; base_column--){
            dir = base_column % 2;
            for(int base_row = IMAGE_WIDTH; base_row > IMAGE_WIDTH -
            image_x_offset[i]; base_row--){
                if(dir == 0){
                    from_row = base_row;
                }else{
                    from_row = 2 * IMAGE_WIDTH - base_row  - image_x_offset[i] + 1;
                }
                inference_level_one(from_row, base_column);
                inference_level(2);
                learn_level_three();
            }
        }

        for(int j = super_index; j < level_three_nodes[0][0].num_QC; j++){
            level_three_nodes[0][0].group_id[j] = i;
        }
        level_three_nodes[0][0].num_TG++;
        super_index = level_three_nodes[0][0].num_QC;
    }

    printf("END level three learning\n");

    write_level_knowledge(3);

    gettimeofday(&tv_end, NULL);
    int consumed_time = 1000000 * (tv_end.tv_sec - tv_begin.tv_sec) +
        tv_end.tv_usec - tv_begin.tv_usec;
    printf("END training, consumed time is %d (ms)\n", consumed_time / 1000);
}

void gen_up_left(){
    FILE* output_file;
    FILE* input_file;
    char* file_name = (char*)malloc(30);
    for(int i = 32; i < 50; i++){
        sprintf(file_name, "19.tga.input", i);
        input_file = fopen(file_name, "r"); 
        if(input_file == NULL){
            continue;
        }
        if(input_file == NULL){
            continue;
        }
        for(int i = 0; i < IMAGE_WIDTH; i++){
            for(int j = 0; j < IMAGE_WIDTH; j++){
                fscanf(input_file, "%d", &(pixels[i][j]));
            }
        }
        fclose(input_file);
        for(int i = 0; i < IMAGE_WIDTH; i++){
            for(int j = 0; j < IMAGE_WIDTH; j++){
                printf("%d ", pixels[i][j]);
            }
            printf("\n");
        }

        int row = 0;
        int column = 0;
        while(true){
            int flag = 1;
            for(int i = 0; i < IMAGE_WIDTH; i++){
                if(pixels[row][i]){
                    flag = 0;
                    break;
                }
            }
            if(!flag){
                break;
            }
            row++;
        }
        while(true){
            int flag = 1;
            for(int i = 0; i < IMAGE_WIDTH; i++){
                if(pixels[i][column]){
                    flag = 0;
                    break;
                }
            }
            if(!flag){
                break;
            }
            column++;
        }

        for(int i = 0; i < IMAGE_WIDTH; i++){
            for(int j = 0; j < IMAGE_WIDTH - column; j++){
                pixels[i][j] = pixels[i][j + column];
            }
            for(int j = IMAGE_WIDTH - column; j < IMAGE_WIDTH; j++){
                pixels[i][j] = 0;
            }
        }
        for(int i = 0; i < IMAGE_WIDTH; i++){
            for(int j = 0; j < IMAGE_WIDTH - row; j++){
                pixels[j][i] = pixels[j +  row][i];
            }
            for(int j = IMAGE_WIDTH - row; j < IMAGE_WIDTH; j++){
                pixels[j][i] = 0;
            }
        }

        output_file = fopen(file_name, "w"); 
        for(int i = 0; i < IMAGE_WIDTH; i++){
            for(int j = 0; j < IMAGE_WIDTH; j++){
                fprintf(output_file, "%d ", pixels[i][j]);
            }
            fprintf(output_file, "\n");
        }
        fclose(output_file);
    }
}

void get_offset(){
    FILE* input_file;
    char* file_name = (char*)malloc(30);
    int row_offset[NUM_IMAGE];
    int column_offset[NUM_IMAGE];
    for(int img_index = 0; img_index < NUM_IMAGE; img_index++){
        sprintf(file_name, "input/%d.input", img_index);
        input_file = fopen(file_name, "r"); 
        for(int i = 0; i < IMAGE_WIDTH; i++){
            for(int j = 0; j < IMAGE_WIDTH; j++){
                fscanf(input_file, "%d", &(pixels[i][j]));
            }
        }
        fclose(input_file);
        int row = 0;
        int column = 0; 
        for(int i = 31; i >= 0; i--){
            int flag = 0;
            for(int j = 0; j < IMAGE_WIDTH; j++){
                if(pixels[i][j] != 0){
                    flag = 1;
                    break;
                }
            }
            if(flag){
                break;
            }
            row++;
        }
        row_offset[img_index] = row;

        for(int i = 31; i >=0; i--){
            int flag = 0;
            for(int j = 0; j < IMAGE_WIDTH; j++){
                if(pixels[j][i] != 0){
                    flag = 1;
                    break;
                }
            }
            if(flag){
                break;
            }
            column++;
        }
        column_offset[img_index] = column;
    }

    for(int i = 0; i < NUM_IMAGE; i++){
        printf("%d %d\n", row_offset[i], column_offset[i]);
    }
}

int main(int argc, char** argv){
    if(argc < 3){
        printf("Usage: ./htm <T|I> <input filepath> [output filepath]\n");
        return 1;
    }

    char mode = argv[1][0];
    char* file_path = argv[2];
    char* result_path;
    if(mode == 'T'){
        train(file_path);
    }else if(mode == 'I'){
        result_path = argv[3];
        inference(file_path, result_path);
    }else{
        printf("Invalide execution mode!\n");
        printf("Usage: ./htm <T|I> <filepath> [num of images]\n");
        return 1;
    }
}
