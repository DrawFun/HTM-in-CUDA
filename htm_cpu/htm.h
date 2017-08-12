#ifndef HTM_HEADER
#define HTM_HEADER
#include "node.h"

#define NUM_LEVELS 3
#define LEVEL1_WIDTH 8
#define LEVEL2_WIDTH 4
#define LEVEL3_WIDTH 1

/* width of image */
#define IMAGE_WIDTH 32

/* width of child block */
#define CHILD_BLOCK_WIDTH 2

/* image to be learned on */
int pixels[IMAGE_WIDTH * 2][IMAGE_WIDTH * 2];

/* number of images to learn */
#define NUM_IMAGE 58

/* level one nodes */
int** max_qc_profile_lv1;
int** max_qc_profile_lv2;
int** max_qc_profile_lv3;
node_t** level_one_nodes;
node_t master_node;

int* image_x_offset;
int* image_y_offset;

/* level two nodes */
//node_t level_two_nodes[LEVEL2_WIDTH][LEVEL2_WIDTH];
node_t** level_two_nodes;

/* level three nodes */
//node_t level_three_nodes[LEVEL3_WIDTH][LEVEL3_WIDTH];
node_t** level_three_nodes;

/* level four nodes */
//node_t level_four_nodes[LEVEL4_WIDTH][LEVEL4_WIDTH];
node_t** level_four_nodes;

/* level five nodes */
//node_t level_five_nodes[LEVEL5_WIDTH][LEVEL5_WIDTH];
node_t** level_five_nodes;

/* level six nodes */
node_t** level_six_node;

/*init master node, used in Master MOde, not used here */
void init_master();

/*get profiled max QC*/
void start_up();

/* initialize level one nodes */
void init_level_one();

/* learn level one nodes */
void learn_level_one(int base_row, int base_column);

/* inference level one nodes */
void inference_level_one(int base_row, int base_column);

/* initialize level nodes, actually only used to init level two */
void init_level(int level);

/* learn level nodes, actually only used to learn level two */
void learn_level(int level);

/* inference level nodes, actually only used to inference level two */
void inference_level(int level);

/* initialize level nodes other than level one */
void init_level_three();

/* learn levle three nodes */
void learn_level_three();

/* inference level three nodes */
void inference_level_three();

/* make temporal groups for specified level */
void make_TG_level(int level);

void start_up(){
    max_qc_profile_lv1 = (int**)malloc(LEVEL1_WIDTH * sizeof(int*));
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        max_qc_profile_lv1[i] = (int*)malloc(LEVEL1_WIDTH * sizeof(int));
    }
    FILE* file = fopen("start_up/level_one_qc", "r");
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
    file = fopen("start_up/level_two_qc", "r");
    for(int i = 0; i < LEVEL2_WIDTH; i++){
        for(int j = 0; j < LEVEL2_WIDTH; j++){
            fscanf(file, "%d\n", &(max_qc_profile_lv2[i][j]));
        }
    }
    fclose(file);

    max_qc_profile_lv3 = (int**)malloc(LEVEL3_WIDTH * sizeof(int*));
    for(int i = 0; i < LEVEL3_WIDTH; i++){
        max_qc_profile_lv3[i] = (int*)malloc(LEVEL3_WIDTH * sizeof(int));
    }
    file = fopen("start_up/level_three_qc", "r");
    for(int i = 0; i < LEVEL3_WIDTH; i++){
        for(int j = 0; j < LEVEL3_WIDTH; j++){
            fscanf(file, "%d\n", &(max_qc_profile_lv3[i][j]));
            max_qc_profile_lv3[i][j] = 1000;
        }
    }
    fclose(file);

    file = fopen("start_up/image_offset", "r");
    image_x_offset = (int*)malloc(NUM_IMAGE * sizeof(int));
    image_y_offset = (int*)malloc(NUM_IMAGE * sizeof(int));
    for(int i = 0; i < NUM_IMAGE; i++){
        fscanf(file, "%d", &(image_x_offset[i]));
        fscanf(file, "%d", &(image_y_offset[i]));
    }
    fclose(file);
}

/* Master mode, not used here */
void init_master(){
        master_node.num_max_QC = MAX_NUM_QC_LEVEL1;
        master_node.QC = (int**)malloc(MAX_NUM_QC_LEVEL1 *
                sizeof(int*));
        for(int i = 0; i < MAX_NUM_QC_LEVEL1; i++){
            master_node.QC[i] = (int*)malloc(NUM_CHILD_LV1 *
                sizeof(int));
            for(int j = 0; j < NUM_CHILD_LV1; j++){
                master_node.QC[i][j] = -1;
            }
        }

        master_node.ED = (float*)malloc(MAX_NUM_QC_LEVEL1 * sizeof(float));
        master_node.time_adj_matrix =
            (int**)malloc(MAX_NUM_QC_LEVEL1 * sizeof(int*));
        for(int k = 0; k < MAX_NUM_QC_LEVEL1; k++){
            master_node.time_adj_matrix[k] =
                (int*)malloc(MAX_NUM_QC_LEVEL1 * sizeof(int));
        }

        master_node.mode = TRAINIGN_MODE;
        master_node.num_child = NUM_CHILD_LV1;
        master_node.num_child_group = (int*)malloc(NUM_CHILD_LV1 *
                sizeof(int));
        for(int k = 0; k < NUM_CHILD_LV1; k++){
            master_node.num_child_group[k] = 2;
        }

        master_node.input = (float**)malloc(NUM_CHILD_LV1 *
                sizeof(float*));
        for(int k = 0; k < NUM_CHILD_LV1; k++){
            master_node.input[k] = (float*)malloc(2 *
                    sizeof(float));
        }

        master_node.num_QC = 0;
        for(int k = 0; k < MAX_NUM_QC_LEVEL1; k++){
            master_node.ED[k] = 0;
        }
        master_node.last_QC = -1;
        for(int k = 0; k < MAX_NUM_QC_LEVEL1; k++){
            for(int l = 0; l < MAX_NUM_QC_LEVEL1; l++){
                master_node.time_adj_matrix[k][l] = 0;
            }
        }
        master_node.num_TG = 0;
        for(int k = 0; k < MAX_NUM_QC; k++){
            master_node.group_id[k] = -1;
        }
        for(int k = 0; k < MAX_NUM_TG; k++){
            master_node.output[k] = 0;
        }
}

/* init level one nodes */
void init_level_one(){
    level_one_nodes = (node_t**)malloc(LEVEL1_WIDTH * sizeof(node_t*));
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        level_one_nodes[i] = (node_t*)malloc(LEVEL1_WIDTH * sizeof(node_t));
    }
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        for(int j = 0; j < LEVEL1_WIDTH; j++){
            level_one_nodes[i][j].id.level = 1;
            level_one_nodes[i][j].id.row = i;
            level_one_nodes[i][j].id.column = j;
            int max_qc = max_qc_profile_lv1[i][j];
            level_one_nodes[i][j].num_max_QC = max_qc; 
            
            level_one_nodes[i][j].QC = (int**)malloc(max_qc *
                    sizeof(int*));
            for(int k = 0; k < max_qc; k++){
                level_one_nodes[i][j].QC[k] = (int*)malloc(NUM_CHILD_LV1 *
                    sizeof(int));
                for(int l = 0; l < NUM_CHILD_LV1; l++){
                    level_one_nodes[i][j].QC[k][l] = -1;
                }
            }

            level_one_nodes[i][j].ED = (float*)malloc(max_qc * sizeof(float));
            level_one_nodes[i][j].time_adj_matrix =
                (int**)malloc(max_qc * sizeof(int*));
            for(int k = 0; k < max_qc; k++){
                level_one_nodes[i][j].time_adj_matrix[k] =
                    (int*)malloc(max_qc * sizeof(int));
            }

            level_one_nodes[i][j].mode = TRAINIGN_MODE;
            level_one_nodes[i][j].num_child = NUM_CHILD_LV1;
            level_one_nodes[i][j].num_child_group = (int*)malloc(NUM_CHILD_LV1 *
                    sizeof(int));
            for(int k = 0; k < NUM_CHILD_LV1; k++){
                level_one_nodes[i][j].num_child_group[k] = 2;
            }

            level_one_nodes[i][j].input = (float**)malloc(NUM_CHILD_LV1 *
                    sizeof(float*));
            for(int k = 0; k < NUM_CHILD_LV1; k++){
                level_one_nodes[i][j].input[k] = (float*)malloc(2 *
                        sizeof(float));
            }

            level_one_nodes[i][j].num_QC = 0;
            for(int k = 0; k < max_qc; k++){
                level_one_nodes[i][j].ED[k] = 0;
            }
            level_one_nodes[i][j].last_QC = -1;
            for(int k = 0; k < max_qc; k++){
                for(int l = 0; l < max_qc; l++){
                    level_one_nodes[i][j].time_adj_matrix[k][l] = 0;
                }
            }
            level_one_nodes[i][j].num_TG = 0;

            for(int k = 0; k < MAX_NUM_QC; k++){
                level_one_nodes[i][j].group_id[k] = -1;
            }
            for(int k = 0; k < MAX_NUM_TG; k++){
                level_one_nodes[i][j].output[k] = 0;
            }
        }
    }
}

/* init nodes for specified level, actually only used for level two here */
void init_level(int level){
    node_t** target_level_nodes;
    node_t** child_level_nodes;
    int** max_qc_profile;
    int num_level_width;
    int num_max_qc;
    int num_child;
    switch(level){
        case 2:
            child_level_nodes = level_one_nodes;
            num_level_width = LEVEL2_WIDTH;
            max_qc_profile = max_qc_profile_lv2;
            num_child = NUM_CHILD_LV2;
            break;
        case 3:
            child_level_nodes = level_two_nodes;
            num_level_width = LEVEL3_WIDTH;
            max_qc_profile = max_qc_profile_lv3;
            num_child = NUM_CHILD_LV3;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }
    target_level_nodes = (node_t**)malloc(num_level_width * sizeof(node_t*));
    for(int i = 0; i < num_level_width; i++){
        target_level_nodes[i] = (node_t*)malloc(num_level_width * sizeof(node_t));
    }
    
    for(int i = 0; i < num_level_width; i++){
        for(int j = 0; j < num_level_width; j++){
            target_level_nodes[i][j].id.level = level;
            target_level_nodes[i][j].id.row = i;
            target_level_nodes[i][j].id.column = j;
            num_max_qc = max_qc_profile[i][j];
            target_level_nodes[i][j].num_max_QC = num_max_qc;
            target_level_nodes[i][j].QC = (int**)malloc(num_max_qc *
                    sizeof(int*));
            for(int k = 0; k < num_max_qc; k++){
                target_level_nodes[i][j].QC[k] = (int*)malloc(num_child *
                        sizeof(int));
                for(int l = 0; l < num_child; l++){
                    target_level_nodes[i][j].QC[k][l] = -1;
                }
            }
            target_level_nodes[i][j].ED = (float*)malloc(num_max_qc* sizeof(float));
            target_level_nodes[i][j].time_adj_matrix =
                (int**)malloc(num_max_qc * sizeof(int*));
            for(int k = 0; k < num_max_qc; k++){
                target_level_nodes[i][j].time_adj_matrix[k] =
                    (int*)malloc(num_max_qc * sizeof(int));
            }
            
            target_level_nodes[i][j].mode = TRAINIGN_MODE;
            target_level_nodes[i][j].num_child = num_child;
            target_level_nodes[i][j].num_child_group = (int*)malloc(num_child *
                    sizeof(int));
            
            /* init child_group, the current node's coodinate is (i, j), then
             * its four childs' coordinates are:
             * (i * CHILD_BLOCK_WIDTH, j * CHILD_BLOCK_WIDTH)
             * (i * CHILD_BLOCK_WIDTH, j * CHILD_BLOCK_WIDTH + 1)
             * (i * CHILD_BLOCK_WIDTH + 1, j * CHILD_BLOCK_WIDTH)
             * (i * CHILD_BLOCK_WIDTH + 1, j * CHILD_BLOCK_WIDTH + 1)
             */
            target_level_nodes[i][j].num_child_group[0] = child_level_nodes[i *
                CHILD_BLOCK_WIDTH][j * CHILD_BLOCK_WIDTH].num_TG;
            target_level_nodes[i][j].num_child_group[1] = child_level_nodes[i *
                CHILD_BLOCK_WIDTH][j * CHILD_BLOCK_WIDTH + 1].num_TG;
            target_level_nodes[i][j].num_child_group[2] = child_level_nodes[i *
                CHILD_BLOCK_WIDTH + 1][j * CHILD_BLOCK_WIDTH].num_TG;
            target_level_nodes[i][j].num_child_group[3] = child_level_nodes[i *
                CHILD_BLOCK_WIDTH+ 1][j * CHILD_BLOCK_WIDTH + 1].num_TG;

            target_level_nodes[i][j].input = (float**)malloc(num_child *
                    sizeof(float*));
            for(int k = 0; k < num_child; k++){
                target_level_nodes[i][j].input[k] =
                    (float*)malloc(target_level_nodes[i][j].num_child_group[k] *
                            sizeof(float));
            }

            target_level_nodes[i][j].num_QC = 0;
            for(int k = 0; k < num_max_qc; k++){
                target_level_nodes[i][j].ED[k] = 0;
            }
            target_level_nodes[i][j].last_QC = -1;
            for(int k = 0; k < num_max_qc; k++){
                for(int l = 0; l < num_max_qc; l++){
                    target_level_nodes[i][j].time_adj_matrix[k][l] = 0;
                }
            }
            target_level_nodes[i][j].num_TG = 0;
            for(int k = 0; k < MAX_NUM_QC; k++){
                target_level_nodes[i][j].group_id[k] = -1;
            }
            for(int k = 0; k < MAX_NUM_TG; k++){
                target_level_nodes[i][j].output[k] = 0;
            }
        }
    }

    switch(level){
        case 2:
            level_two_nodes = target_level_nodes;
            break;
        case 3:
            level_three_nodes = target_level_nodes;
            break;
        case 4:
            level_four_nodes = target_level_nodes;
            break;
        case 5:
            level_five_nodes = target_level_nodes;
            break;
        case 6:
            level_six_node = target_level_nodes;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }
}

/* init nodes in level three */
void init_level_three(){
    level_three_nodes = (node_t**)malloc(LEVEL3_WIDTH * sizeof(node_t*));
    for(int i = 0; i < LEVEL3_WIDTH; i++){
        level_three_nodes[i] = (node_t*)malloc(LEVEL3_WIDTH * sizeof(node_t));
    }
    for(int i = 0; i < LEVEL3_WIDTH; i++){
        for(int j = 0; j < LEVEL3_WIDTH; j++){
            level_three_nodes[i][j].id.level = 3;
            level_three_nodes[i][j].id.row = i;
            level_three_nodes[i][j].id.column = j;
            int max_qc = MAX_NUM_QC_LEVEL3; 
            level_three_nodes[i][j].num_max_QC = max_qc; 
            
            level_three_nodes[i][j].QC = (int**)malloc(max_qc *
                    sizeof(int*));
            for(int k = 0; k < max_qc; k++){
                level_three_nodes[i][j].QC[k] = (int*)malloc(NUM_CHILD_LV3 *
                    sizeof(int));
                for(int l = 0; l < NUM_CHILD_LV3; l++){
                    level_three_nodes[i][j].QC[k][l] = -1;
                }
            }

            level_three_nodes[i][j].ED = (float*)malloc(max_qc * sizeof(float));

            //top node learn in supervised mode, no need for time_adj_matrix

            level_three_nodes[i][j].mode = TRAINIGN_MODE;
            level_three_nodes[i][j].num_child = NUM_CHILD_LV3;
            level_three_nodes[i][j].num_child_group = (int*)malloc(NUM_CHILD_LV3 *
                    sizeof(int));
            for(int k = 0; k < 4; k++){
                for(int l = 0; l < 4; l++){
                    level_three_nodes[i][j].num_child_group[k * 4 + l] =
                        level_two_nodes[k][l].num_TG;
                }
            }

            level_three_nodes[i][j].input = (float**)malloc(NUM_CHILD_LV3 *
                    sizeof(float*));
            for(int k = 0; k < NUM_CHILD_LV3; k++){
                level_three_nodes[i][j].input[k] =
                    (float*)malloc(level_three_nodes[i][j].num_child_group[k] *
                        sizeof(float));
            }

            level_three_nodes[i][j].num_QC = 0;
            for(int k = 0; k < max_qc; k++){
                level_three_nodes[i][j].ED[k] = 0;
            }
            level_three_nodes[i][j].last_QC = -1;
            level_three_nodes[i][j].num_TG = 0;

            for(int k = 0; k < MAX_NUM_QC; k++){
                level_three_nodes[i][j].group_id[k] = -1;
            }
            for(int k = 0; k < MAX_NUM_TG; k++){
                level_three_nodes[i][j].output[k] = 0;
            }
        }
    }
}

/* init nodes from learned file for specified level */
void init_from_file(int level){
    node_t** target_level_nodes;
    node_t** child_level_nodes;
    int num_level_width;
    int num_max_qc;
    int num_child;
    switch(level){
        case 1:
            num_level_width = LEVEL1_WIDTH;
            num_max_qc = MAX_NUM_QC_LEVEL1;
            num_child = NUM_CHILD_LV1;
            break;
        case 2:
            num_level_width = LEVEL2_WIDTH;
            num_max_qc = MAX_NUM_QC_LEVEL2;
            num_child = NUM_CHILD_LV2;
            child_level_nodes = level_one_nodes;
            break;
        case 3:
            num_level_width = LEVEL3_WIDTH;
            num_max_qc = MAX_NUM_QC_LEVEL3;
            num_child = NUM_CHILD_LV3;
            child_level_nodes = level_two_nodes;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }

    target_level_nodes = (node_t**)malloc(num_level_width * sizeof(node_t*));
    for(int i = 0; i < num_level_width; i++){
        target_level_nodes[i] = (node_t*)malloc(num_level_width * sizeof(node_t));
    }
    for(int i = 0; i < num_level_width; i++){
        for(int j = 0; j < num_level_width; j++){
            target_level_nodes[i][j].id.level = level;
            target_level_nodes[i][j].id.row = i;
            target_level_nodes[i][j].id.column = j;
            target_level_nodes[i][j].num_max_QC = num_max_qc;

            target_level_nodes[i][j].num_child = num_child;
            target_level_nodes[i][j].num_child_group = (int*)malloc(num_child *
                    sizeof(int));

            if(level == 1){
                for(int k = 0; k < num_child; k++){
                    target_level_nodes[i][j].num_child_group[k] = 2;
                }
            }else{
                target_level_nodes[i][j].num_child_group[0] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH][j * CHILD_BLOCK_WIDTH].num_TG;
                target_level_nodes[i][j].num_child_group[1] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH][j * CHILD_BLOCK_WIDTH + 1].num_TG;
                target_level_nodes[i][j].num_child_group[2] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH + 1][j * CHILD_BLOCK_WIDTH].num_TG;
                target_level_nodes[i][j].num_child_group[3] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH+ 1][j * CHILD_BLOCK_WIDTH + 1].num_TG;
            }

            target_level_nodes[i][j].input = (float**)malloc(num_child *
                    sizeof(float*));
            for(int k = 0; k < num_child; k++){
                target_level_nodes[i][j].input[k] =
                    (float*)malloc(target_level_nodes[i][j].num_child_group[k] *
                            sizeof(float));
            }

            target_level_nodes[i][j].QC = (int**)malloc(num_max_qc *
                    sizeof(int*));
            for(int k = 0; k < num_max_qc; k++){
                target_level_nodes[i][j].QC[k] = (int*)malloc(num_child *
                        sizeof(int));
            }
            target_level_nodes[i][j].ED = (float*)malloc(num_max_qc * sizeof(float));
            target_level_nodes[i][j].time_adj_matrix =
                (int**)malloc(num_max_qc * sizeof(int*));
            for(int k = 0; k < num_max_qc; k++){
                target_level_nodes[i][j].time_adj_matrix[k] =
                    (int*)malloc(num_max_qc * sizeof(int));
            }

            read_knowledge(&(target_level_nodes[i][j]));

            for(int k = 0; k < num_max_qc; k++){
                target_level_nodes[i][j].ED[k] = 0; 
            }


        }
    }

    switch(level){
        case 1:
            level_one_nodes = target_level_nodes;
            break;
        case 2:
            level_two_nodes = target_level_nodes;
            break;
        case 3:
            level_three_nodes = target_level_nodes;
            break;
        case 4:
            level_four_nodes = target_level_nodes;
            break;
        case 5:
            level_five_nodes = target_level_nodes;
            break;
        case 6:
            level_six_node = target_level_nodes;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }

}

/* write learned knowledge to file for specified level */
void write_level_knowledge(int level){
    node_t** target_level_nodes;
    int num_level_width;
    switch(level){
        case 1:
            target_level_nodes = level_one_nodes;
            num_level_width = LEVEL1_WIDTH;
            break;
        case 2:
            target_level_nodes = level_two_nodes;
            num_level_width = LEVEL2_WIDTH;
            break;
        case 3:
            target_level_nodes = level_three_nodes;
            num_level_width = LEVEL3_WIDTH;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }
    for(int i = 0; i < num_level_width; i++){
        for(int j = 0; j < num_level_width; j++){
            write_knowledge(&(target_level_nodes[i][j]));
        }
    }
}

/* learn level one nodes */
void learn_level_one(int base_row, int base_column){
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        for(int j = 0; j < LEVEL1_WIDTH; j++){
            for(int l = 0; l < 4; l++){
                for(int m = 0; m < 4; m++){
                    level_one_nodes[i][j].input[4 * l + m][0] = 1 -
                        pixels[base_row + i * 4 + l][base_column + j * 4 + m];
                    level_one_nodes[i][j].input[4 * l + m][1] = pixels[base_row + i * 4 + l][base_column + j * 4 + m];
                }
            }

            learn(&(level_one_nodes[i][j]));
        }
    }
}

/* learn nodes for specified level, actually only used for level two nodes */
void learn_level(int level){
    node_t** target_level_nodes;
    node_t** child_level_nodes;
    int num_level_width;
    switch(level){
        case 2:
            target_level_nodes = level_two_nodes;
            child_level_nodes = level_one_nodes;
            num_level_width = LEVEL2_WIDTH;
            break;
        case 3:
            target_level_nodes = level_three_nodes;
            child_level_nodes = level_two_nodes;
            num_level_width = LEVEL3_WIDTH;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }

    /* get input of each node from its child nodes */
    for(int i = 0; i < num_level_width; i++){
        for(int j = 0; j < num_level_width; j++){
            /* get input of child0, set child output to zero after reading */
            for(int k = 0; k < target_level_nodes[i][j].num_child_group[0]; k++){
                target_level_nodes[i][j].input[0][k] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH][j * CHILD_BLOCK_WIDTH].output[k];
                child_level_nodes[i * CHILD_BLOCK_WIDTH][j *
                    CHILD_BLOCK_WIDTH].output[k] = 0;
            }
            /* get input of child1 set child output to zero after reading */
            for(int k = 0; k < target_level_nodes[i][j].num_child_group[1]; k++){
                target_level_nodes[i][j].input[1][k] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH][j * CHILD_BLOCK_WIDTH + 1].output[k];
                child_level_nodes[i * CHILD_BLOCK_WIDTH][j *
                    CHILD_BLOCK_WIDTH + 1].output[k] = 0;
            }
            /* get input of child2 set child output to zero after reading */
            for(int k = 0; k < target_level_nodes[i][j].num_child_group[2]; k++){
                target_level_nodes[i][j].input[2][k] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH + 1][j * CHILD_BLOCK_WIDTH].output[k];
                child_level_nodes[i * CHILD_BLOCK_WIDTH + 1][j *
                    CHILD_BLOCK_WIDTH].output[k] = 0;
            }
            /* get input of child3 set child output to zero after reading */
            for(int k = 0; k < target_level_nodes[i][j].num_child_group[3]; k++){
                target_level_nodes[i][j].input[3][k] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH + 1][j * CHILD_BLOCK_WIDTH + 1].output[k];
                child_level_nodes[i * CHILD_BLOCK_WIDTH + 1][j *
                    CHILD_BLOCK_WIDTH + 1].output[k] = 0;
            }

            /* learn the node */
            learn(&(target_level_nodes[i][j]));
        }
    }
}

/* learn level three(top level) node */
void learn_level_three(){
    // get input for each node of level one from pixels and learn
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < level_three_nodes[0][0].num_child_group[4 * i +
                    j]; k++){
                level_three_nodes[0][0].input[4 * i + j][k] =
                    level_two_nodes[i][j].output[k];
            }
        }
    }

    learn(&(level_three_nodes[0][0]));
}

/* inference nodes in level one */
void inference_level_one(int base_row, int base_column){
    /* the current node's coodinate is (i, j), then
    * its four childs' coordinates are:
    * (i * CHILD_BLOCK_WIDTH, j * CHILD_BLOCK_WIDTH)
    * (i * CHILD_BLOCK_WIDTH, j * CHILD_BLOCK_WIDTH + 1)
    * (i * CHILD_BLOCK_WIDTH + 1, j * CHILD_BLOCK_WIDTH)
    * (i * CHILD_BLOCK_WIDTH + 1, j * CHILD_BLOCK_WIDTH + 1)
    */

    /* get input for each node of level one from pixels and learn */
    for(int i = 0; i < LEVEL1_WIDTH; i++){
        for(int j = 0; j < LEVEL1_WIDTH; j++){
            for(int l = 0; l < 4; l++){
                for(int m = 0; m < 4; m++){
                    level_one_nodes[i][j].input[4 * l + m][0] = 1 -
                        pixels[base_row + i * 4 + l][base_column + j * 4 + m];
                    level_one_nodes[i][j].input[4 * l + m][1] = pixels[base_row + i * 4 + l][base_column + j * 4 + m];
                }
            }

            inference(&(level_one_nodes[i][j]));
        }
    }
}

/* inference nodes for specified level, actually only used to inference level two */
void inference_level(int level){
    node_t** target_level_nodes;
    node_t** child_level_nodes;
    int num_level_width;
    switch(level){
        case 2:
            target_level_nodes = level_two_nodes;
            child_level_nodes = level_one_nodes;
            num_level_width = LEVEL2_WIDTH;
            break;
        case 3:
            target_level_nodes = level_three_nodes;
            child_level_nodes = level_two_nodes;
            num_level_width = LEVEL3_WIDTH;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }

    /* get input of each node from its child nodes */
    for(int i = 0; i < num_level_width; i++){
        for(int j = 0; j < num_level_width; j++){
            /* get input of child0*/
            for(int k = 0; k < target_level_nodes[i][j].num_child_group[0]; k++){
                target_level_nodes[i][j].input[0][k] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH][j * CHILD_BLOCK_WIDTH].output[k];
                child_level_nodes[i * CHILD_BLOCK_WIDTH][j *
                    CHILD_BLOCK_WIDTH].output[k] = 0;
            }
            /* get input of child1*/
            for(int k = 0; k < target_level_nodes[i][j].num_child_group[1]; k++){
                target_level_nodes[i][j].input[1][k] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH][j * CHILD_BLOCK_WIDTH + 1].output[k];
                child_level_nodes[i * CHILD_BLOCK_WIDTH][j *
                    CHILD_BLOCK_WIDTH + 1].output[k] = 0;
            }
            /* get input of child2*/
            for(int k = 0; k < target_level_nodes[i][j].num_child_group[2]; k++){
                target_level_nodes[i][j].input[2][k] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH + 1][j * CHILD_BLOCK_WIDTH].output[k];
                child_level_nodes[i * CHILD_BLOCK_WIDTH + 1][j *
                    CHILD_BLOCK_WIDTH].output[k] = 0;
            }
            /* get input of child3*/
            for(int k = 0; k < target_level_nodes[i][j].num_child_group[3]; k++){
                target_level_nodes[i][j].input[3][k] = child_level_nodes[i *
                    CHILD_BLOCK_WIDTH + 1][j * CHILD_BLOCK_WIDTH + 1].output[k];
                child_level_nodes[i * CHILD_BLOCK_WIDTH + 1][j *
                    CHILD_BLOCK_WIDTH + 1].output[k] = 0;
            }

            /* inference the node */
            inference(&(target_level_nodes[i][j]));
        }
    }
}

/* inference level three(top level) node */
void inference_level_three(){
    // get input for each node of level one from pixels and learn
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < level_three_nodes[0][0].num_child_group[4 * i +
                    j]; k++){
                level_three_nodes[0][0].input[4 * i + j][k] =
                    level_two_nodes[i][j].output[k];
            }
        }
    }

    inference(&(level_three_nodes[0][0]));
}

/* make temporal group for specified level */
void make_TG_level(int level){
    node_t** target_level_nodes;
    int num_level_width;
    switch(level){
        case 1:
            target_level_nodes = level_one_nodes;
            num_level_width = LEVEL1_WIDTH;
            break;
        case 2:
            target_level_nodes = level_two_nodes;
            num_level_width = LEVEL2_WIDTH;
            break;
        case 3:
            target_level_nodes = level_three_nodes;
            num_level_width = LEVEL3_WIDTH;
            break;
        default:
            printf("illegal level %d\n", level);
            assert(0);
    }
    for(int i = 0; i < num_level_width; i++){
        for(int j = 0; j < num_level_width; j++){
            make_TG(&(target_level_nodes[i][j]));
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

/*
int infer_map[NUM_TG_TOP] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'A', 'B', 'B',
    'C', 'C', 'D','D', 'E', 'E', 'F', 'F', 'G', 'G', 'H', 'H', 'I', 'I', 'K',
    'K', 'L', 'L', 'M', 'M', 'N', 'N', 'O', 'O', 'P', 'P', 'Q', 'Q', 'R', 'R',
    'S', 'S', 'T', 'T', 'U', 'U', 'V', 'V', 'W', 'W', 'X', 'X', 'Y', 'Y', 'Z',
    'Z'};
int infer_result[NUM_INFER_IMAGE];
// */
#endif
