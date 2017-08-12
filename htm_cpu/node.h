#ifndef NODE_HEADER
#define NODE_HEADER
#include "stdio.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* mode of the node */
#define TRAINIGN_MODE 0
#define INFERENCE_MODE 1

/* max number of child nodes of each nodes*/
#define NUM_CHILD_LV1 16
#define NUM_CHILD_LV2 4 
#define NUM_CHILD_LV3 16
#define MAX_NUM_CHILD 4

/* Euclidean distance to decide whether a incoming pattern is a new pattern*/
#define ED_THRESHHOLD 0

/* max number of quantization centers(coincidences) of each node */
#define MAX_NUM_QC_LEVEL1 3000
#define MAX_NUM_QC_LEVEL2 7000
#define MAX_NUM_QC_LEVEL3 40000
#define MAX_NUM_QC_LEVEL4 2000
#define MAX_NUM_QC_LEVEL5 1500
#define MAX_NUM_QC_LEVEL6 1500
#define MAX_NUM_QC 20000 

/* max number of temporal groups of each node */
#define MAX_NUM_TG 3000

/* max size of temporal groups of each node */
#define MAX_TG_SIZE_LEVEL1 10 
#define MAX_TG_SIZE_LEVEL2 12 
#define MAX_TG_SIZE_LEVEL3 10
#define MAX_TG_SIZE_LEVEL4 10
#define MAX_TG_SIZE_LEVEL5 10 
#define MAX_TG_SIZE 12 

/* number of top neighbors */
#define NUM_TOP_NEIGHBOR 10

/* value of natural logarithm */
#define NATURAL_LOG 2.718281

/* configurable value of sigma square */
#define SIGMA_SQUARE 0.1

/* number of TGs saw by top node */
#define NUM_TG_TOP 60

/* number of images to inference */
#define NUM_INFER_IMAGE 250

/* number of steps that an image move for eye-move inference */
#define NUM_STEP_EYE_MOVE 12

/* recognized index */
int rec_index = -1;
/* recognized probability */
float rec_prob = 0;

/* structure of node id */
typedef struct node_id{
    int level;
    int row;
    int column;
}node_id_t;

/* structure of Lv1 node */
typedef struct node{
    /* id of the node */
    node_id_t id;
    /* mode of the node */
    int mode;
    /* number of lv1 node's children */
    int num_child;
    /* actual number of temporal groups of each child node */
    int* num_child_group;
    /* input from child nodes */
    float** input;

    /* number of QCs of the node */
    int num_QC;
    /* max number of QCs of the node */
    int num_max_QC;
    /* each of the QC of the node */
    int** QC; 
    /* euclidean distance from each existing QC */
    float* ED;

    /* index of QC of last time, should be initialized to -1*/
    int last_QC;
    /* time-adjacency marix of the node */
    //int time_adj_matrix[MAX_NUM_QC][MAX_NUM_QC];
    int** time_adj_matrix;
    /* actual number of temporal groups of the node */
    int num_TG;
    /* id of each of the QC indicating the TG the QC belongs to, should be
     * initialized to -1 */
    int group_id[MAX_NUM_QC];

    /* probability of each of the QC */
    float prob_QC[MAX_NUM_QC];
    /* output of the node, probability distribution over TGs, should be
     * initialized to zero */
    float output[MAX_NUM_TG];
}node_t;

typedef struct sum_id{
    int sum;
    int id;
    sum_id* prev;
    sum_id* next;
}sum_id_t;

/* learn for the specified node */
void learn(node_t* node);

/* inference for the specified node */
void inference(node_t* node);

/* make temporal group for the specified node */
void make_TG(node_t* node);

void learn(node_t* node){
    int closest_index = -1;
    int min_ed = 10000;
    int* input_QC = (int*)malloc(node->num_child * sizeof(int));
    for(int i = 0; i < node->num_child; i++){
        input_QC[i] = 0;
    }

    for(int i = 0; i < node->num_child; i++){
        // before learn the node, input_QC should be zero
        assert(input_QC[i] == 0);
    }

    float max_prob = -1;
    int max_index = -1;
    for(int i = 0; i < node->num_child; i++){
        for(int j = 0; j < node->num_child_group[i]; j++){
            if(node->input[i][j] > max_prob){
                max_prob = node->input[i][j];
                max_index = j;
            }
        }
        input_QC[i] = max_index;
        max_prob = -1;
        max_index = -1;
    }

    int same = 1;
    for(int i = 0; i < node->num_QC; i++){
        same = 1;
        for(int j = 0; j < node->num_child; j++){
            if(node->QC[i][j] != input_QC[j]){
                same = 0;
                break;
            }
        }
        if(same){
            closest_index = i;
            break;
        }
    }

    if(node->num_QC == 0){
        same = 0;
    }

    if(!same && min_ed == 0){
        // should never come here
        printf("strange!\n");
        for(int i = 0; i < node->num_child; i++){
            printf("%d ", input_QC[i]);
        }
        printf("\n");

        for(int i = 0; i < node->num_QC; i++){
            for(int j = 0; j < node->num_child; j++){
                printf("%d ", node->QC[i][j]);
            }
            printf("\n");
        }
    }
    if(!same){
        node->num_QC++;
        if(node->num_QC > node->num_max_QC){
            // should never come here
            printf("(%d, %d), exceed\n", node->id.row, node->id.column);
        }
        for(int i = 0; i < node->num_child; i++){
            node->QC[node->num_QC - 1][i] = input_QC[i];
        }
        closest_index = node->num_QC - 1;
    }

    // train time-adjacency matrix
    if(node->id.level < 3){
        if(node->last_QC == -1){
            node->last_QC = 0;
        }else{
            node->time_adj_matrix[node->last_QC][closest_index]++; 
            node->last_QC = closest_index; 
        }
    }

    free(input_QC);
}

/* make temporal group for the specified node according to information stored in
 * time adjacent matrix
 * use bucket sort algorithm to do sort work
 */
void make_TG(node_t* node){
    /* sort the QC by row sum of the node's time_adj_matrix using bucket sorting algorithm 
     * the ordered result is stored in ordered_QC vector */
    sum_id_t ori_QC[node->num_QC];
    sum_id_t ordered_QC[node->num_QC];
    for(int i = 0; i < node->num_QC; i++){
        ori_QC[i].sum = 0;
        ori_QC[i].id = i;
        for(int j = 0; j < node->num_QC; j++){
            ori_QC[i].sum += node->time_adj_matrix[i][j];
        }
        ori_QC[i].prev = NULL;
        ori_QC[i].next = NULL;
    }
    
    sum_id_t buckets[10];
    sum_id_t* buckets_tail[10];
    for(int i = 0; i < 10; i++){
        // for buckets entry, sum acts as the number of nodes of the entry of
        // the current iterator, while id acts as the number of nodes of the
        // entry of the current time
        buckets[i].sum = 0;
        buckets[i].id = 0;
        buckets[i].prev = NULL;
        buckets[i].next = NULL;
        buckets_tail[i] = &(buckets[i]);
    }
    int remainder = -1;
    for(int i = 0; i < node->num_QC; i++){
        remainder = ori_QC[i].sum % 10;
        ori_QC[i].prev = buckets_tail[remainder];
        buckets_tail[remainder]->next = &(ori_QC[i]);
        buckets_tail[remainder] = &(ori_QC[i]);
        buckets[remainder].id++;
    }

    int divider = 10;
    int not_finished = 0;
    sum_id_t* cur_ptr;
    do{
        not_finished = 0;
        for(int i = 0; i < 10; i++){
            buckets[i].sum = buckets[i].id;
        }

        for(int i = 0; i < 10; i++){
            for(int j = 0; j < buckets[i].sum; j++){
                cur_ptr = buckets[i].next;
                assert(cur_ptr != NULL);
                if(cur_ptr->sum / divider > 0){
                    not_finished = 1;
                }
                remainder = (cur_ptr->sum / divider) % 10;
                cur_ptr->prev->next = cur_ptr->next;
                if(cur_ptr->next != NULL){
                    cur_ptr->next->prev = cur_ptr->prev;
                }else{
                    buckets_tail[i] = &(buckets[i]);
                }
                buckets[i].id--;

                buckets_tail[remainder]->next = cur_ptr;
                cur_ptr->prev = buckets_tail[remainder];
                cur_ptr->next = NULL;
                buckets_tail[remainder] = cur_ptr;
                buckets[remainder].id++;
            }
        }
        divider *= 10;
    }while(not_finished);
    
    divider = 0;    // this time divider act as index of ordered_QC
    for(int i = 9; i >= 0; i--){
        cur_ptr = buckets_tail[i];
        for(int j = 0; j < buckets[i].id; j++){
            ordered_QC[divider].sum = cur_ptr->sum;
            ordered_QC[divider].id = cur_ptr->id;
            divider++;
            cur_ptr = cur_ptr->prev;
        }
    }

    for(int i = 0;i < node->num_QC; i++){
        if(i > 0){
            if(ordered_QC[i].sum > ordered_QC[i - 1].sum){
                assert(0);
            }
        }
    }

    /* group the QCs */
    divider = node->num_QC;    // this time divider acts as the number of ungrouped QCs
    int insert_index = 0;    // index to insert QC to new_group
    int check_index = 0;    // index to check entry in QC
    int ordered_QC_index = 0;  // index to find the largest ungrouped QC
    sum_id_t ordered_QC_group[node->num_QC - 1];
    int max_tg_size;
    switch(node->id.level){
        case 1:
            max_tg_size = MAX_TG_SIZE_LEVEL1;
            break;
        case 2:
            max_tg_size = MAX_TG_SIZE_LEVEL2;
            break;
        case 3:
            max_tg_size = MAX_TG_SIZE_LEVEL3;
            break;
        case 4:
            max_tg_size = MAX_TG_SIZE_LEVEL4;
            break;
        case 5:
            max_tg_size = MAX_TG_SIZE_LEVEL5;
            break;
    }
    int new_group[MAX_TG_SIZE];    // new group
    // each iterator makes a new group
    while(divider){
        insert_index = 0;
        check_index = 0;
        /* find the largest ungrouped QC */
        while(node->group_id[ordered_QC[ordered_QC_index].id] != -1){
            ordered_QC_index++;
        }
        new_group[insert_index++] = ordered_QC[ordered_QC_index].id;
        divider--;
        node->group_id[ordered_QC[ordered_QC_index].id] = node->num_TG;

        while(divider && check_index != insert_index && insert_index < max_tg_size){
            int id = new_group[check_index++];    // index of the QC

            /* sort the id's neighbors by their connectness to the id */
            for(int i = 0; i < id; i++){
                ori_QC[i].sum = node->time_adj_matrix[id][i] + node->time_adj_matrix[i][id];
                ori_QC[i].id = i;
                ori_QC[i].prev = NULL;
                ori_QC[i].next = NULL;
            }
            for(int i = id + 1; i < node->num_QC; i++){
                ori_QC[i - 1].sum = node->time_adj_matrix[id][i] + node->time_adj_matrix[i][id];
                ori_QC[i - 1].id = i;
                ori_QC[i - 1].prev = NULL;
                ori_QC[i - 1].next = NULL;
            }

            for(int i = 0; i < 10; i++){
                // for buckets entry, sum acts as the number of nodes of the entry of
                // the current iterator, while id acts as the number of nodes of the
                // entry of the current time
                buckets[i].sum = 0;
                buckets[i].id = 0;
                buckets[i].prev = NULL;
                buckets[i].next = NULL;
                buckets_tail[i] = &(buckets[i]);
            }
            int remainder = -1;
            for(int i = 0; i < node->num_QC - 1; i++){
                remainder = ori_QC[i].sum % 10;
                ori_QC[i].prev = buckets_tail[remainder];
                buckets_tail[remainder]->next = &(ori_QC[i]);
                buckets_tail[remainder] = &(ori_QC[i]);
                buckets[remainder].id++;
            }

            int divider_group = 10;
            not_finished = 0;
            do{
                not_finished = 0;
                for(int i = 0; i < 10; i++){
                    buckets[i].sum = buckets[i].id;
                }

                for(int i = 0; i < 10; i++){
                    for(int j = 0; j < buckets[i].sum; j++){
                        cur_ptr = buckets[i].next;
                        assert(cur_ptr != NULL);
                        if(cur_ptr->sum / divider_group > 0){
                            not_finished = 1;
                        }
                        remainder = (cur_ptr->sum / divider_group) % 10;
                        cur_ptr->prev->next = cur_ptr->next;
                        if(cur_ptr->next != NULL){
                            cur_ptr->next->prev = cur_ptr->prev;
                        }else{
                            buckets_tail[i] = &(buckets[i]);
                        }
                        buckets[i].id--;

                        buckets_tail[remainder]->next = cur_ptr;
                        cur_ptr->prev = buckets_tail[remainder];
                        cur_ptr->next = NULL;
                        buckets_tail[remainder] = cur_ptr;
                        buckets[remainder].id++;
                    }
                }
                divider_group *= 10;
            }while(not_finished);
    
            divider_group = 0;    
            for(int i = 9; i >= 0; i--){
                cur_ptr = buckets_tail[i];
                for(int j = 0; j < buckets[i].id; j++){
                    ordered_QC_group[divider_group].sum = cur_ptr->sum;
                    ordered_QC_group[divider_group].id = cur_ptr->id;
                    divider_group++;
                    cur_ptr = cur_ptr->prev;
                }
            }

            divider_group = 0;    // this time divider group acts as the number of neighbors checked
            int neighbor_index = 0;
            /* add new QC into the current TG */
            while(divider && divider_group < NUM_TOP_NEIGHBOR && insert_index < max_tg_size){
                int id = ordered_QC_group[neighbor_index++].id;
                divider_group++;
                if(node->group_id[id] == -1){
                    new_group[insert_index++] = id;

                    node->group_id[id] = node->num_TG;
                    divider--;
                }
            }
        }
        node->num_TG++;
    }
}

/* inference for the specified node */
void inference(node_t* node){
    for(int i = 0;i < node->num_TG; i++){
        node->output[i] = 0;
    }
    /* calculate probability distribution over QCs record it to vector prob_QC */
    float ed = 0;    // euclidean distance
    int QC_value = -1;    // value of the the specific child of the specific QC
    float diff = 0;
    for(int i = 0; i < node->num_QC; i++){
        ed = 0;
        for(int j = 0; j < node->num_child; j++){
            QC_value = node->QC[i][j];
            int k = 0;
            for(k = 0; k < QC_value; k++){
                diff = node->input[j][k];
            }
            diff = 1 - node->input[j][QC_value];
            ed += diff * diff;
            for(k = QC_value + 1; k < node->num_child_group[j]; k++){
                diff = node->input[j][k];
            }
        }
        node->prob_QC[i] = pow(NATURAL_LOG, 0 - ed / SIGMA_SQUARE);
    }

    /* select the largest QC prob of the TG as the prob of the TG */
    for(int i = 0; i < node->num_QC; i++){
        if(node->output[node->group_id[i]] < node->prob_QC[i]){
            node->output[node->group_id[i]] = node->prob_QC[i];
        }
    }
    /* make sure no location exceeding num_TG is calculated */
    for(int i = node->num_TG; i < MAX_NUM_TG; i++){
        assert(node->output[i] == 0);
    }
    /* normalize the output vector */
    float sum_prob = 0;
    for(int i = 0; i < node->num_TG; i++){
        sum_prob += node->output[i];
    }
    for(int i = 0; i < node->num_TG; i++){
        node->output[i] /= sum_prob;
    }

    if(node->id.level == 3){
        int tmp_index = -1;
        float tmp_prob = -1;
        for(int i = 0; i < node->num_TG; i++){
            if(node->output[i] > tmp_prob){
                tmp_index = i;
                tmp_prob = node->output[i];
            }
        }

        if(tmp_prob > rec_prob){
            rec_index = tmp_index;
            rec_prob = tmp_prob;
        }
    }
}

/* write learned knowledge into file specified by the node's id 
 * content written into file is listed as follows:
 * mode
 * num_child
 * num_child_group
 * num_QC
 * QC
 * time_adj_matrix
 * num_TG
 * group_id
 * */
void write_knowledge(node_t* node){
    char* file_name = (char*)malloc(16);
    sprintf(file_name, "learned/%d_%d_%d", node->id.level, node->id.row, node->id.column);
    FILE* write_file = fopen(file_name, "w");
    fprintf(write_file, "%d\n", node->mode);
    fprintf(write_file, "%d\n", node->num_child);
    for(int i = 0; i < node->num_child; i++){
        fprintf(write_file, "%d\n", node->num_child_group[i]);
    }
    fprintf(write_file, "%d\n", node->num_QC);
    for(int i = 0; i < node->num_QC; i++){
        for(int j = 0; j < node->num_child; j++){
            fprintf(write_file, "%d ", node->QC[i][j]);
        }
        fprintf(write_file, "\n");
    }
    if(node->id.level < 3){
        for(int i = 0; i < node->num_QC; i++){
            for(int j = 0; j < node->num_QC; j++){
                fprintf(write_file, "%d ", node->time_adj_matrix[i][j]);
            }
            fprintf(write_file, "\n");
        }
    }
    fprintf(write_file, "%d\n", node->num_TG);
    for(int i = 0; i < node->num_QC; i++){
        fprintf(write_file, "%d ", node->group_id[i]);
    }
    fclose(write_file);
}

/* read from previous learned knowledge */
void read_knowledge(node_t* node){
    char* file_name = (char*)malloc(16);
    sprintf(file_name, "learned/%d_%d_%d", node->id.level, node->id.row, node->id.column);
    FILE* read_file = fopen(file_name, "r");
    fscanf(read_file, "%d", &(node->mode));
    fscanf(read_file, "%d", &(node->num_child));
    for(int i = 0; i < node->num_child; i++){
        fscanf(read_file, "%d", &(node->num_child_group[i]));
    }
    fscanf(read_file, "%d", &(node->num_QC));
    for(int i = 0; i < node->num_QC; i++){
        for(int j = 0; j < node->num_child; j++){
            fscanf(read_file, "%d", &(node->QC[i][j]));
        }
    }

    if(node->id.level < 3){
        for(int i = 0; i < node->num_QC; i++){
            for(int j = 0; j < node->num_QC; j++){
                fscanf(read_file, "%d", &(node->time_adj_matrix[i][j]));
            }
        }
    }
    fscanf(read_file, "%d", &(node->num_TG));
    for(int i = 0; i < node->num_QC; i++){
        fscanf(read_file, "%d", &(node->group_id[i]));
    }
    fclose(read_file);
}

/* read knowledge for level three since level 3 has no time adjacent matrix
 * (note that level three(top level) node is learned in supervised mode*/
void read_knowledge_lv3(node_t* node){
    char* file_name = (char*)malloc(16);
    sprintf(file_name, "learned/%d_%d_%d", node->id.level, node->id.row, node->id.column);
    FILE* read_file = fopen(file_name, "r");
    fscanf(read_file, "%d", &(node->mode));
    fscanf(read_file, "%d", &(node->num_child));
    for(int i = 0; i < node->num_child; i++){
        fscanf(read_file, "%d", &(node->num_child_group[i]));
    }
    fscanf(read_file, "%d", &(node->num_QC));
    node->num_TG = NUM_TG_TOP;
    for(int i = 0; i < node->num_QC; i++){
        for(int j = 0; j < node->num_child; j++){
            fscanf(read_file, "%d", &(node->QC[i][j]));
        }
    }
    fscanf(read_file, "%d", &(node->num_TG));
}
#endif
