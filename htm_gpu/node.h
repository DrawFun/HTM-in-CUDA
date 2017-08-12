#ifndef NODE_HEADER
#define NODE_HEADER
#include "stdio.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* mode of the node */
#define TRAINING_MODE 0
#define INFERENCE_MODE 1

/* number of levels */
#define NUM_LEVELS 3
/* width of level one nodes. 64 nodes in total */
#define LEVEL1_WIDTH 8
/* width of level two nodes. 16 nodes in total */
#define LEVEL2_WIDTH 4
/* width of level three nodes. 1 node in total */
#define LEVEL3_WIDTH 1

/* number of child nodes of each level */
#define NUM_CHILD_LV1 16
#define NUM_CHILD_LV2 4 
#define NUM_CHILD_LV3 16
#define MAX_NUM_CHILD 16

/* max number of quantization centers(coincidences) of each node */
#define MAX_NUM_QC_LEVEL1 3000
#define MAX_NUM_QC_LEVEL2 3300
#define MAX_NUM_QC_LEVEL3 10000

/* max number of temporal groups of each node */
#define MAX_NUM_TG 700

/* max size of temporal groups of node from each level */
#define MAX_TG_SIZE_LEVEL1 10 
#define MAX_TG_SIZE_LEVEL2 12 
#define MAX_TG_SIZE 12 

/* number of top neighbors */
#define NUM_TOP_NEIGHBOR 10

/* value of natural logarithm */
#define NATURAL_LOG 2.718281

/* configurable value of sigma square */
/* SIGMA_SQUARE value when train */
#define SIGMA_SQUARE_TRAIN 0.1
/* SIGMA_SQUARE value when inference */
#define SIGMA_SQUARE_INFER 1.5
#define SIGMA_SQUARE 0.1
/* real sigma_suqre value used currently */
float sigma_square;

/* number of images to learn */
int num_images = 1;

/* width of image */
#define IMAGE_WIDTH 32

/* width of image */
#define ST_IMAGE_WIDTH (IMAGE_WIDTH * 2) 

/* number of threads to inference level 1 */
#define NUM_THREADS_INFERENCE_LV1 64

/* number of threads to learn level 1 */
#define NUM_THREADS_LEARN_LV1 64

/* number of threads to learn level 2 */
#define NUM_THREADS_LEARN_LV2 144

/* number of threads to learn inference 2 */
#define NUM_THREADS_INFERENCE_LV2 256

/* number of threads to learn level 3 */
#define NUM_THREADS_LEARN_LV3 144

/* number of threads to inference level 3 */
#define NUM_THREADS_INFERENCE_LV3 512 

/* number of temporal groups of the top level node */
#define NUM_TG_TOP 58

/* number of images to inference */
#define NUM_INFER_IMAGE 250

/* number of steps an image move for eye-move inference */
#define NUM_STEP_EYE_MOVE 10

/* structure to collect data of eye movement */
typedef struct occur_prob{
    int occur;
    float prob;
}occur_prob_t;

/* structure of node */
typedef struct node{
    /* id of the node */
    int level;
    int row;
    int column;
    /* number of QCs of the node */
    int num_QC;
    /* max number of QCs of the node */
    int num_max_QC;   
    /* number of the node's children */
    int num_child;
    
    /* actual number of temporal groups of each child node */
    int num_child_group[MAX_NUM_CHILD];

    /* input QC */
    int input_QC[MAX_NUM_CHILD];

    /* each of the QC of the node on cpu */
    int* QC;
    /* each of the QC of the node on gpu */
    int* QC_gpu;

    int closest_index;
    /* index of QC of last time, should be initialized to -1*/
    int last_QC;
    /* time-adjacency marix of the node */
    int* time_adj_matrix;
    int* time_adj_matrix_gpu;
    /* actual number of temporal groups of the node */
    int num_TG;
    /* id of each of the QC indicating which TG the QC belongs to, should be
     * initialized to -1 */
    int* group_id;
    int* group_id_gpu;

    /* probability of each of the QC */
    float* prob_QC;
    float* prob_QC_gpu;
    /* output of the node, probability distribution over TGs, should be
     * initialized to zero */
    int inference_groupid;
}node_t;

/* structure used in bucket sorting */
typedef struct sum_id{
    int sum;
    int id;
    sum_id* prev;
    sum_id* next;
}sum_id_t;

/* make temporal group of the specified node according to time information
 * stored in time adjacent matrix
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
            ori_QC[i].sum += node->time_adj_matrix[i * node->num_max_QC + j];
        }
        ori_QC[i].prev = NULL;
        ori_QC[i].next = NULL;
    }
    
    sum_id_t buckets[10];
    sum_id_t* buckets_tail[10];
    for(int i = 0; i < 10; i++){
        /* for buckets entry, sum acts as the number of nodes of the entry of
        * the current iterator, while id acts as the number of nodes of the
        * entry of the current time */
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
    switch(node->level){
        case 1:
            max_tg_size = MAX_TG_SIZE_LEVEL1;
            break;
        case 2:
            max_tg_size = MAX_TG_SIZE_LEVEL2;
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
                ori_QC[i].sum = node->time_adj_matrix[id * node->num_max_QC + i] + 
                    node->time_adj_matrix[i * node->num_max_QC + id];
                ori_QC[i].id = i;
                ori_QC[i].prev = NULL;
                ori_QC[i].next = NULL;
            }
            for(int i = id + 1; i < node->num_QC; i++){
                ori_QC[i - 1].sum = node->time_adj_matrix[id * node->num_max_QC + i] +
                    node->time_adj_matrix[i * node->num_max_QC + id];
                ori_QC[i - 1].id = i;
                ori_QC[i - 1].prev = NULL;
                ori_QC[i - 1].next = NULL;
            }

            for(int i = 0; i < 10; i++){
                /* for buckets entry, sum acts as the number of nodes of the entry of
                * the current iterator, while id acts as the number of nodes of the
                * entry of the current time */
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

/* write learned knowledge into file specified by the node's id 
 * content written into file is listed as follows:
 * num_child
 * num_child_group
 * num_QC
 * QC
 * num_TG
 * group_id
 * */
void write_knowledge(node_t* node){
    char* file_name = (char*)malloc(16);
    sprintf(file_name, "learned/%d_%d_%d", node->level, node->row, node->column);
    FILE* write_file = fopen(file_name, "w");
    fprintf(write_file, "%d\n", node->num_child);
    for(int i = 0; i < node->num_child; i++){
        fprintf(write_file, "%d\n", node->num_child_group[i]);
    }
    fprintf(write_file, "%d\n", node->num_QC);
    for(int i = 0; i < node->num_QC; i++){
        for(int j = 0; j < node->num_child; j++){
            fprintf(write_file, "%d ", node->QC[i * node->num_child + j]);
        }
        fprintf(write_file, "\n");
    }
    fprintf(write_file, "%d\n", node->num_TG);
    for(int i = 0; i < node->num_QC; i++){
        fprintf(write_file, "%d ", node->group_id[i]);
    }
    fclose(write_file);
}
#endif
