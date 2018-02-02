// Author: Wes Kendall
// Copyright 2012 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Program that computes the average of an array of elements in parallel using
// MPI_Scatterv and MPI_Gather
//
// Modified from avgv.c by Stephen Cook 2018
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}

// Create two arrays, splitting elements_to_split between num_bins bins, and
// giving the displacement.
void split_num_between_processes(int elements_to_split, int num_bins,
                                 int *num_elements_by_process,
                                 int *displacements) {
  // Specify how these will be distributed amongst the processes
  int element_position = 0;
  int i;
  for (i = 0; i < num_bins - 1; i++) {
    displacements[i] = element_position;
    // Integer division
    num_elements_by_process[i] = elements_to_split / num_bins;
    element_position += elements_to_split / num_bins;
  }
  // Assign the remaining elements to the last process.  If num_bins does
  // not divide elements_to_split exactly, the last process will be assigned
  // the more elements that the others.
  displacements[num_bins - 1] = element_position;
  num_elements_by_process[num_bins - 1] = elements_to_split - element_position;
}

// Computes the average of an array of numbers
float compute_avg(float *array, int num_elements) {
  float sum = 0.f;
  int i;
  for (i = 0; i < num_elements; i++) {
    sum += array[i];
  }
  return sum / num_elements;
}

// Compute the weighted average of an array of numbers
float compute_weighted_avg(float *array, int *weights_array,
                           int num_elements) {
  float sum = 0.f;
  float sum_weights = 0.f;
  int i;
  for (i = 0; i < num_elements; i++) {
    sum += weights_array[i] * array[i];
    sum_weights += weights_array[i];
  }
  return sum / sum_weights;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: avg total_num_elements\n");
    exit(1);
  }

  int total_num_elements = atoi(argv[1]);
  // Seed the random number generator to get different results each time
  srand(time(NULL));

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Create a random array of elements on the root process. Its elements will
  // be distributed between the processes
  float *rand_nums = NULL;
  if (world_rank == 0) {
    rand_nums = create_rand_nums(total_num_elements);
  }


  // Specify how these will be distributed amongst the processes
  int *num_elements_by_process = (int *)malloc(sizeof(int) * world_size);
  assert(num_elements_by_process != NULL);
  int *displacements = (int *)malloc(sizeof(int) * world_size);
  assert(displacements != NULL);
  split_num_between_processes(total_num_elements, world_size,
                              num_elements_by_process, displacements);

  // For each process, create a buffer that will hold a subset of the entire
  // array
  float *sub_rand_nums = (float *)malloc(sizeof(float) * num_elements_by_process[world_rank]);
  assert(sub_rand_nums != NULL);

  // Scatter the random numbers from the root process to all processes in
  // the MPI world
  MPI_Scatterv(rand_nums, num_elements_by_process, displacements, MPI_FLOAT,
              sub_rand_nums, num_elements_by_process[world_rank], MPI_FLOAT, 0,
              MPI_COMM_WORLD);

  // Compute the average of your subset
  float sub_avg = compute_avg(sub_rand_nums, num_elements_by_process[world_rank]);

  // Gather all partial averages down to the root process
  float *sub_avgs = NULL;
  if (world_rank == 0) {
    sub_avgs = (float *)malloc(sizeof(float) * world_size);
    assert(sub_avgs != NULL);
  }
  MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Now that we have all of the partial averages on the root, compute the
  // total average of all numbers. Since we are assuming each process computed
  // an average across an equal amount of elements, this computation will
  // produce the correct answer.
  if (world_rank == 0) {
    float avg = compute_weighted_avg(sub_avgs, num_elements_by_process, world_size);
    printf("Avg of all elements is %f\n", avg);
    // Compute the average across the original data for comparison
    float original_data_avg =
      compute_avg(rand_nums, total_num_elements);
    printf("Avg computed across original data is %f\n", original_data_avg);
  }

  // Clean up
  if (world_rank == 0) {
    free(rand_nums);
    free(sub_avgs);
  }
  free(sub_rand_nums);
  free(num_elements_by_process);
  free(displacements);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
