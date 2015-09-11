
#ifndef KERNEL_H_
#define KERNEL_H_
#include <math.h>
#include <stdio.h>
// __device__ int __float_as_int   (   float   x    )

__device__ int end;
__device__ int syncCounter;
const int INF = 32767;

/////////////////////////////////////////////////////////////////////////////
//                          Iterative Minimal Time
/////////////////////////////////////////////////////////////////////////////
__global__ void ItMinTime(int* ignTimeIn, int* ignTimeOut, int* ignTimeStep,
                          float* rothData, int* times, float* L_n, bool* check, 
                          int size, int rowSize, int colSize){
   /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW*/
   int nCol[16] =        {  0,  1,  1,  1,  0, -1, -1, -1, -1, 1, 2, 2, 1, -1, -2, -2};
   int nRow[16] =        {  1,  1,  0, -1, -1, -1,  0,  1, 2, 2, 1, -1, -2, -2, -1, 1};
   // printf("Iterative Minimal Time\n");
   float ignCell = 0;
   float ignCellNew = 0;
   float ignTimeMin = INF;
   
   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   int ncell, nrow, ncol, row, col;
   float ignTimeNew, ROS;

   while(cell < size){
      row = cell / rowSize;
      col = cell - rowSize*row;
      // printf("%d ", cell);

      // Do nothing if converged
      if(check[cell] == true){
        cell += blockDim.x * gridDim.x; 
         // atomicAdd(&end, 1);
        // printf("if_statement_1\n");
        continue;
      }
      
      // Check for simulation completion
      ignCell = ignTimeIn[cell];
      ignCellNew = ignTimeOut[cell];
      // Convergence Test
      if(fabs(ignCell - ignCellNew) < 2 && ignCell != INF
            && ignCellNew != INF && check[cell] != true){
        check[cell] = true;
        cell += blockDim.x * gridDim.x;  
         // atomicAdd(&end, 1);
        // printf("CONVERGED");
        continue;
      }
      if(ignCell > 0){
        // ignTimeMin = INF; 
        // printf("ignCell > 0\n");
        ignTimeMin = INF;
        // Loop through neighbors
        for(int n = 0; n < 16; n++){
            // find neighbor cell index     
            nrow = row + nRow[n];
            ncol = col + nCol[n];
            if ( nrow<0 || nrow>= rowSize || ncol<0 || ncol>=  colSize ){
               continue;
            }
            ncell = ncol + nrow*colSize;

            // ROS = rothData[3*cell + 0] * (1.0f - rothData[3*cell + 1]) / 
            //     (1.0 - rothData[3*cell + 1] * cos(rothData[3*cell + 2] * 3.14159f/180));
            ROS = rothData[3*ncell + 0] * (1.0f - rothData[3*ncell + 1]) / 
                (1.0 - rothData[3*ncell + 1] * cos(rothData[3*ncell + 2] * 3.14159f/180));
            ignTimeNew = ignTimeIn[ncell] + L_n[n] / ROS * 100;
            // ignTimeNew = ignCellN + (L_n[n] / ROS) * 100;
            // printf("ignTimeNew: %f ", ignTimeNew);
            ignTimeMin = ignTimeNew*(ignTimeNew < ignTimeMin) + ignTimeMin*(ignTimeNew >= ignTimeMin);
            // printf("ignTimeNewMin: %f \n", ignTimeNew);
        }
        // ignTimeStep[cell] = (int)ignTimeMin; // atomic min here?
        ignTimeOut[cell] = (int)ignTimeMin;
        // atomicMin(&ignTimeOut[cell], (int)ignTimeMin);
      }
      cell += blockDim.x * gridDim.x;
   }
   // printf("Testing IMT Kernel\n");
   if(blockIdx.x * blockDim.x + threadIdx.x == 0)
      end = 0;
}

/////////////////////////////////////////////////////////////////////////////
//                             Copy Kernel (IMT)
/////////////////////////////////////////////////////////////////////////////
__global__ void copyKernelIMT(int* input, int* output, bool* check, int size){
   // copy from output to input
   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   // printf("%d ", cell);
   // end = 0;

   while(cell < size){
      // input[cell] = output[cell];
      if(check[cell] == true){
         // printf("true\n");
         atomicAdd(&end, 1);
      }
      cell += blockDim.x * gridDim.x;
      // printf("%d ", end);
   }
}
#endif // KERNEL_H_
