
#ifndef KERNEL_H_
#define KERNEL_H_
#include <math.h>
#include <stdio.h>
// __device__ int __float_as_int   (   float   x    )

__device__ int end;
__device__ int syncCounter;
const int INF = 32767;

/////////////////////////////////////////////////////////////////////////////
//                              Minimal Time
/////////////////////////////////////////////////////////////////////////////
__global__ void MinTime(int* ignTime, int* rothData, int* times, 
                        float* L_n, int size, int rowSize,
                        int colSize){
   /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW*/
   int nCol[16] =        {  0,  1,  1,  1,  0, -1, -1, -1, -1, 1, 2, 2, 1, -1, -2, -2};
   int nRow[16] =        {  1,  1,  0, -1, -1, -1,  0,  1, 2, 2, 1, -1, -2, -2, -1, 1};
   
   // Calculate ThreadID
   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   int ncell, nrow, ncol, row, col;
   float /*ignCell, ignCellN, timeNext, timeNow,*/ ROS;
   int ignCell, ignCellN, timeNow, timeNext;

   timeNow = times[0]; // timeNow = timeNext
   // printf("%d ", times[1]);
   timeNext = INF;
   // printf("%d ", 5);

   while(cell < size){
    // printf("%d ", cell);
    // timeNext = INF;
      row = cell / rowSize;
      col = cell - rowSize*row;
      ignCell = ignTime[cell];

      // Do atomic update of TimeNext Var (atomicMin)
      if(ignCell > timeNow){
        int old = atomicMin(&times[1], ignCell);
        if(ignCell < old){
          timeNext = ignCell;
          // printf("First If \n");
        }
      }
      else if(ignCell == timeNow){ // I am on fire now, and will propagate 
         // Check through neighbors
         for(int n = 0; n < 16; n++){
            // // Propagate from burning cells      
            nrow = row + nRow[n];
            ncol = col + nCol[n];
         // printf("nrow: %d ncol: %d\n",nrow,ncol);
            if ( nrow<0 || nrow>= rowSize || ncol<0 || ncol>=  colSize ){
               continue;
            }
            ncell = ncol + nrow*colSize;
            ignCellN = ignTime[ncell];

            // If neighbor is unburned
            if(ignCellN > timeNow){
              // compute ignition time
              ROS = rothData[3*cell + 0] * (1.0 - rothData[3*cell + 1]) / 
                    (1.0 - rothData[3*cell + 1] * cos(rothData[3*cell + 2] * 3.14159/180));

              float ignTimeNew = timeNow + (L_n[n] / ROS)*100;

              // Update Output TOA Map
              atomicMin(&ignTime[ncell], (int)ignTimeNew);

              if(ncell == 0)
                atomicAdd(&end, 1);
              if(ncell == (rowSize - 1))
                atomicAdd(&end, 1);
              if(ncell == (nrow*rowSize-1))
                atomicAdd(&end, 1);
              if(ncell == size -1)
                atomicAdd(&end, 1);

              // Local timeNext update
              if((int)ignTimeNew < timeNext){
                timeNext = (int)ignTimeNew;
              }
            }
         }
         // Perform global timeNext update
         atomicMin(&times[1], timeNext);
      }

      // Do striding
      cell += blockDim.x * gridDim.x;
   }


}

/////////////////////////////////////////////////////////////////////////////
//                             Time Update (MT)
/////////////////////////////////////////////////////////////////////////////
__global__ void timeKernelMT(int* times){
  times[0] = times[1];
  times[1] = INF;
}

#endif // KERNEL_H_


