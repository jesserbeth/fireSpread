
#ifndef KERNEL_H_
#define KERNEL_H_
#include <math.h>
#include <stdio.h>
// __device__ int __float_as_int   (   float   x    )

__device__ int end;
__device__ int syncCounter;
const int INF = 999999;

/////////////////////////////////////////////////////////////////////////////
//                            Burning Distances
/////////////////////////////////////////////////////////////////////////////
__global__ void BurnDist(float* ignTimeIn, float* ignTimeOut,float* burnDist,
                         float* rothData, float* times,float* L_n, int size,
                         int rowSize, int colSize, float timeStep, float t){
      /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW*/
   int nCol[8] =        {  0,  1,  1,  1,  0, -1, -1, -1};
   int nRow[8] =        {  1,  1,  0, -1, -1, -1,  0,  1};
   // printf("Iterative Minimal Time\n");
   float ignTime, ignTimeN;
   float dist;

   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   int ncell, nrow, ncol, row, col, distCell;
   float ROS;

      while(cell < size){
         row = cell / rowSize;
         col = cell - rowSize*row;
         // printf("%d ", cell);
         ignTime = ignTimeIn[cell];
         if(ignTime == INF){
            cell += blockDim.x * gridDim.x;
            continue;
          }

          // check neighbors for ignition
          for(int n = 0; n < 8; n++){
            // printf("%d ", n);
            distCell = cell * 8;
              nrow = row + nRow[n];
              ncol = col + nCol[n];
              if ( nrow<0 || nrow>=rowSize || ncol<0 || ncol>=colSize )
                  continue;
              ncell = ncol + nrow*colSize;

              // check for already lit
              ignTimeN = ignTimeIn[ncell];
              if(ignTimeN < INF){
                  continue;
              }

              // Calc roth values
              ROS = rothData[3*cell + 0] * (1.0 - rothData[3*cell + 1]) / 
                (1.0 - rothData[3*cell + 1] * cos(rothData[3*cell + 2] * 3.14159/180.));

              // Burn distance
              dist = burnDist[distCell+n];
              dist = dist - ROS*timeStep;
              burnDist[distCell+n] = dist;

              // Propogate fire 
              if(dist <= 0){
                float old = atomicExch(&ignTimeOut[ncell], t);
                if(old < t)
                   atomicExch(&ignTimeOut[ncell], old);
                if(ncell == 0)
                  atomicAdd(&end, 1);
                if(ncell == (rowSize - 1))
                  atomicAdd(&end, 1);
                if(ncell == (nrow*rowSize-1))
                  atomicAdd(&end, 1);
                if(ncell == size -1)
                  atomicAdd(&end, 1);
              }

          }
         cell += blockDim.x * gridDim.x;
      }
}


/////////////////////////////////////////////////////////////////////////////
//                             Copy Kernel (BD)
/////////////////////////////////////////////////////////////////////////////
__global__ void copyKernelBD(float* input, float* output, int size){
   // copy from output to input
   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   // printf("%d ", cell);
   // end = 0;

   while(cell < size){
      input[cell] = output[cell];
      // if(check[cell] == true){
      //    // printf("true\n");
      //    atomicAdd(&end, 1);
      // }
      cell += blockDim.x * gridDim.x;
      // printf("%d ", end);
   }
   // printf("copy_out\n");
}
#endif // KERNEL_H_
