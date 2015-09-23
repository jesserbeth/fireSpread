#include <iostream>
#include <fstream>
#include "fireSim.h"
#include <sys/time.h>
#include "kernel.h"


// const int INF = 999999;

// __device__ int end;

// __global__ void MT(){

//   end = 1;
//   printf("Kernel: %d\n", end);
// }

#define MT 1
#define IMT 0
#define BD 0
const int SIZE = 2048;
int main(){
  // float memTime, calcTime;
  cudaError_t devError = cudaSetDevice(0);
  // std::cerr << "Error: " << cudaGetErrorString(devError) << std::endl;
  // cudaDeviceProp prop;
  // devError = cudaGetDeviceProperties ( &prop,0);
  // cout << "MaxthreadsPerBlock: " << prop.maxThreadsPerBlock << endl;
  // cout << "Name: " << prop.name << endl;
  // cout << "RegPerBlock: " << prop.regsPerBlock << endl;
  // int SIMTYPE = 1;
  int B = 1024;
  int T = 128; 
  for(int S = SIZE; S <= SIZE; S<<=1){
    cout << "Timing: " << S  << "x" << S << "Size" << endl;
      // Declare simulation variables
      // int cell, row, col, nrow, ncol, ncell;
      // char simType[20];
      std::ofstream fout;

      // Initialize simulator
      fireSim sim(S,S);
      struct timeval start, fin;

    sim.init();
    sim.updateSpreadData();

    // Allocate Roth Data for GPU
    float* gpuRoth;
    int* gpuTime;
    int* timeSteppers;
    float* loc_L_n;
    float* loc_burnDist;
    bool* check;
    gpuRoth = (float*)malloc(sim.simDimX*sim.simDimY*3*sizeof(float));
    gpuTime = (int*)malloc(sim.simDimX*sim.simDimY*sizeof(int));
    timeSteppers = (int*)malloc(2*sizeof(int));
    loc_L_n = (float*)malloc(16*sizeof(float));
    check = (bool*)malloc(sim.simDimX*sim.simDimY*sizeof(bool));
    loc_burnDist = (float*)malloc(8*sim.simDimX*sim.simDimY*sizeof(float));


    for(int k = 0, cell = 0, tcell = 0; k < sim.simDimX; k++){
      for(int c = 0; c < sim.simDimY; c++, cell+=3, tcell++){
        // cout << cell << endl;
        gpuRoth[cell + 0] = sim.rothData[k][c].x;
        gpuRoth[cell + 1] = sim.rothData[k][c].y;
        gpuRoth[cell + 2] = sim.rothData[k][c].z;
        gpuTime[tcell] = sim.ignTime[tcell];

        check[tcell] = false;
      }
    }

    // Allocate Time data for GPU 
    // float* timeSteppers = new float[2];
    // cout << "CPU: " << endl;
    for(int i = 0; i < 16; i++){
      loc_L_n[i] = sim.L_n[i];
    }

    timeSteppers[0] = 0;
    timeSteppers[1] = 0;
    // timeSteppers[1] = INF;

    char simType[20];

    sprintf(simType, "../out/MT");

    // sprintf(simType, "../out/GPU_DEBUG");
    // sprintf(simType, "../out/GPU_DEBUG");
   
    // Allocate Cuda Variables
    gettimeofday(&start, NULL);
    int *g_ignTime;
    float *g_rothData;
    int *g_times;
    float *g_L_n;

    cudaError_t err = cudaMalloc( (void**) &g_ignTime, sim.simDimX*sim.simDimY*sizeof(int));
    err = cudaMalloc( (void**) &g_rothData, sim.simDimX*sim.simDimY*3*sizeof(float));
    err = cudaMalloc( (void**) &g_times, 2*sizeof(int));
    err = cudaMalloc( (void**) &g_L_n, 16*sizeof(float));

    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
      }

    err = cudaMemcpy(g_ignTime, gpuTime, sim.simDimX*sim.simDimY*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_rothData, gpuRoth, sim.simDimX*sim.simDimY*3*sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_times, timeSteppers, 2*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_L_n, loc_L_n, 16*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Kernel Loop
    int counter = 0;
    // terminate = 0;
    cout << "Kicking off Kernels" << endl;
    typeof(syncCounter) terminate = -1;
//    int B = 1024;
    // int T = 100;
//    int T = sim.simDimX*sim.simDimY / B;
    // int B = S;
    // int T = S;

    //if(T >= 1024){
//      T = S;
    if(S < B)
      B = S;
    if(S < T) 
      T = S; 
      // B = sim.simDimX*sim.simDimY / T;
    //}
    while(terminate <= 0){
    // while(counter < 1969){
      counter++;
      // Do calculations
     MinTime<<<B,T>>>(g_ignTime, g_rothData, 
                           g_times, g_L_n, sim.simDimX*sim.simDimY,
                           sim.simDimX, sim.simDimY);
      // Update Time Kernel 
      timeKernelMT<<<1,1>>>(g_times);

      // cudaDeviceSynchronize();
      err = cudaMemcpyFromSymbol(&terminate, end, sizeof(end), 0, 
                                 cudaMemcpyDeviceToHost);
      // err = cudaMemcpyFromSymbol(&terminate, syncCounter, sizeof(syncCounter), 0, 
      //                            cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
          std::cerr << "Error copying from GPU: " << cudaGetErrorString(err) << std::endl;
          exit(1);
      }
      // cout << terminate <<endl;
      // if(terminate < sim.simDimX*sim.simDimY)
      //   terminate = -1;

      if(terminate < 4)
        terminate = -1;
    }
    int finishCount = 0;
    // Catch last corner to terminate simulation
    while(finishCount <= 3){
      counter++;
      finishCount++;
      // Do calculations
      MinTime<<<B,T>>>(g_ignTime, g_rothData, 
                           g_times, g_L_n, sim.simDimX*sim.simDimY,
                           sim.simDimX, sim.simDimY);
      // Update Time Kernel 
      timeKernelMT<<<1,1>>>(g_times);
    }
    terminate = 0;
    // cudaEventRecord(end, 0);
    // cudaEventSynchronize(end);

    // cudaEventElapsedTime( &calcTime, start, end);
    cout << "Simulation Complete" << endl;
    // Copy back to device
    err = cudaMemcpy(gpuTime, g_ignTime, sim.simDimX*sim.simDimY*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying from GPU: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

      // cudaEventRecord(m_end, 0);
      // cudaEventSynchronize(m_end);

      // cudaEventElapsedTime(&memTime, m_start, m_end);

      gettimeofday(&fin, NULL);

      double t_init = fin.tv_usec + fin.tv_sec * 1000000.0;
      t_init -= start.tv_usec + start.tv_sec * 1000000.0;
      t_init /= 1000000.0;   
      std::cout << "Processing init on " << sim.simDimX << " cells took " << t_init << " seconds" << std::endl;
      
    // cudaEventDestroy( start );
    // cudaEventDestroy( end );
    // cudaEventDestroy( m_start );
    // cudaEventDestroy( m_end );

      // Free memory
      cudaFree(g_ignTime);
      cudaFree(g_rothData);
      cudaFree(g_times);
      cudaFree(g_L_n);

      // Write data to file
      char threadNum[21];
      sprintf(threadNum, "_%d_%d", sim.simDimX, sim.simDimY);
      char csv[] = ".csv";
      strcat(simType,threadNum);
      strcat(simType,csv);
      fout.open(simType);
      printf("Using %d Blocks and %d Threads with %d Iterations\n", B,T,counter);
      printf("Writing to %s\n", simType);
      for(int i = 0; i < sim.simDimX*sim.simDimY; i++){
        // std::cout << ignTime[i] << " ,";
        if(i %sim.simDimX == 0 && i !=0){
            // std::cout << std::endl;
            fout << '\n';
        }
        // fout << (int)sim.ignTime[i] << " ";
        // fout << (int)ignTimeNew[i] << " ";
        fout << gpuTime[i] / 100<< " ";
      }
      fout.close();
   cout << "-------------" << endl << endl;
   cout << gpuTime[0] << endl;
   }


   return 0;
}
