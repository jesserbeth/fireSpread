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
   for(int S = 64; S <= 64; S<<=1){
    cout << "Timing: " << S << "x" << S << " Input" << endl;
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
    float* timeSteppers;
    float* loc_L_n;
    float* loc_burnDist;
    bool* check;
    gpuRoth = (float*)malloc(sim.simDimX*sim.simDimY*3*sizeof(float));
    gpuTime = (int*)malloc(sim.simDimX*sim.simDimY*sizeof(int));
    timeSteppers = (float*)malloc(2*sizeof(float));
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

    sprintf(simType, "../out/IMT");

    // sprintf(simType, "../out/GPU_DEBUG");
    // sprintf(simType, "../out/GPU_DEBUG");

    // Allocate Cuda Variables
    gettimeofday(&start, NULL);
    int *g_ignTime_in;
    int *g_ignTime_out;
    float *g_rothData;
    int *g_times;
    float *g_L_n;
    int *g_ignTime_step;
    bool  *g_check;

    cudaError_t err = cudaMalloc( (void**) &g_ignTime_in, sim.simDimX*sim.simDimY*sizeof(int));
    err = cudaMalloc( (void**) &g_ignTime_out, sim.simDimX*sim.simDimY*sizeof(int));
    err = cudaMalloc( (void**) &g_rothData, sim.simDimX*sim.simDimY*3*sizeof(float));
    err = cudaMalloc( (void**) &g_times, 2*sizeof(int));
    err = cudaMalloc( (void**) &g_L_n, 16*sizeof(float));
    err = cudaMalloc( (void**) &g_check, sim.simDimX*sim.simDimY*sizeof(bool));
    err = cudaMalloc( (void**) &g_ignTime_step, sim.simDimX*sim.simDimY*sizeof(float));

    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
      }

    err = cudaMemcpy(g_ignTime_in, gpuTime, sim.simDimX*sim.simDimY*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_ignTime_out, gpuTime, sim.simDimX*sim.simDimY*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_rothData, gpuRoth, sim.simDimX*sim.simDimY*3*sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_times, timeSteppers, 2*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_L_n, loc_L_n, 16*sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_ignTime_step, gpuTime, sim.simDimX*sim.simDimY*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(g_check, check, sim.simDimX*sim.simDimY*sizeof(bool), cudaMemcpyHostToDevice);

    // Kernel Loop
    int counter = 0;
    // terminate = 0;
    cout << "Kicking off Kernels" << endl;
    typeof(syncCounter) terminate = -1;
    // int B = 128;
    // int T = 128;
    int B = S;
    int T = S;
    if(T >= 1024){
      T = 512;
      B = sim.simDimX*sim.simDimY / T;
    }
    while(terminate <= 0){
    // while(counter < 34){
      counter++;
      // ITERATIVE MINIMAL TIME
        // Do calculations
        ItMinTime<<<B,T>>>(g_ignTime_in,g_ignTime_out, g_ignTime_step, g_rothData, 
        // ItMinTime<<<1,1>>>(g_ignTime_in,g_ignTime_out, g_ignTime_step, g_rothData,
                           g_times, g_L_n, g_check, sim.simDimX*sim.simDimY,
                           sim.simDimX, sim.simDimY);
        // cout << "step caclulated\n";
        // Copy from output to write
        copyKernelIMT<<<B,T>>>(g_ignTime_in, g_ignTime_step,
                            g_check, sim.simDimX*sim.simDimY);

        cudaDeviceSynchronize();
        // for(int k = 0; k < sim.simDimX*sim.simDimY; k++)
        //   cout << check[k] <<  " ";
        // cout << endl;
        err = cudaMemcpyFromSymbol(&terminate, end, sizeof(end), 0, 
                                   cudaMemcpyDeviceToHost);
        // err = cudaMemcpyFromSymbol(&terminate, syncCounter, sizeof(syncCounter), 0, 
        //                            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Error copying from GPU: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
        // cout << terminate <<endl;
        if(terminate < sim.simDimX*sim.simDimY)
          terminate = -1;

        // cout << counter <<endl;
        // Swap Pointers for loop
        int *swap = g_ignTime_in;
        g_ignTime_in = g_ignTime_out;
        g_ignTime_out = swap;

    }
    terminate = 0;
    // cudaEventRecord(end, 0);
    // cudaEventSynchronize(end);

    // cudaEventElapsedTime( &calcTime, start, end);
    cout << "Simulation Complete" << endl;
    // Copy back to device
    err = cudaMemcpy(gpuTime, g_ignTime_in, sim.simDimX*sim.simDimY*sizeof(int), cudaMemcpyDeviceToHost);
    // err = cudaMemcpy(gpuRoth, g_rothData, sim.simDimX*sim.simDimY*3*sizeof(float), cudaMemcpyDeviceToHost);
    // err = cudaMemcpy(timeSteppers, g_times, 2*sizeof(float), cudaMemcpyDeviceToHost);
    // err = cudaMemcpy(sim.L_n, g_L_n, 16*sizeof(float), cudaMemcpyDeviceToHost);
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
      cudaFree(g_ignTime_in);
      cudaFree(g_ignTime_out);
      cudaFree(g_rothData);
      cudaFree(g_times);
      cudaFree(g_L_n);
      // cudaFree(g_burnDist);
    cudaFree(g_ignTime_step);
    cudaFree(g_check);
// #endif
// #if BD
//     cudaFree(g_burnDist);
// #endif
//       if (err != cudaSuccess) {
//         std::cerr << "Error copying from GPU: " << cudaGetErrorString(err) << std::endl;
//         exit(1);
//     }

      // //////////// Debugging output test
      // for(int i = 0; i < sim.simDimX*sim.simDimY; i++){
      //   if(gpuTime[i] != INF)
      //     cout << gpuTime[i] << " ";
      // }
      // cout << endl;



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
        fout << (int)gpuTime[i] /100 << " ";
      }
      fout.close();
      cout << terminate << endl;
    err = cudaMemcpyToSymbol(end, &terminate, sizeof(end), 0, 
                             cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying from GPU: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
   }

   return 0;
}