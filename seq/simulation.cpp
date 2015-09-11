#include <iostream>
#include <fstream>
#include "fireSim.h"
#include <sys/time.h>

const int INF = 9999999;

#define PROFILE 1
#define BURNDIST 0
#define MT 1
#define IMT 0

// enum simulation_type {
//    BURNDIST = 0,
//    MT, 
//    IMT
// };

int main(){
   for(int T = 2048; T <= 2048; T<<=1){
   // for(int T = 256; T <= 256; T<<=1){
      // Declare simulation variables
      int cell, row, col, nrow, ncol, ncell;
      // char simType[20];
      std::ofstream fout;

      // Initialize simulator
      fireSim sim(T,T);
      struct timeval start, fin;
      float pi = 3.14159;
      float ROS = 0;
      float superSize = sqrt(pow(sim.cellSize, 2) + pow(sim.cellSize*2, 2));
      /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW*/
      static int nCol[16] =        {  0,  1,  1,  1,  0, -1, -1, -1, -1, 1, 2, 2, 1, -1, -2, -2};
      static int nRow[16] =        {  1,  1,  0, -1, -1, -1,  0,  1, 2, 2, 1, -1, -2, -2, -1, 1};
      
      // 

   #if PROFILE
      gettimeofday(&start, NULL);
      sim.init();
      // End Timer
      gettimeofday(&fin, NULL);
      double t_init = fin.tv_usec + fin.tv_sec * 1000000.0;
      t_init -= start.tv_usec + start.tv_sec * 1000000.0;
      t_init /= 1000000.0;   
      std::cout << "Processing init on " << sim.simDimX << " cells took " << t_init << " seconds" << std::endl;
      
      gettimeofday(&start, NULL);
      sim.updateSpreadData();
      // End Timer
      gettimeofday(&fin, NULL);
      double t_upSpread = fin.tv_usec + fin.tv_sec * 1000000.0;
      t_upSpread -= start.tv_usec + start.tv_sec * 1000000.0;
      t_upSpread /= 1000000.0;   
      std::cout << "Processing updateSpreadData on " << sim.simDimX << " cells took " << t_upSpread << " seconds" << std::endl;
   #else
      sim.init();
      sim.updateSpreadData();
   #endif


   #if MT
      /////////////////////////////////////////////////////////////////////////////
      //                              Minimal Time
      /////////////////////////////////////////////////////////////////////////////
      std::cout << "Beginning Simulation (Minimal Time)" << std::endl;
      gettimeofday(&start, NULL);
      int counter = 0;
      char simType[20];
      sprintf(simType, "../out/MT");
      while(sim.timeNext < INF){
        sim.timeNow = sim.timeNext;
        sim.timeNext = INF;
        counter++;

        // Loop through all cells
        // for(cell = 0; cell < CELLS; cell++){
        for ( cell=0, row=0; row<sim.simDimX; row++ ){
            for ( col=0; col<sim.simDimY; col++, cell++ ){
              // printf("Cell: %d\n", cell);
                if(sim.timeNext > sim.ignTime[cell] && sim.ignTime[cell] > sim.timeNow){
                    sim.timeNext = sim.ignTime[cell];
                    // printf("Hitting here: %d \n", cell);
                }
                else if( sim.ignTime[cell] == sim.timeNow){
                    for(int n = 0; n < 16; n++){
                    // for(int n = 0; n < 8; n++){
                        // find neighbor cell index
                        nrow = row + nRow[n];
                        ncol = col + nCol[n];
                        // std::cout << row << ' ' << col << ' ' << std::endl;
                        if ( nrow<0 || nrow>= sim.simDimX || ncol<0 || ncol>=  sim.simDimY )
                            continue;
                        ncell = ncol + nrow*sim.simDimY;

                        // If neighbor is unburned
                        if(sim.ignTime[ncell] > sim.timeNow){
                            // compute ignition time
                            ROS = sim.rothData[row][col].x * (1.0 - sim.rothData[row][col].y) / 
                                  (1.0 - sim.rothData[row][col].y * cos(sim.rothData[row][col].z * pi/180));

                            float ignTimeNew = sim.timeNow + (sim.L_n[n] / ROS) * 100;

                            if(ignTimeNew < sim.ignTime[ncell]){
                                sim.ignTime[ncell] = (int)ignTimeNew;
                            }
                            if(ignTimeNew < sim.timeNext){
                                sim.timeNext = (int)ignTimeNew;
                            }
                        }
                    }
                }
            }
        }
      }
      std::cout << "End of Simulation" << std::endl;
      // End Timer
   #if PROFILE
      gettimeofday(&fin, NULL);
      double t_sim = fin.tv_usec + fin.tv_sec * 1000000.0;
      t_sim -= start.tv_usec + start.tv_sec * 1000000.0;
      t_sim /= 1000000.0;   
      std::cout << "Processing simulation on " << sim.simDimX << " cells took " << t_sim << " seconds" << std::endl;
   #endif
   #endif



   #if IMT
      /////////////////////////////////////////////////////////////////////////////
      //                          Iterative Minimal Time
      /////////////////////////////////////////////////////////////////////////////
      std::cout << "Beginning Simulation (Iterative Minimal Time)" << std::endl;
      gettimeofday(&start, NULL);
      float ignCell = 0.;
      float ignCellNew = 0.;
      float ignTimeMin = INF;
      int simCounter = 0;
      bool* check = new bool[sim.simDimX*sim.simDimY];
      for(int z = 0; z < sim.simDimX*sim.simDimY; z++){
          check[z] = false;
      }

      // int counter = 0;
      char simType[20];
      sprintf(simType, "../out/IMT");
      while(simCounter < sim.simDimX*sim.simDimY){
        // counter++;

        // Loop through all cells
        // for(cell = 0; cell < CELLS; cell++){
        for ( cell=0, row=0; row<sim.simDimX; row++ ){
            for ( col=0; col<sim.simDimY; col++, cell++ ){
                if(check[cell] == true)
                    continue;
                // Check for simulation completion
                ignCell = sim.ignTime[cell];
                ignCellNew = sim.ignTimeNew[cell];
                // std::cout << ignCell << ' ' << ignTimeNew[cell];
                if(fabs(ignCell - ignCellNew) < .00001 && ignCell != INF
                        && ignCellNew != INF && check[cell] != true){
                    simCounter++;
                    check[cell] = true;
                    continue;
                }

                if(ignCell > 0){
                    // ignTimeMin = INF;
                    ignTimeMin = INF;
                    // Loop through neighbors
                    for(int n = 0; n < 16; n++){
                        // find neighbor cell index
                        nrow = row + nRow[n];
                        ncol = col + nCol[n];
                        // std::cout << row << ' ' << col << ' ' << std::endl;
                        if ( nrow<0 || nrow>= sim.simDimX || ncol<0 || ncol>=  sim.simDimY )
                            continue;
                        ncell = ncol + nrow*sim.simDimY;

                        ROS = sim.rothData[nrow][ncol].x * (1.0 - sim.rothData[nrow][ncol].y) / 
                              (1.0 - sim.rothData[nrow][ncol].y * cos(sim.rothData[nrow][ncol].z * pi/180));
                        float ignTimeNew = sim.ignTime[ncell] + (sim.L_n[n] / ROS) *100;
                        ignTimeMin = ignTimeNew*(ignTimeNew < ignTimeMin) + ignTimeMin*(ignTimeNew >= ignTimeMin);
                    }
                    sim.ignTimeNew[cell] = (int)ignTimeMin;
                }
            }
        }

        // Swap pointers to loop
        float *temp = sim.ignTime;
        sim.ignTime = sim.ignTimeNew;
        sim.ignTimeNew = temp;
      }
      std::cout << "End of Simulation" << std::endl;
      // End Timer
   #if PROFILE
      gettimeofday(&fin, NULL);
      double t_sim = fin.tv_usec + fin.tv_sec * 1000000.0;
      t_sim -= start.tv_usec + start.tv_sec * 1000000.0;
      t_sim /= 1000000.0;   
      std::cout << "Processing simulation on " << sim.simDimX << " cells took " << t_sim << " seconds" << std::endl;
   #endif
   #endif



   #if BURNDIST
      /////////////////////////////////////////////////////////////////////////////
      //                            Burning Distances
      /////////////////////////////////////////////////////////////////////////////
      std::cout << "Beginning Simulation (Burning Distances)" << std::endl;
      int corner = 0;
      char simType[20];
      sprintf(simType, "../out/BD");
      gettimeofday(&start, NULL);
      
      float t = 0.0;
      while(corner < 4){   
        for ( cell=0, row=0; row<sim.simDimX; row++ ){
            for ( col=0; col<sim.simDimY; col++, cell++ ){
                // check not "ignited"
               if(sim.ignTime[cell] == INF){
                    continue;
                }
                // check neighbors for ignition
                for(int n = 0; n < 8; n++){
                // for(int n = 0; n < 16; n++){
                    nrow = row + nRow[n];
                    ncol = col + nCol[n];
                    if ( nrow<0 || nrow>=sim.simDimX || ncol<0 || ncol>=sim.simDimY )
                        continue;
                    ncell = ncol + nrow*sim.simDimY;

                    // check for already lit
                    if(sim.ignTime[ncell] < INF){
                        continue;
                    }
                    // Calc roth values

                    ROS = sim.rothData[row][col].x * (1.0 - sim.rothData[row][col].y) / 
                        (1.0 - sim.rothData[row][col].y * cos(sim.rothData[row][col].z * pi/180));

                    // Burn distance 
                    sim.burnDist[ncell][n] = sim.burnDistance(sim.burnDist[ncell][n], 
                                                              ROS,
                                                              sim.timeStep);
                    // Propogate fire step:
                    if(sim.burnDist[ncell][n] == 0){
                        sim.ignTimeNew[ncell] = t;
                        if(nrow == (sim.simDimX-1) && ncol == (sim.simDimY-1)){
                            corner += 1;
                        }
                        if(nrow == 0 && ncol == (sim.simDimY-1)){
                            corner += 1;
                        }
                        if(nrow == 0 && ncol == 0){
                            corner += 1;
                        }
                        if(nrow == (sim.simDimX-1) && ncol == 0){
                            corner += 1;
                        }
                    }

                }
            }
        }
        for(int i = 0; i < sim.simDimX*sim.simDimY; i++){
            if(sim.ignTimeNew[i] < INF){
                sim.ignTime[i] = sim.ignTimeNew[i];
                sim.ignTimeNew[i] = INF;
            }
        }
        if(corner == 4)
            break;
         t+= sim.timeStep;
      }

      std::cout << "End of Simulation" << std::endl << std::endl;
      // End Timer
   #if PROFILE
      gettimeofday(&fin, NULL);
      double t_sim = fin.tv_usec + fin.tv_sec * 1000000.0;
      t_sim -= start.tv_usec + start.tv_sec * 1000000.0;
      t_sim /= 1000000.0;   
      std::cout << "Processing simulation on " << sim.simDimX << " cells took " << t_sim << " seconds" << std::endl;
   #endif
   #endif


      // Write data to file
      char threadNum[21];
      sprintf(threadNum, "_%d_%d", sim.simDimX, sim.simDimY);
      char csv[] = "_S.csv";
      strcat(simType,threadNum);
      strcat(simType,csv);
      fout.open(simType);
      printf("Writing out to %s\n", simType);

      for(int i = 0; i < sim.simDimX*sim.simDimY; i++){
        // std::cout << ignTime[i] << " ,";
        if(i %sim.simDimX == 0 && i !=0){
            // std::cout << std::endl;
            fout << '\n';
        }
        fout << (int)sim.ignTime[i] /100 << " ";
        // fout << (int)ignTimeNew[i] << " ";
      }
      fout.close();

   #if PROFILE
      double t_tot = t_sim + t_init + t_upSpread;
      std::cout << "---------------- PROFILE -----------------" << std::endl;
      std::cout << "Initialization Time: " << (t_init / t_tot)*100 << std::endl;
      std::cout << "Update Spread Time: " << (t_upSpread / t_tot)*100 << std::endl;
      std::cout << "Simulation Time: " << (t_sim / t_tot)*100 << std::endl<< std::endl<< std::endl;
   #endif
      
   }

   return 0;
}