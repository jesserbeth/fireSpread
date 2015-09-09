#include <iostream>
#include <fstream>
#include <math.h>


float burnDistance(float dist, float rate, float timeStep){
    // lower distance based on roth rate
        // t = d / r;
        // d = d - r * timeStep
    dist = dist - rate * timeStep;
    if( dist < 0){
        dist = 0;
    }
    return dist;
}


int main(){
    // vars
    std::ofstream fout;
    fout.open("test.csv");
    int totalTime = 100;
    float timeStep = .5;
    int Rows = 100;
    int Cols = 100;
    int ignMap[Rows*Cols];
    int ignMapStep[Rows*Cols];
    float rateMap[Rows*Cols];
    int stepMap[Rows*Cols];
    // float rateMap[Rows*Cols];
    float baseDistance = 5.;
    float** burnDist = new float*[Rows*Cols];
    for(int i = 0; i < Rows*Cols; i++){
        burnDist[i] = new float[8];
    }

    for(int i = 0; i < Rows*Cols; i++){
        ignMap[i] = 0;
        ignMapStep[i] = 0;
        stepMap[i] = 0;
        rateMap[i] = 0.0;
        for(int j = 0; j < 8; j++){
            if(j % 2 == 1){
                burnDist[i][j] = baseDistance * sqrt(2);
                // std::cout << burnDist[i][j] << std::endl;
            }
            else
                burnDist[i][j] = baseDistance;
        }
    }
    // for(int i = 0; i < Rows*Cols; i++){
    //     for(int j = 0; j < 8; j++){
    //         std::cout << burnDist[i][j] << ' ';
    //     }
    //     if(i % Rows == 0)
    //         std::cout << std::endl;
    // }
    int row, col, cell, ncell, ncol, nrow;
    /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW */
    static int nCol[8] =        {  0,  1,  1,  1,  0, -1, -1, -1};
    static int nRow[8] =        {  1,  1,  0, -1, -1, -1,  0,  1};
   

    //ignite point
    ignMap[5250] = 1;
    rateMap[5250] = 2.;
    stepMap[5250] = 0;
    bool spreadFire = false;
    // int stepCounter = 0;

    // loop through time:
    for(int t = 0; t < 200; t++){
        for ( cell=0, row=0; row<Rows; row++ ){
            for ( col=0; col<Cols; col++, cell++ ){
                // check if already "ignited"
                if(ignMap[cell] == 0){
                    continue;
                }
                // std::cout << row << ' ' << ignMap[cell] << std::endl;
                // check neighbors for ignition
                spreadFire = true;
                for(int n = 0; n < 8; n++){
                    nrow = row + nRow[n];
                    ncol = col + nCol[n];
                    if ( nrow<0 || nrow>=Rows || ncol<0 || ncol>=Cols )
                        continue;
                    ncell = ncol + nrow*Cols;

                    // check for already lit
                    if(ignMap[ncell] == 1){
                        continue;
                    }

                    // Burn distance 
                    // burnDist[ncell] -= 1;
                    burnDist[ncell][n] = burnDistance(burnDist[ncell][n], rateMap[cell], timeStep);
                    // std::cout << burnDist[ncell][n]<< std::endl;
                    if(burnDist[ncell][n] == 0){
                        ignMapStep[ncell] = 1;
                        stepMap[ncell] = t;
                        rateMap[ncell] = rateMap[cell];
                        
                    }
                }
            }
            // stepCounter++;
        }
        for(int i = 0; i < Rows*Cols; i++){
            if(ignMapStep[i] == 1)
                ignMap[i] = 1;
        }
    }

    for(int i = 0; i < Rows*Cols; i++){
        // std::cout << stepMap[i] << " ,";
        if(i %Rows == 0 && i !=0){
            // std::cout << std::endl;
            fout << '\n';
        }
        fout << stepMap[i] << " ";
    }
    fout.close();
}