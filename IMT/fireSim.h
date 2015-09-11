#include <iostream>
#include <vector>
#include "FuelModel.h"
#include "FuelMoisture.h"
#include <string>
// Include gdal libraries to parse .dem files
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

using namespace std;

class point{
   public:
      point(){
         x = 0;
         y = 0;
      }
      int x;
      int y;
};

class vec2{
   public: 
      vec2(){
         x = y = -1;
      }
      vec2(float _x, float _y){
         x = _x;
         y = _y;
      }
      vec2 operator=(const vec2& vector){
         if(this == &vector)
            return *this;
         x = vector.x;
         y = vector.y;
         return *this;
      }
      float x,y;
};
class vec3{
   public: 
      vec3(){
         x = y = z = -1;
      }
      vec3(float _x, float _y, float _z){
         x = _x;
         y = _y;
         z = _z;
      }
      vec3 operator=(const vec3& vector){
         if(this == &vector)
            return *this;
         x = vector.x;
         y = vector.y;
         z = vector.z;
         return *this;
      }
      float x,y,z;
};
class vec4{
   public: 
      vec4(){
         x = y = z = w = -1;
      }
      vec4(float _x, float _y, float _z, float _w){
         x = _x;
         y = _y;
         z = _z;
         w = _w;
      }
      vec4 operator=(const vec4& vector){
         if(this == &vector)
            return *this;
         x = vector.x;
         y = vector.y;
         z = vector.z;
         w = vector.w;
         return *this;
      }
      float x,y,z,w;
};

class fireSim{
   public: 
      fireSim(int _x = 100,int _y = 100); // this will set to default test state
      ~fireSim();
      void init();
      void updateSpreadData();
      void propagateFire();
      void burnDistance();
      void accelerateFire();
      void triggerNextEvent();
      float clamp(float, float, float);

      float* accelerate(float*, float*, float);
      float testCrownRate(float, float, float, float, float);
      void setSimSize(int, int);
      float burnDistance(float, float, float);

   // private: 

      // Simulation Data
      vec4** rothData; // x - maxSpreadRate, y - spreadDirection, z - ellipseEccentricity
      float** timeOfArrival;
      float** originalTimeOfArrival;
      vec4** orthoSpreadRate;
      vec4** diagSpreadRate;
      vec4** orthoMaxSpreadRate;
      vec4** diagMaxSpreadRate;
      vec4** orthoBurnDistance;
      vec4** diagBurnDistance;
      int**   updateStamp;
      point** sourceDataTexture;

      float** crownThreshold;
      float** crownActiveRate;
      float** canopyHeight;
      float** spreadData;

      float startTime; 
      float baseTime;
      float endTime;
      int lastStamp;
      int currentStamp;
      float accelerationConstant;
      
      float outputTOA;
      vec4* outputOrthoRates;
      vec4* outputDiagRates;
      int timeStamp;
      float* outputSourceData;

      // Simulation members from test Sim
      float timeNow;
      float timeNext;
      float* ignTime;
      float* ignTimeNew;
      float** burnDist;
      float* L_n;


      // Rothermel Data Members
      int* fuelTexture;
      vec4*  deadSAVBurnableBuffer;
      vec4*  dead1hBuffer;
      vec4*  dead10hBuffer;
      vec4*  dead100hBuffer;
      vec4*  liveHBuffer;
      vec4*  liveWBuffer;
      vec4*  fineDeadExtinctionsDensityBuffer;
      vec4*  areasReactionFactorsBuffer;
      vec4*  slopeWindFactorsBuffer;
      vec4*  residenceFluxLiveSAVBuffer;
      vec2*  fuelSAVAccelBuffer;
      // Roth textures
      vec3* slopeAspectElevationTexture;
      vec2** windTexture;
      vec3* deadMoisturesTexture;
      vec2* liveMoisturesTexture;
      char* fuelTextureFile;
      char* slopeAspectElevationTextureFile;

      int simDimX;
      int simDimY;
      float cellSize;
      float timeStep;

      std::vector<sim::FuelModel> _models;
      std::vector<sim::FuelMoisture> _moistures;
};

