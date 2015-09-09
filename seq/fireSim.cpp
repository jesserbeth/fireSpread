#include "fireSim.h"
#include <algorithm>
#include <cmath>


const int INF = 9999999;

#define b printf("%u\n", __LINE__);


template<typename T> T* GISToFloatArray(char*, int, int);
template int* GISToFloatArray<int>(char*, int, int);
template float* GISToFloatArray<float>(char*, int, int);

/*
Constructor: builds simplest test case for testing code
*/
fireSim::fireSim(int _x, int _y){
   std::cout << "Initializing Simulation to Test Setting" << std::endl;

   // declare 2d map data
   simDimX = _x;
   simDimY = _y;

   _models = sim::readFuelModels("default.fmd");
   _moistures = sim::readFuelMoistures("../data/kyle.fms");
   int numModels = _models.size();
   int numMoistModels = _moistures.size();


   timeOfArrival = new float*[simDimX];
   rothData = new vec4*[simDimX];
   originalTimeOfArrival = new float*[simDimX];
   orthoSpreadRate = new vec4*[simDimX];
   diagSpreadRate = new vec4*[simDimX];
   orthoMaxSpreadRate = new vec4*[simDimX];
   diagMaxSpreadRate = new vec4*[simDimX];
   orthoBurnDistance = new vec4*[simDimX];
   diagBurnDistance = new vec4*[simDimX];
   updateStamp = new int*[simDimX];
   sourceDataTexture = new point*[simDimX];

   crownThreshold = new float*[simDimX];
   crownActiveRate = new float*[simDimX];
   canopyHeight = new float*[simDimX];
   spreadData = new float*[simDimX];

   // rothermel vals
   fuelTexture = NULL;
   slopeAspectElevationTexture = new vec3[simDimX*simDimY];

   deadSAVBurnableBuffer = new vec4[numModels];
   dead1hBuffer = new vec4[numModels];
   dead10hBuffer = new vec4[numModels];
   dead100hBuffer = new vec4[numModels];
   liveHBuffer = new vec4[numModels];
   liveWBuffer = new vec4[numModels];
   fineDeadExtinctionsDensityBuffer = new vec4[numModels];
   areasReactionFactorsBuffer = new vec4[numModels];
   slopeWindFactorsBuffer = new vec4[numModels];
   residenceFluxLiveSAVBuffer = new vec4[numModels];
   fuelSAVAccelBuffer = new vec2[numModels];

   windTexture = new vec2*[simDimX];
   deadMoisturesTexture = new vec3[numMoistModels];
   liveMoisturesTexture = new vec2[numMoistModels];


   for(int i = 0; i < simDimX; i++){
      timeOfArrival[i] = new float[simDimY];
      rothData[i] = new vec4[simDimY];
      originalTimeOfArrival[i] = new float[simDimY];
      orthoSpreadRate[i] = new vec4[simDimY];
      diagSpreadRate[i] = new vec4[simDimY];
      orthoMaxSpreadRate[i] = new vec4[simDimY];
      diagMaxSpreadRate[i] = new vec4[simDimY];
      orthoBurnDistance[i] = new vec4[simDimY];
      diagBurnDistance[i] = new vec4[simDimY];
      updateStamp[i] = new int[simDimY];
      sourceDataTexture[i] = new point[simDimY];

      crownThreshold[i] = new float[simDimY];
      crownActiveRate[i] = new float[simDimY];
      canopyHeight[i] = new float[simDimY];
      spreadData[i] = new float[simDimY];

      // rothermel
      windTexture[i] = new vec2[simDimY];
   }

   startTime = 0.; 
   baseTime = 0.;
   endTime = 1000.;
   lastStamp = 0.;
   currentStamp = 0.;
   accelerationConstant = 1.0;

   fuelTextureFile = new char[18];
   fuelTextureFile = "../data/fixed.fuel";
   slopeAspectElevationTextureFile = new char[17];
   slopeAspectElevationTextureFile = "../data/fixed.dem";

   // Simulation Data Members
   timeNow = 0.0;
   timeNext = 0.0;
   ignTime = new float[simDimX * simDimY];
   ignTimeNew = new float[simDimX * simDimY];
   burnDist = new float*[simDimX * simDimY];
   for(int i = 0; i < simDimX * simDimY; i++){
      burnDist[i] = new float[8];
   }

   cellSize = 300;
   L_n = new float[16];
   float orthoSize = cellSize;
   float diagSize = cellSize * sqrt(2);
   float superSize = sqrt(pow(cellSize, 2) + pow(cellSize*2, 2));
   static float L_n_tmp[16] =  { orthoSize, diagSize, orthoSize, diagSize, orthoSize, diagSize,
                                 orthoSize, diagSize, superSize,superSize,superSize,superSize,
                                 superSize,superSize,superSize,superSize};
   for(int i = 0; i < 16; i++){
      L_n[i] = L_n_tmp[i];
   }
   // cout << "constructed" << endl;
}

/*
Destructor: builds simplest test case for testing code
*/
fireSim::~fireSim(){
   // delete all memory: need to be more clever with more complex sims
   int sizeX = 256;
   int sizeY = 256;

   // blah blah
      // std::cout << "Deallocating memory" << std::endl;

   simDimX = sizeX;
   simDimY = sizeY;

   // _models = sim::readFuelModels("default.fmd");
   // _moistures = sim::readFuelMoistures("../data/kyle.fms");
   // int numModels = _models.size();


   // for(int i = 0; i < simDimX; i++){
   //    delete timeOfArrival[i];
   //    delete rothData[i];
   //    delete fuelTexture[i];;
   //    delete originalTimeOfArrival[i];;
   //    delete orthoSpreadRate[i];
   //    delete diagSpreadRate[i];
   //    delete orthoMaxSpreadRate[i];
   //    delete diagMaxSpreadRate[i];
   //    delete orthoBurnDistance[i];
   //    delete diagBurnDistance[i];
   //    delete updateStamp[i];
   //    delete sourceDataTexture[i];;

   //    delete crownThreshold[i];;
   //    delete crownActiveRate[i];;
   //    delete canopyHeight[i];;
   //    delete spreadData[i];;

   //    // rothermel
   //    delete fuelTexture[i];;
   //    delete slopeAspectElevationTexture[i];
   //    delete windTexture[i];
   // }
   // delete timeOfArrival;
   // delete rothData;
   // delete fuelTexture;
   // delete originalTimeOfArrival;
   // delete orthoSpreadRate;
   // delete diagSpreadRate;
   // delete orthoMaxSpreadRate;
   // delete diagMaxSpreadRate;
   // delete orthoBurnDistance;
   // delete diagBurnDistance;
   // delete updateStamp;
   // delete sourceDataTexture;

   // delete crownThreshold;
   // delete crownActiveRate;
   // delete canopyHeight;
   // delete spreadData;

   // // rothermel vals
   // delete fuelTexture;

   // delete deadSAVBurnableBuffer;
   // delete dead1hBuffer;
   // delete dead10hBuffer;
   // delete dead100hBuffer;
   // delete liveHBuffer;
   // delete liveWBuffer;
   // delete fineDeadExtinctionsDensityBuffer;
   // delete areasReactionFactorsBuffer;
   // delete slopeWindFactorsBuffer;
   // delete residenceFluxLiveSAVBuffer;
   // delete fuelSAVAccelBuffer;

   // delete slopeAspectElevationTexture;
   // delete windTexture;
   // delete deadMoisturesTexture;
   // delete liveMoisturesTexture;



   startTime = 0.; 
   baseTime = 0.;
   endTime = 1000.;
   lastStamp = 0.;
   currentStamp = 0.;
   accelerationConstant = 1.0;
   // std::cout << "end of destructor" << std::endl;

}



/*
Function: Init
Input: TBD
Shader base: rothermel
Purpose: Initializes the sim. 
*/
void fireSim::init(){
   // read from files:
   int cell = 0;
   float* slopeTexTmp = NULL;
   GDALAllRegister();
   fuelTexture = GISToFloatArray<int>(fuelTextureFile, simDimX, simDimY);
   slopeTexTmp = GISToFloatArray<float>(slopeAspectElevationTextureFile, simDimX*3, simDimY*3);

   for(int i = 0; i < simDimX; i++){
      for(int j = 0; j < simDimY; j++, cell++){
         timeOfArrival[i][j] = 20.;
         rothData[i][j].x = rothData[i][j].y = rothData[i][j].z = 0.;
         // fuelTexture[i][j] = 0.;
         originalTimeOfArrival[i][j] = 20.;
         orthoSpreadRate[i][j].x = orthoSpreadRate[i][j].y = orthoSpreadRate[i][j].z = orthoSpreadRate[i][j].w = 1.;
         diagSpreadRate[i][j].x = diagSpreadRate[i][j].y = diagSpreadRate[i][j].z = diagSpreadRate[i][j].w = 1.;
         orthoMaxSpreadRate[i][j].x =orthoMaxSpreadRate[i][j].y = orthoMaxSpreadRate[i][j].z = orthoMaxSpreadRate[i][j].w = 100.;
         diagMaxSpreadRate[i][j].x = diagMaxSpreadRate[i][j].y = diagMaxSpreadRate[i][j].z = diagMaxSpreadRate[i][j].w = 100.;
         orthoBurnDistance[i][j].x = orthoBurnDistance[i][j].y = orthoBurnDistance[i][j].z = orthoBurnDistance[i][j].w = .2;
         diagBurnDistance[i][j].x = diagBurnDistance[i][j].y = diagBurnDistance[i][j].z = diagBurnDistance[i][j].w = .2;
         updateStamp[i][j] = 0.;
         sourceDataTexture[i][j].x = sourceDataTexture[i][j].y = 0.;

         crownThreshold[i][j] = 0.0;
         crownActiveRate[i][j] = 100000.;
         canopyHeight[i][j] = 0.;
         spreadData[i][j] = 0.;

         // Rothermel Data Members
         windTexture[i][j].x = windTexture[i][j].y = 0.;

         slopeAspectElevationTexture[cell].x = slopeTexTmp[3*cell];
         slopeAspectElevationTexture[cell].y = slopeTexTmp[3*cell+1];
         slopeAspectElevationTexture[cell].z = slopeTexTmp[3*cell+2];

         ignTime[cell] = INF;
         ignTimeNew[cell] = INF;
         for(int k = 0; k < 8; k++){
            burnDist[cell][k] = L_n[k];
         }
      }
   }

   spreadData[5][5] = 100;
   int ignSpot = simDimX * simDimY / 2 + simDimY / 2;
   ignTime[ignSpot] = 0;
   ignTimeNew[ignSpot] = 0;
   timeStep = 2.0;


   int i = 0;
   for (std::vector<sim::FuelModel>::iterator it = _models.begin(); 
        it != _models.end(); it++, i++)
   {
      dead1hBuffer[i].x = it->effectiveHeatingNumber[sim::Dead1h];
      dead1hBuffer[i].y = it->load[sim::Dead1h];
      dead1hBuffer[i].z = it->areaWeightingFactor[sim::Dead1h];
      dead1hBuffer[i].w = it->fuelMoisture[sim::Dead1h];
      
      dead10hBuffer[i].x = it->effectiveHeatingNumber[sim::Dead10h];
      dead10hBuffer[i].y = it->load[sim::Dead10h];
      dead10hBuffer[i].z = it->areaWeightingFactor[sim::Dead10h];
      dead10hBuffer[i].w = it->fuelMoisture[sim::Dead10h];
      
      dead100hBuffer[i].x = it->effectiveHeatingNumber[sim::Dead100h];
      dead100hBuffer[i].y = it->load[sim::Dead100h];
      dead100hBuffer[i].z = it->areaWeightingFactor[sim::Dead100h];
      dead100hBuffer[i].w = it->fuelMoisture[sim::Dead100h];
      
      liveHBuffer[i].x = it->effectiveHeatingNumber[sim::LiveH];
      liveHBuffer[i].y = it->load[sim::LiveH];
      liveHBuffer[i].z = it->areaWeightingFactor[sim::LiveH];
      liveHBuffer[i].w = it->fuelMoisture[sim::LiveH];
      
      liveWBuffer[i].x = it->effectiveHeatingNumber[sim::LiveW];
      liveWBuffer[i].y = it->load[sim::LiveW];
      liveWBuffer[i].z = it->areaWeightingFactor[sim::LiveW];
      liveWBuffer[i].w = it->fuelMoisture[sim::LiveW];

      fineDeadExtinctionsDensityBuffer[i].x = it->fineDeadRatio;
      fineDeadExtinctionsDensityBuffer[i].y = it->extinctionMoisture;
      fineDeadExtinctionsDensityBuffer[i].z = it->liveExtinction;
      fineDeadExtinctionsDensityBuffer[i].w = it->fuelDensity;

      areasReactionFactorsBuffer[i].x = it->deadArea;
      areasReactionFactorsBuffer[i].y = it->liveArea;
      areasReactionFactorsBuffer[i].z = it->deadReactionFactor;
      areasReactionFactorsBuffer[i].w = it->liveReactionFactor;

      slopeWindFactorsBuffer[i].x = it->slopeK;
      slopeWindFactorsBuffer[i].y = it->windK;
      slopeWindFactorsBuffer[i].z = it->windB;
      slopeWindFactorsBuffer[i].w = it->windE;

      residenceFluxLiveSAVBuffer[i].x = it->residenceTime;
      residenceFluxLiveSAVBuffer[i].y = it->propagatingFlux;
      residenceFluxLiveSAVBuffer[i].z = it->SAV[sim::LiveH];
      residenceFluxLiveSAVBuffer[i].w = it->SAV[sim::LiveW];

      deadSAVBurnableBuffer[i].x = it->SAV[sim::Dead1h];
      deadSAVBurnableBuffer[i].y = it->SAV[sim::Dead10h];
      deadSAVBurnableBuffer[i].z = it->SAV[sim::Dead100h];
      // deadSAVBurnableBuffer[i].w = it->burnable? 100.0f : 0.0f;
      deadSAVBurnableBuffer[i].w = 100.0f;

      fuelSAVAccelBuffer[i].x = it->fuelSAV;
      fuelSAVAccelBuffer[i].y = it->accelerationConstant;
   }

   i = 0;
   for (std::vector<sim::FuelMoisture>::iterator it = _moistures.begin(); 
        it != _moistures.end(); it++, i++)
   {         
         deadMoisturesTexture[i].x = it->dead1h;
         deadMoisturesTexture[i].y = it->dead10h;
         deadMoisturesTexture[i].z = it->dead100h;

         liveMoisturesTexture[i].x = it->liveH;
         liveMoisturesTexture[i].y = it->liveW;
   }
}

/*
Function: updateSpreadData
Input: The necessary inputs are the values that are found in the textures/buffers
       in the FuelModel.h/Simulator.cpp files in Roger's code
Shader base: rothermel
Purpose: This runs rothermel's equations to initialize simulation
*/
void fireSim::updateSpreadData(){
   // This is where rothermel's shader is implemented

   cout << "Updating Spread Data . . ." << endl;
   int cell = 0;   /* row, col, and index of neighbor cell */
   float dist = 10.;
   int counter = 0;
   
   for(int i = 0; i < simDimX; i++){
      for(int j = 0; j < simDimY; j++, cell++){
         // Shader code: int fuelModel = texture2D(fuelTexture, gl_TexCoord[1].st).r;
            // gl_TexCoord[1].st corresponds to fuelTexture.xy
         // int fuelModel = fuelTexture[cell];
         // cout << "FUEL MODEL " << fuelModel << endl;

         // FOR TESTING: Must fix interpolation of fuelModel data
         int fuelModel = 1;

         vec4 dead1h, deadSAVBurnable, dead10h, dead100h, liveH, 
              liveW, fineDeadExtinctionsDensity, areasReactionFactors,
              slopeWindFactors, residenceFluxLiveSAV;
         vec2 fuelSAVAccel;

         vec3 slopeAspectElevation;
         vec2 wind;
         vec3 deadMoistures;
         vec2 liveMoistures;

         // Get data into vars
         // cout << deadSAVBurnableBuffer[fuelModel].x << " " << 
         //         deadSAVBurnableBuffer[fuelModel].y << " " <<
         //         deadSAVBurnableBuffer[fuelModel].z << " " <<
         //         deadSAVBurnableBuffer[fuelModel].w << endl;
         deadSAVBurnable = deadSAVBurnableBuffer[fuelModel];
         // cout << deadSAVBurnable.x << " " << 
         //         deadSAVBurnable.y << " " <<
         //         deadSAVBurnable.z << " " <<
         //         deadSAVBurnable.w << endl;
         if(deadSAVBurnable.w < 50.0){
            cout << "skipping" << endl;
            continue;
         }


         dead1h = dead1hBuffer[fuelModel];
         dead10h = dead10hBuffer[fuelModel];
         dead100h = dead100hBuffer[fuelModel];
         liveH = liveHBuffer[fuelModel];
         liveW = liveWBuffer[fuelModel];
         fineDeadExtinctionsDensity = fineDeadExtinctionsDensityBuffer[fuelModel];
         areasReactionFactors = areasReactionFactorsBuffer[fuelModel];
         slopeWindFactors = slopeWindFactorsBuffer[fuelModel];
         residenceFluxLiveSAV = residenceFluxLiveSAVBuffer[fuelModel];
         fuelSAVAccel = fuelSAVAccelBuffer[fuelModel];
         float fuelSAV = fuelSAVAccel.x;
         float accelerationConstant = fuelSAVAccel.y;

         slopeAspectElevation = slopeAspectElevationTexture[cell];

         wind = windTexture[i][j];
         deadMoistures = deadMoisturesTexture[fuelModel];
         liveMoistures = liveMoisturesTexture[fuelModel];

         float maxSpreadRate = 0.;
         float ellipseEccentricity = 0.;
         float spreadDirection = 0.;
         float spreadModifier = 0.;
         vec3 timeLagClass;


         if (deadSAVBurnable.x > 192.0)
            timeLagClass.x = deadMoistures.x;
         else if (deadSAVBurnable.x > 48.0)
            timeLagClass.x = deadMoistures.y;
         else
            timeLagClass.x = deadMoistures.z;

         if (deadSAVBurnable.y > 192.0)
            timeLagClass.y = deadMoistures.x;
         else if (deadSAVBurnable.y > 48.0)
            timeLagClass.y = deadMoistures.y;
         else
            timeLagClass.y = deadMoistures.z;

         if (deadSAVBurnable.z > 192.0)
            timeLagClass.z = deadMoistures.x;
         else if (deadSAVBurnable.z > 48.0)
            timeLagClass.z = deadMoistures.y;
         else
            timeLagClass.z = deadMoistures.z;

         float weightedFuelModel = 
            timeLagClass.x * dead1h.x * dead1h.y +
            timeLagClass.y * dead10h.x * dead10h.y +
            timeLagClass.z * dead100h.x * dead100h.y;

         float fuelMoistures[5];
         fuelMoistures[0] = timeLagClass.x;
         fuelMoistures[1] = timeLagClass.y;
         fuelMoistures[2] = timeLagClass.z;
         fuelMoistures[3] = liveMoistures.x;
         fuelMoistures[4] = liveMoistures.y;
         // for(int c = 0; c < 5; c++){
         //    cout << fuelMoistures[c] << endl;
         // }

         float liveExtinction = 0.0;
         if(liveH.y > 0.0 || liveW.y > 0.0){
            float fineDeadMoisture = 0.0;
            if (fineDeadExtinctionsDensity.x > 0.0)
               fineDeadMoisture = weightedFuelModel / fineDeadExtinctionsDensity.x;

            liveExtinction =
               (fineDeadExtinctionsDensity.z * 
                (1.0 - fineDeadMoisture / fineDeadExtinctionsDensity.y)) - 0.226;
            liveExtinction = max(liveExtinction, fineDeadExtinctionsDensity.y);
         }
         
         float heatOfIgnition =
            areasReactionFactors.x *
               ((250.0 + 1116.0 * fuelMoistures[0]) * dead1h.z * dead1h.x +
                (250.0 + 1116.0 * fuelMoistures[1]) * dead10h.z * dead10h.x +
                (250.0 + 1116.0 * fuelMoistures[2]) * dead100h.z * dead100h.x) +
            areasReactionFactors.y *
               ((250.0 + 1116.0 * fuelMoistures[3]) * liveH.z * liveH.x +
                (250.0 + 1116.0 * fuelMoistures[4]) * liveW.z * liveW.x);
         heatOfIgnition *= fineDeadExtinctionsDensity.w;

         float liveMoisture = liveH.z * fuelMoistures[3] + liveW.z * fuelMoistures[4];
         float deadMoisture = dead1h.z * fuelMoistures[0] + 
                              dead10h.z * fuelMoistures[1] + 
                              dead100h.z * fuelMoistures[2];

         float reactionIntensity = 0.0;

         if (liveExtinction > 0.0)
            {
               float r = liveMoisture / liveExtinction;
               if (r < 1.0)
                  reactionIntensity += areasReactionFactors.w * (1.0 - 
                                                                 (2.59 * r) + 
                                                                 (5.11 * r * r) - 
                                                      (3.52 * r * r * r));
            }
            if (fineDeadExtinctionsDensity.y > 0.0)
            {
               float r = deadMoisture / fineDeadExtinctionsDensity.y;
               if (r < 1.0)
                  reactionIntensity += areasReactionFactors.z * (1.0 - 
                                                                 (2.59 * r) + 
                                                                 (5.11 * r * r) - 
                                                      (3.52 * r * r * r));
            }

            float heatPerUnitArea = reactionIntensity * residenceFluxLiveSAV.x;
            float baseSpreadRate = 0.0;

            if (heatOfIgnition > 0.0)
               baseSpreadRate = reactionIntensity * residenceFluxLiveSAV.y / heatOfIgnition;
            // cout << "baseSpreadRate" << baseSpreadRate << endl;
            
            float slopeFactor = slopeWindFactors.x * slopeAspectElevation.x * slopeAspectElevation.x;
            float windFactor = 0.0;
            if (wind.x > 0.0)
               windFactor = slopeWindFactors.y * pow(wind.x, slopeWindFactors.z);

            spreadModifier = slopeFactor + windFactor;
            // cout << slopeFactor << " " << windFactor << endl;
            float upslope;
            if (slopeAspectElevation.y >= 180.0)
               upslope = slopeAspectElevation.y - 180.0;
            else
               upslope = slopeAspectElevation.y + 180.0;

            int checkEffectiveWindspeed = 0;
            int updateEffectiveWindspeed = 0;
            float effectiveWindspeed = 0.0;
            if (baseSpreadRate <= 0.0)
            {
               maxSpreadRate = 0.0;
               spreadDirection = 0.0;
// b
            }
            else if (spreadModifier <= 0.0)
            {
               maxSpreadRate = baseSpreadRate;
               spreadDirection = 0.0;
// b
            }
            else if (slopeAspectElevation.x <= 0.0)
            {
               effectiveWindspeed = wind.x;
               maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
               spreadDirection = wind.y;
               checkEffectiveWindspeed = 1;
// b
            }
            else if (wind.x <= 0.0)
            {
               maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
               spreadDirection = upslope;
               updateEffectiveWindspeed = 1;
               checkEffectiveWindspeed = 1;
// b
            }
            else if (fabs(wind.y - upslope) < 0.000001)
            {
               maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
               spreadDirection = upslope;
               updateEffectiveWindspeed = 1;
               checkEffectiveWindspeed = 1;
// b
            }
            else
            {
               float angleDelta;
               if (upslope <= wind.y)
                  angleDelta = wind.y - upslope;
               else
                  angleDelta = 360.0 - upslope + wind.y;
               angleDelta *= 3.14159 / 180.0;
               float slopeRate = baseSpreadRate * slopeFactor;
               float windRate = baseSpreadRate * windFactor;
               float x = slopeRate + windRate * cos(angleDelta);
               float y = windRate * sin(angleDelta);
               float addedSpeed = sqrt(x * x + y * y);
               maxSpreadRate = baseSpreadRate + addedSpeed;

               spreadModifier = maxSpreadRate / baseSpreadRate - 1.0;
               // cout << "spreadmoid: " << spreadModifier << endl;
               if (spreadModifier > 0.0)
                  updateEffectiveWindspeed = 1;
               checkEffectiveWindspeed = 1;

               float addedAngle = 0.0;
               if (addedSpeed > 0.0)
                  addedAngle = asin(clamp(fabs(y) / addedSpeed, -1.0, 1.0));
               float angleOffset = 0.0;
               if (x >= 0.0)
               {
                  if (y >= 0.0)
                     angleOffset = addedAngle;
                  else
                     angleOffset = 2.0 * 3.14159 - addedAngle;
               }
               else
               {
                  if (y >= 0.0)
                     angleOffset = 3.14159 + addedAngle;
                  else
                     angleOffset = 3.14159 - angleOffset;
               }
               spreadDirection = upslope + angleOffset * 180.0 / 3.14159;
               if (spreadDirection > 360.0)
                  spreadDirection -= 360.0;
            }

            if (updateEffectiveWindspeed == 1)
            {
               effectiveWindspeed = pow((spreadModifier * slopeWindFactors.w), (1.0 / slopeWindFactors.z));
            }
            if (checkEffectiveWindspeed == 1)
            {
               float maxWind = 0.9 * reactionIntensity;
               if (effectiveWindspeed > maxWind)
               {
                  if (maxWind <= 0.0)
                     spreadModifier = 0.0;
                  else
                     spreadModifier = slopeWindFactors.y * pow(maxWind, slopeWindFactors.z);
                  maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
                  effectiveWindspeed = maxWind;
               }
            }
            ellipseEccentricity = 0.0;
            if (effectiveWindspeed > 0.0)
            {
               float lengthWidthRatio = 1.0 + 0.002840909 * effectiveWindspeed;
               ellipseEccentricity = sqrt(lengthWidthRatio * lengthWidthRatio - 1.0) / lengthWidthRatio;
            }
            //maxSpreadRate = maxSpreadRate * (1.0 - exp(-accelerationConstant * burnTime / 60.0));
            //float firelineIntensity = 
            // 3.4613 * (384.0 * (reactionIntensity / 0.189275)) * 
            //    (maxSpreadRate * 0.30480060960) / (60.0 * fuelSAV);
            //firelineIntensity =
            // reactionIntensity * 12.6 * maxSpreadRate / (60.0 * fuelSAV);
            float intensityModifier =
               3.4613 * (384.0 * (reactionIntensity / 0.189275)) *
                  (0.30480060960) / (60.0 * fuelSAV);
            // gl_FragData[0] = vec4(maxSpreadRate, 
            //                       ellipseEccentricity, spreadDirection, intensityModifier);

            rothData[i][j].x = maxSpreadRate;
            // cout << maxSpreadRate;
            rothData[i][j].y = spreadDirection;
            // cout << spreadDirection;
            rothData[i][j].z = ellipseEccentricity;
            // cout << ellipseEccentricity << endl;

      }
   }
   // cout << counter << endl;
}

/*
Function: propagateFire
Input: TBD
Shader base: propagateAccel
Purpose: 

void fireSim::propagateFire(){
   // must loop through all points in lattice
   for(int i = 0; i < simDimX; i++){ // loop through rows
      for(int j = 0; j < simDimY; j++){ // loop through cols
         point n,s,e,w,nw,sw,ne,se;
         bool toprow = false;
         bool bottomrow = false;
         bool leftcol = false;
         bool rightcol = false;
         
         float toa = timeOfArrival[i][j];
         // cout << "before cont" << endl;
         if(toa <= startTime)
            continue;
         // cout << "after cont" << endl;
         float sourceData[4] = {sourceDataTexture[i][j].x, sourceDataTexture[i][j].y, i, j};
         point sourceDir;
         sourceDir.x = sourceData[0];
         sourceDir.y = sourceData[1];
         float originalToa = originalTimeOfArrival[i][j];
         float* orthoBurnDistances;
         orthoBurnDistances = new float[4];
         orthoBurnDistances[0] = orthoBurnDistance[i][j].x;
         orthoBurnDistances[1] = orthoBurnDistance[i][j].y;
         orthoBurnDistances[2] = orthoBurnDistance[i][j].z;
         orthoBurnDistances[3] = orthoBurnDistance[i][j].w;
         float* diagBurnDistances;
         diagBurnDistances = new float[4];
         diagBurnDistances[0] = diagBurnDistance[i][j].x;
         diagBurnDistances[1] = diagBurnDistance[i][j].y;
         diagBurnDistances[2] = diagBurnDistance[i][j].z;
         diagBurnDistances[3] = diagBurnDistance[i][j].w;

         // check x bounds
         if(i-1 >= 0){
            nw.x = i-1;
            n.x = i-1;
            ne.x = i-1;
            toprow = true;
         }
         else{
            nw.x = 0;
            n.x = 0;
            ne.x = 0;
            toprow = false;
         }
         if(i+1 < simDimX){
            sw.x = i+1;
            s.x = i+1;
            se.x = i+1;
            bottomrow = true;
         }
         else{
            sw.x = 0;
            s.x = 0;
            se.x = 0;
            bottomrow = false;
         }
         w.x = i;
         e.x = i;
         // check y bounds
         if(j-1 >=0){
            nw.y = j-1;
            w.y = j-1;
            sw.y = j-1;
            leftcol = true;
         }
         else{
            nw.y = 0;
            w.y = 0;
            sw.y = 0;
            leftcol = false;
         }
         if(j+1 < simDimY){
            ne.y = j+1;
            e.y = j+1;
            se.y = j+1;
            rightcol = true;
         }
         else{
            ne.y = 0;
            e.y = 0;
            se.y = 0;
            rightcol = false;

         }
         n.y = j;
         s.y = j;
         // if(toprow == true){
         //    bool updatenw = updateStamp[nw.x][nw.y] == lastStamp;
         //    bool updaten = updateStamp[n.x][n.y] == lastStamp;
         //    bool updatene = updateStamp[ne.x][ne.y] == lastStamp;
         // }
         // if(bottomrow == true){
         //    bool updatesw = updateStamp[sw.x][sw.y] == lastStamp;
         //    bool updates = updateStamp[s.x][s.y] == lastStamp;
         //    bool updatese = updateStamp[se.x][se.y] == lastStamp;
         // }
         // if(leftcol == true)
         //    bool updatew = updateStamp[s.x][s.y] == lastStamp;
         // if(rightcol == true)
         //    bool updatese = updateStamp[se.x][se.y] == lastStamp;

         // check if any updates necessary
         // if(!(updatenw | updaten | updatene | updatew | updatee | updatesw | updates | updatese))
         //    continue;

         bool toaCorrupt = updateStamp[sourceDir.x][sourceDir.y] == lastStamp;
         float newToa = toa;
         float toaLimit = toa;
         if(toaCorrupt || lastStamp == 0){
            newToa = originalToa;
            toaLimit = originalToa;
         }
         // REMEMBER THAT X and Y correspond to pointCoord!!
         point direction;
         direction.x = i;
         direction.y = j;
         float newRate = 0.0;
         float dt = 0.0;

         // check for boundaries as you propagate
         // if(toprow == true){
            // update NW
            {
               float t = timeOfArrival[nw.x][nw.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = diagSpreadRate[nw.x][nw.y].w;
                  float dist = diagBurnDistances[0];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = nw.x;
                     direction.y = nw.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update N
            {
               float t = timeOfArrival[n.x][n.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = orthoSpreadRate[n.x][n.y].w;
                  float dist = orthoBurnDistances[0];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = n.x;
                     direction.y = n.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update NE
            {
               float t = timeOfArrival[ne.x][ne.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = diagSpreadRate[ne.x][ne.y].z;
                  float dist = diagBurnDistances[1];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = ne.x;
                     direction.y = ne.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
         // }
         // if()
            // Update W
            {
               float t = timeOfArrival[w.x][w.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = orthoSpreadRate[w.x][w.y].z;
                  float dist = orthoBurnDistances[1];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = w.x;
                     direction.y = w.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update E
            {
               float t = timeOfArrival[e.x][e.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = orthoSpreadRate[e.x][e.y].y;
                  float dist = orthoBurnDistances[2];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = e.x;
                     direction.y = e.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update SW
            {
               float t = timeOfArrival[sw.x][sw.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = diagSpreadRate[sw.x][sw.y].y;
                  float dist = diagBurnDistances[2];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = sw.x;
                     direction.y = sw.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update S
            {
               float t = timeOfArrival[s.x][s.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = orthoSpreadRate[s.x][s.y].x;
                  float dist = orthoBurnDistances[3];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = s.x;
                     direction.y = s.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update SE
               // cout << "OUT" << endl;
            {
               float t = timeOfArrival[se.x][se.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = diagSpreadRate[se.x][se.y].x;
                  float dist = diagBurnDistances[3];
                  float burnTime = dist / rate;
                  t += burnTime;
                  // cout << t << endl;
                  if(t < newToa){
                     newToa = t;
                     direction.x = se.x;
                     direction.y = se.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }

            // if(newToa >= toaLimit || newToa > endTime)
            //    continue;
            // cout << "test" << endl;
            float maxOrthoRates[4] = {orthoMaxSpreadRate[i][j].x,
                                      orthoMaxSpreadRate[i][j].y,
                                      orthoMaxSpreadRate[i][j].z,
                                      orthoMaxSpreadRate[i][j].w};
            float maxDiagRates[4] = {diagMaxSpreadRate[i][j].x,
                                     diagMaxSpreadRate[i][j].y,
                                     diagMaxSpreadRate[i][j].z,
                                     diagMaxSpreadRate[i][j].w};
            float* currentOrthoRates;
            currentOrthoRates = new float[4];
            currentOrthoRates[0] = diagSpreadRate[direction.x][direction.y].x;
            currentOrthoRates[1] = diagSpreadRate[direction.x][direction.y].y;
            currentOrthoRates[2] = diagSpreadRate[direction.x][direction.y].z;
            currentOrthoRates[3] = diagSpreadRate[direction.x][direction.y].w;

            float* currentDiagRates;
            currentDiagRates = new float[4];
            currentDiagRates[0] = diagSpreadRate[direction.x][direction.y].x;
            currentDiagRates[1] = diagSpreadRate[direction.x][direction.y].y;
            currentDiagRates[2] = diagSpreadRate[direction.x][direction.y].z;
            currentDiagRates[3] = diagSpreadRate[direction.x][direction.y].w;

            float _canopyHeight = canopyHeight[i][j];
            if(_canopyHeight > 0.0){
               float _crownActiveRate = crownActiveRate[i][j];
               float _crownThreshold = crownThreshold[i][j];
               float intensityModifier = spreadData[i][j];
               // Ortho Rates
               maxOrthoRates[0] = testCrownRate(currentOrthoRates[0],
                                                maxOrthoRates[0],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxOrthoRates[1] = testCrownRate(currentOrthoRates[1],
                                                maxOrthoRates[1],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxOrthoRates[2] = testCrownRate(currentOrthoRates[2],
                                                maxOrthoRates[2],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxOrthoRates[3] = testCrownRate(currentOrthoRates[3],
                                                maxOrthoRates[3],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               // Diag Rates
               maxDiagRates[0] = testCrownRate(currentDiagRates[0],
                                                maxDiagRates[0],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxDiagRates[1] = testCrownRate(currentDiagRates[1],
                                                maxDiagRates[1],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxDiagRates[2] = testCrownRate(currentDiagRates[2],
                                                maxDiagRates[2],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxDiagRates[3] = testCrownRate(currentDiagRates[3],
                                                maxDiagRates[3],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
            }

            currentOrthoRates = accelerate(currentOrthoRates, maxOrthoRates, dt);
            currentDiagRates = accelerate(currentDiagRates, maxDiagRates, dt);

            // Write results
            timeOfArrival[i][j] = newToa;
            diagSpreadRate[i][j].x = currentDiagRates[0];
            diagSpreadRate[i][j].y = currentDiagRates[1];
            diagSpreadRate[i][j].z = currentDiagRates[2];
            diagSpreadRate[i][j].w = currentDiagRates[3];
            orthoSpreadRate[i][j].x = currentOrthoRates[0];
            orthoSpreadRate[i][j].y = currentOrthoRates[1];
            orthoSpreadRate[i][j].z = currentOrthoRates[2];
            orthoSpreadRate[i][j].w = currentOrthoRates[3];

            timeStamp = currentStamp;
            sourceDataTexture[i][j].x = newRate;

      }
   }
}*/

/*
Function: burnDistance
Input: TBD
Shader base: partialBurn
Purpose: 
*/
void fireSim::burnDistance(){

}

/*
Function: accelerateFire
Input: TBD
Shader base: acceleration
Purpose: 
*/
void fireSim::accelerateFire()   {

}

/*
Function: triggerNextEvent
Input: TBD
Purpose: Step time through simulation
*/
void fireSim::triggerNextEvent(){

}


/*//////////////////////////////////////////////////////////////////
                     Support Functions
//////////////////////////////////////////////////////////////////*/


/*
Function: accelerate
Input: TBD
Purpose: run acceleration code
*/
float* fireSim::accelerate(float* current, float* maxRate, float dt){
   for(int i = 0; i < 4; i++){
      current[i] = min(current[i], maxRate[i]);
   }
   // clamp
   float ratio[4];
   for(int i = 0; i < 4; i++){
      float tmp = current[i] / maxRate[i];
      if(tmp < 0.1){
         ratio[i] = 0.1;
      }
      else if(tmp > 0.9){
         ratio[i] = 0.9;
      }
      else
         ratio[i] = tmp;
   }

   // find timeToMax
   float timeToMax[4];
   for(int i = 0; i < 4; i++){
      timeToMax[i] = -log(1.0 - ratio[i]) / accelerationConstant;
   }

   // clamp
   for(int i = 0; i < 4; i++){
      float tmp = dt / timeToMax[i];
      if(tmp < 0.0){
         tmp = 0.0;
      }
      else if(tmp > 1.0){
         tmp = 1.0;
      }

      current[i] = tmp * (maxRate[i] - current[i]) + current[i];
   }

   return current;
}

/*
Function: testCrownRate
Input: TBD
Purpose: tests crown rate in each update step
*/
float fireSim::testCrownRate(float spreadRate,
               float maxRate,
                    float intensityModifier, 
               float crownActiveRate, 
               float crownThreshold)
{
   if(maxRate <= 0.0)
      return 0.0;

   spreadRate *= 60.0;
   spreadRate /= 3.28;
   maxRate *= 60.0;
   maxRate /= 3.28;
   float firelineIntensity = spreadRate * intensityModifier;
   if(firelineIntensity > crownThreshold){
      float maxCrownRate = 3.34 * maxRate;
      float surfaceFuelConsumption = crownThreshold * spreadRate / firelineIntensity;
      float crownCoefficient = -log(0.1)/(0.9 * (crownActiveRate - surfaceFuelConsumption));
      float crownFractionBurned = 1.0 - exp(-crownCoefficient * (spreadRate - surfaceFuelConsumption));
      if(crownFractionBurned < 0.0)
         crownFractionBurned = 0.0;
      if(crownFractionBurned > 1.0)
         crownFractionBurned = 1.0;
      float crownRate = spreadRate + crownFractionBurned * (maxCrownRate - spreadRate);
      if(crownRate >= crownActiveRate)
         maxRate = max(crownRate, maxRate);

   }
   return maxRate * 3.28 / 60.0;
}

/*
Function: testCrownRate
Input: TBD
Purpose: tests crown rate in each update step
*/
float fireSim::clamp(float val, float flr, float ceiling){
   if(val >= flr && val <= ceiling){
      return val;
   }
   if(val < flr){
      return flr;
   }
   return ceiling;
}


/*
Function: setSimSize
Input: height,width of grid for testing
Purpose: allows size to be set for generation of data in simulation tests
*/
void fireSim::setSimSize(int x, int y){
   simDimX = x;
   simDimY = y;
}

/*
Function: burnDistance
Input: distance, rate, timestep 
Purpose: reduce distance for burning over several timesteps
*/
float fireSim::burnDistance(float dist, float rate, float step){
    // lower distance based on roth rate
        // t = d / r;
        // d = d - r * timeStep
    dist = dist - rate * step;
    if( dist < 0){
        dist = 0;
    }
    return dist;
}


/*
Function: accelerate
Input: TBD
Purpose: run acceleration code
*/
template<typename T>
T* GISToFloatArray(char* fname, int interpWidth, int interpHeight)
{
  // Important note ------ Gdal considers images to be north up
  // the origin of datasets takes place in the upper-left or North-West corner.
  // Now to create a GDAL dataset
  // auto ds = ((GDALDataset*) GDALOpen(fname,GA_ReadOnly));
  GDALDataset* ds = ((GDALDataset*) GDALOpen(fname,GA_ReadOnly));
  if(ds == NULL)
  {
    return NULL;
  }
  
  // Creating a Raster band variable
  // A band represents one whole dataset within a dataset
  // in your case your files have one band.
  GDALRasterBand  *poBand;
  int             nBlockXSize, nBlockYSize;
  int             bGotMin, bGotMax;
  double          adfMinMax[2];
  
  // Assign the band      
  poBand = ds->GetRasterBand( 1 );
  poBand->GetBlockSize( &nBlockXSize, &nBlockYSize );

  // find the min and max
  adfMinMax[0] = poBand->GetMinimum( &bGotMin );
  adfMinMax[1] = poBand->GetMaximum( &bGotMax );
  if( ! (bGotMin && bGotMax) )
    GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);
  int min = adfMinMax[0];
  int max = adfMinMax[1];

  // get the width and height of the band or dataset
  int width = poBand->GetXSize();
  int height = poBand->GetYSize();

  // GDAL can handle files that have multiple datasets jammed witin it
  int bands = ds->GetRasterCount();

  // the float variable to hold the DEM!
  T *pafScanline;
  // std::cout << "Min: " << adfMinMax[0] << " Max: " << adfMinMax[1] << endl;
  int dsize = 256;
  // pafScanline = (T *) CPLMalloc(sizeof(T)*width*height);
  pafScanline = (T *) CPLMalloc(sizeof(T)*interpWidth*interpHeight);

  // Lets acquire the data.  ..... this funciton will interpolate for you
  // poBand->RasterIO(GF_Read,0,0,width,height,pafScanline,width,height,GDT_Float32,0,0);
  poBand->RasterIO(GF_Read,0,0,width,height,pafScanline,interpWidth,interpHeight,GDT_Float32,0,0);
  //        chage these two to interpolate automatically ^      ^

  // The Geotransform gives information on where a dataset is located in the world
  // and the resolution.
  // for more information look at http://www.gdal.org/gdal_datamodel.html
  double geot[6];
  ds->GetGeoTransform(geot);

  // Get the x resolution per pixel(south and west) and y resolution per pixel (north and south)
  // float xres = geot[1];
  // float yres = geot[5];
  // string proj;
  // proj = string(ds->GetProjectionRef());

  // You can get the projection string
  // The projection gives information about the coordinate system a dataset is in
  // This is important to allow other GIS softwares to place datasets into the same
  // coordinate space 
  // char* test = &proj[0];

  // The origin of the dataset 
  // float startx = geot[0]; // east - west coord.
  // float starty = geot[3]; // north - south coord.

  
  // here is some code that I used to push that 1D array into a 2D array
  // I believe this puts everything in the correct order....
  /*for(int i = 0; i < hsize; i++)
  {
    for(int j = 0; j < wsize; j++)
    {
      //cout << i << j << endl << pafS;
      vecs[i][j] = pafScanline[((int)i)*(int)wsize+((int)j)];
      if(vecs[i][j]>0 && vecs[i][j] > max)
      {
          max = vecs[i][j];
      }
      if(vecs[i][j]>0 && vecs[i][j] < min)
      {
          min = vecs[i][j];
      }
    }
   }*/
   //CPLFree(pafScanline);
   return pafScanline;

}