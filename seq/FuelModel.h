#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string.h>

namespace sim
{

enum FuelClass
{
	Dead1h = 0,
	Dead10h = 1,
	Dead100h = 2,
	LiveH = 3,
	LiveW = 4,
	NumClasses = 5
};

struct FuelModel
{
	FuelModel()
	{
		modelNumber = -9999;
		burnable = false;
	}

	FuelModel(int _modelNumber, 
	          float _Dead1hour, 
			  float _Dead10hour, 
			  float _Dead100hour, 
			  float _liveHerbaceous, 
			  float _liveWoody,
			  float _Dead1hSAV,
			  float _liveHerbaceousSAV,
			  float _liveWoodySAV,
			  float _fuelDepth,
			  float _extinctionMoisture,
			  float _deadHeatContent,
			  float _liveHeatContent)
	{
		burnable = true;
		modelNumber = _modelNumber;
		fuelDepth = _fuelDepth;
		extinctionMoisture = _extinctionMoisture;
		rosAdjustment = 0.0f;
		
		load[Dead1h] = _Dead1hour;
		load[Dead10h] = _Dead10hour;
		load[Dead100h] = _Dead100hour;
		load[LiveH] = _liveHerbaceous;
		load[LiveW] = _liveWoody;
		
		SAV[Dead1h] = _Dead1hSAV;
		SAV[Dead10h] = 109.0;
		SAV[Dead100h] = 30.0;
		SAV[LiveH] = _liveHerbaceousSAV;
		SAV[LiveW] = _liveWoodySAV;

		density[Dead1h] = 32.0;
		density[Dead10h] = 32.0;
		density[Dead100h] = 32.0;
		density[LiveH] = 32.0;
		density[LiveW] = 32.0;

		heatContent[Dead1h] = _deadHeatContent;
		heatContent[Dead10h] = _deadHeatContent;
		heatContent[Dead100h] = _deadHeatContent;
		heatContent[LiveH] = _liveHeatContent;
		heatContent[LiveW] = _liveHeatContent;

		totalSilica[Dead1h] = 0.0555;
		totalSilica[Dead10h] = 0.0555;
		totalSilica[Dead100h] = 0.0555;
		totalSilica[LiveH] = 0.0555;
		totalSilica[LiveW] = 0.0555;

		effectiveSilica[Dead1h] = 0.0100;
		effectiveSilica[Dead10h] = 0.0100;
		effectiveSilica[Dead100h] = 0.0100;
		effectiveSilica[LiveH] = 0.0100;
		effectiveSilica[LiveW] = 0.0100;

		numFuelParticles = 0;
		for (int i = 0; i < NumClasses; ++i)
		{
			if (load[i] > 0.0f)
				++numFuelParticles;
		}

		for (int i = 0; i < NumClasses; ++i)
		{
			if (density[i] > 0.0f)
				surfaceArea[i] = load[i] * SAV[i] / density[i];
			else
				surfaceArea[i] = 0.0f;

			if (SAV[i] > 0.0f)
				effectiveHeatingNumber[i] = exp(-138.0 / SAV[i]);
			else
				effectiveHeatingNumber[i] = 0.0;
			areaWeightingFactor[i] = 0.0;
		}

		liveArea = surfaceArea[LiveH] + surfaceArea[LiveW];
		deadArea = surfaceArea[Dead1h] + surfaceArea[Dead10h] + surfaceArea[Dead100h];

		if (liveArea > 0.0f)
		{
			areaWeightingFactor[LiveH] = surfaceArea[LiveH] / liveArea;
			areaWeightingFactor[LiveW] = surfaceArea[LiveW] / liveArea;
		}
		if (deadArea > 0.0f)
		{
			areaWeightingFactor[Dead1h] = surfaceArea[Dead1h] / deadArea;
			areaWeightingFactor[Dead10h] = surfaceArea[Dead10h] / deadArea;
			areaWeightingFactor[Dead100h] = surfaceArea[Dead100h] / deadArea;
		}

		float totalArea = liveArea + deadArea;
		liveArea /= totalArea;
		deadArea /= totalArea;

		fuelDensity = 0.0f;
		if (fuelDepth > 0.0f)
		{
			fuelDensity = load[LiveH] + load[LiveW] + load[Dead1h] +
			              load[Dead10h] + load[Dead100h];
			fuelDensity /= fuelDepth;
		}

		float liveSAV = areaWeightingFactor[LiveH] * SAV[LiveH] +
		                areaWeightingFactor[LiveW] * SAV[LiveW];
		float deadSAV = areaWeightingFactor[Dead1h] * SAV[Dead1h] +
		                areaWeightingFactor[Dead10h] * SAV[Dead10h] +
		                areaWeightingFactor[Dead100h] * SAV[Dead100h];
		fuelSAV = liveSAV * liveArea + deadSAV * deadArea;

		packingRatio = 0.0f;
		for (int i = 0; i < NumClasses; ++i)
		{
			if (density[i] > 0.0f)
				packingRatio += load[i] / density[i];
		}
		if (fuelDepth > 0.0f)
			packingRatio /= fuelDepth;

		float liveLoad = areaWeightingFactor[LiveH] * load[LiveH] * 
		                     (1.0 - totalSilica[LiveH]) +
		                 areaWeightingFactor[LiveW] * load[LiveW] * 
						     (1.0 - totalSilica[LiveW]);
		float deadLoad = areaWeightingFactor[Dead1h] * load[Dead1h] * 
		                     (1.0 - totalSilica[Dead1h]) +
		                 areaWeightingFactor[Dead10h] * load[Dead10h] * 
						     (1.0 - totalSilica[Dead10h]) +
						 areaWeightingFactor[Dead100h] * load[Dead100h] * 
						     (1.0 - totalSilica[Dead100h]);

		float liveHeatContent = areaWeightingFactor[LiveH] * heatContent[LiveH] +
		                        areaWeightingFactor[LiveW] * heatContent[LiveW];
		float deadHeatContent = areaWeightingFactor[Dead1h] * heatContent[Dead1h] +
		                        areaWeightingFactor[Dead10h] * heatContent[Dead10h] + 
								areaWeightingFactor[Dead100h] * heatContent[Dead100h];

		float liveEffectiveSilica = areaWeightingFactor[LiveH] * effectiveSilica[LiveH] +
		                            areaWeightingFactor[LiveW] * effectiveSilica[LiveW];
		float deadEffectiveSilica = areaWeightingFactor[Dead1h] * effectiveSilica[Dead1h] +
		                            areaWeightingFactor[Dead10h] * effectiveSilica[Dead10h] +
									areaWeightingFactor[Dead100h] * effectiveSilica[Dead100h];

		float liveSilicaFactor = 1.0f;
		if (liveEffectiveSilica > 0.0f)
			liveSilicaFactor = std::min(1.0, 0.174f / pow(liveEffectiveSilica, 0.19f));
		float deadSilicaFactor = 1.0f;
		if (deadEffectiveSilica > 0.0f)
			deadSilicaFactor = std::min(1.0, 0.174f / pow(deadEffectiveSilica, 0.19f));

		liveReactionFactor = liveLoad * liveHeatContent * liveSilicaFactor;
		deadReactionFactor = deadLoad * deadHeatContent * deadSilicaFactor;

		residenceTime = 384.0 / fuelSAV;
		propagatingFlux = exp((0.792 + 0.681 * sqrt(fuelSAV)) * (packingRatio + 0.1)) /
		                  (192.0 + 0.2595 * fuelSAV);

		//Seriously, I have no idea what this crap is
		float betaOpt = 3.348 / pow(fuelSAV, 0.8189);
		float ratio = packingRatio / betaOpt;
		float aa = 133.0 / pow(fuelSAV, 0.7913);
		float sigma15 = pow(fuelSAV, 1.5);
		float maxGamma = sigma15 / (495.0 + 0.0594 * sigma15);
		float gamma = maxGamma * pow(ratio, aa) * exp(aa * (1.0 - ratio));

		liveReactionFactor *= gamma;
		deadReactionFactor *= gamma;

		slopeK = 5.275 * pow(packingRatio, -0.3);
		windB = 0.02526 * pow(fuelSAV, 0.54);
		float c = 7.47 * exp(-0.133 * pow(fuelSAV, 0.55));
		float e = 0.715 * exp(-0.000359 * fuelSAV);
		windK = c * pow(ratio, -e);
		windE = pow(ratio, e) / c;

		float flive = 0.0f;
		if (SAV[LiveH] > 0.0f)
			flive += load[LiveH] * exp(-500.0 / SAV[LiveH]);
		if (SAV[LiveW] > 0.0f)
			flive += load[LiveW] * exp(-500.0 / SAV[LiveW]);
		liveExtinction = 0.0;
		if (liveLoad > 0.0f)
		{
			fineDeadRatio = load[Dead1h] * effectiveHeatingNumber[Dead1h] +
		    	            load[Dead10h] * effectiveHeatingNumber[Dead10h] +
							load[Dead100h] * effectiveHeatingNumber[Dead100h];
			if (flive > 0.0f)
				liveExtinction = 2.9 * fineDeadRatio / flive;
		}
		else
			fineDeadRatio = 0.0f;

		fuelMoisture[Dead1h] = 0.0;
		fuelMoisture[Dead10h] = 0.0;
		fuelMoisture[Dead100h] = 0.0;
		fuelMoisture[LiveH] = 0.0;
		fuelMoisture[LiveW] = 0.0;
		accelerationConstant = 0.115;
	}

	int modelNumber;
	bool burnable;
	float fuelDepth;
	float extinctionMoisture;
	float rosAdjustment;
	float numFuelParticles;
	float fuelDensity;
	float fuelSAV;
	float packingRatio;
	float load[NumClasses];
	float SAV[NumClasses];
	float density[NumClasses];
	float heatContent[NumClasses];
	float totalSilica[NumClasses];
	float effectiveSilica[NumClasses];
	float surfaceArea[NumClasses];
	float effectiveHeatingNumber[NumClasses];
	float areaWeightingFactor[NumClasses];
	float liveArea;
	float deadArea;
	float fuelMoisture[NumClasses];
	float accelerationConstant;
	
	float liveReactionFactor;
	float deadReactionFactor;
	float residenceTime;
	float propagatingFlux;
	float slopeK;
	float windB;
	float windK;
	float windE;
	float fineDeadRatio;
	float liveExtinction;
};

inline std::vector<FuelModel> readFuelModels(const char* filename)
{
	std::vector<FuelModel> models;
	FILE* fp = fopen(filename, "r");
	if (!fp)
	{
		printf("WARNING: Could not open FMD file %s\n", filename);
		return models;
	}

	float loadMultiplier = 1.0;
	float SAVMultiplier = 1.0;
	float depthMultiplier = 1.0;
	float heatContentMultiplier = 1.0;

	char c;
	while ((c = getc(fp)) != EOF)
	{
		ungetc(c, fp);
		int size = 10;
		char* line = (char*) malloc(size * sizeof(char));
		for (int i = 0; true; ++i)
		{
			if (i == size)
			{
				size *= 2;
				line = (char*)realloc(line, sizeof(char) * size);
			}
			c = getc(fp);
			line[i] = c;
			if (c == '\n' || c == EOF)
			{
				line[i] = 0;
				break;
			}
		}
		if (c == EOF)
			break;
		if (line[0] == '#')
		{
			free(line);
			continue;
		}
		char delims[] = " =,\t";
		char* token = strtok(line, delims);
		if (token == NULL)
		{
			free(line);
			continue;
		}

		if (strcmp(token, "ENGLISH") == 0)
			continue;
		else if (strcmp(token, "METRIC") == 0)
		{
			loadMultiplier = 0.020481614;
			SAVMultiplier = 30.499999954;
			depthMultiplier = 0.032808399;
			heatContentMultiplier = 0.429922614;
			continue;
		}
		int fuelModel = atoi(token);

		float values[12];
		memset(values, 0, sizeof(float) * 12);
		bool completeLine = true;
		for (int i = 0; i < 12; ++i)
		{
			token = strtok(NULL, delims);
			if (token == NULL)
			{
				completeLine = false;
				break;
			}
			values[i] = atof(token);
		}
		if (!completeLine)
		{
			printf("WARNING: Fuel Model %d was improperly formatted...discarding.\n", fuelModel);
			continue;
		}

		if (fuelModel > (int)(models.size()) - 1)
		{
			models.resize(fuelModel + 1, FuelModel());
		}

		const float loadConversion = 0.0459136823f;
		
		models[fuelModel] = FuelModel(fuelModel,
									  values[0] * loadMultiplier * loadConversion,
									  values[1] * loadMultiplier * loadConversion,
									  values[2] * loadMultiplier * loadConversion,
									  values[3] * loadMultiplier * loadConversion,
									  values[4] * loadMultiplier * loadConversion,
									  values[5] * SAVMultiplier,
									  values[6] * SAVMultiplier,
									  values[7] * SAVMultiplier,
									  values[8] * depthMultiplier,
									  values[9],
									  values[10] * heatContentMultiplier,
									  values[11] * heatContentMultiplier);
	}

	fclose(fp);
	return models;
}

}
