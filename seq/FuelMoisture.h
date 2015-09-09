#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string.h>

namespace sim
{

struct FuelMoisture
{
	FuelMoisture(int _fuelModel = 0, 
	             float _dead1h = 0,
				 float _dead10h = 0,
				 float _dead100h = 0,
				 float _liveH = 0,
				 float _liveW = 0)
	{
		fuelModel = _fuelModel;
		dead1h = _dead1h;
		dead10h = _dead10h;
		dead100h = _dead100h;
		liveH = _liveH;
		liveW = _liveW;
	}

	int fuelModel;
	float dead1h;
	float dead10h;
	float dead100h;
	float liveH;
	float liveW;
};

inline std::vector<FuelMoisture> readFuelMoistures(const char* filename)
{
	std::vector<FuelMoisture> moistures;
	FILE* fp = fopen(filename, "r");
	if (!fp)
	{
		printf("WARNING: Could not open FMS file %s\n", filename);
		return moistures;
	}

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

		int fuelModel = atoi(token);

		float values[5];
		memset(values, 0, sizeof(float) * 5);
		bool completeLine = true;
		for (int i = 0; i < 5; ++i)
		{
			token = strtok(NULL, delims);
			if (token == NULL)
			{
				completeLine = false;
				break;
			}
			values[i] = atof(token) / 100.0f;
		}
		if (!completeLine)
		{
			printf("WARNING: Moisture for Model %d was improperly formatted...discarding.\n", 
			       fuelModel);
			continue;
		}

		if (fuelModel > (int)(moistures.size()) - 1)
		{
			moistures.resize(fuelModel + 1, FuelMoisture());
		}
		moistures[fuelModel] = FuelMoisture(fuelModel,
		                                    values[0],
		                                    values[1],
		                                    values[2],
		                                    values[3],
		                                    values[4]);
	}

	fclose(fp);
	return moistures;
}

}
