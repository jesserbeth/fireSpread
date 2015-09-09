# all: fireSim

# fireSim: simulation.cpp fireSim.h fireSim.cpp FuelModel.h FuelMoisture.h
# 	g++ -g - Wall -o fireSim simulation.cpp fireSim.h fireSim.cpp FuelModel.h FuelMoisture.h

# clean:
# 	rm -f *.o
# 	rm -f *~

FILES= simulation.cpp fireSim.cpp
CXXFLAGS= -g -Wall -o 
SOURCE= fireSim
INCLUDES= -I /usr/include/gdal/
LIBS=  -L/usr/lib -lgdal
CC=g++

all: fireSim

fireSim: simulation.cpp fireSim.h fireSim.cpp FuelModel.h FuelMoisture.h
	# g++ -g -Wall -o fireSim simulation.cpp fireSim.h fireSim.cpp FuelModel.h FuelMoisture.h
	$(CC) $(CXXFLAGS) $(SOURCE) $(FILES) $(LIBS) $(INCLUDES) 

clean:
	rm -f *.o
	rm -f *~


# # OSX
# CC=g++
# LIBS=  -L/usr/lib -lgdal

# # Compiler flags
# # CXXFLAGS= -g -Wall -std=c++11 -stdlib=libc++
# CXXFLAGS= -g -Wall -std=c++11  -o 
# INCLUDES= -I /usr/include/gdal/
# SOURCE= simulation.cpp
# FILES= simulation.cpp fireSim.h fireSim.cpp FuelModel.h FuelMoisture.h
# MAIN: simulation.cpp fireSim.h fireSim.cpp FuelModel.h FuelMoisture.h
# 	$(CC) $(CXXFLAGS) $(SOURCE) -o $(FILES) $(LIBS) $(INCLUDES) 

