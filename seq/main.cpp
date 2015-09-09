#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

#include <iostream>
#include <string>
using namespace std;


int interpHeight = 64;
int interpWidth = 64;


float* GISToFloatArray(char* fname);
int main(int argc, char** argv)
{
   GDALAllRegister();
   float* data = NULL;
   if(argc > 1)
   {
      data = GISToFloatArray(argv[1]);
      if(data != NULL)
      {
       cout << "First Value: " << data[0] << endl;
       // for(int i = 0; i < interpHeight*interpWidth; i++){
       // 	  if(i % 50 == 0)
       // 	  	cout << endl;
       // 	  cout << data[i] << " ";
       // }
      }
      // cout << endl;
   }  
	return 0;
}

float* GISToFloatArray(char* fname)
{
  // Important note ------ Gdal considers images to be north up
  // the origin of datasets takes place in the upper-left or North-West corner.
  // Now to create a GDAL dataset
  auto ds = ((GDALDataset*) GDALOpen(fname,GA_ReadOnly));
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
  float *pafScanline;
  std::cout << "Min: " << adfMinMax[0] << " Max: " << adfMinMax[1] << endl;
  int dsize = 256;
  pafScanline = (float *) CPLMalloc(sizeof(float)*width*height);

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
  float xres = geot[1];
  float yres = geot[5];
  string proj;
  proj = string(ds->GetProjectionRef());

  // You can get the projection string
  // The projection gives information about the coordinate system a dataset is in
  // This is important to allow other GIS softwares to place datasets into the same
  // coordinate space
  char* test = &proj[0];

  // The origin of the dataset 
  float startx = geot[0]; // east - west coord.
  float starty = geot[3]; // north - south coord.

  
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
