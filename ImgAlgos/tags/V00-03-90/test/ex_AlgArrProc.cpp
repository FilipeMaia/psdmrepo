//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class ImgAlgos/AlgArrProc
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "ImgAlgos/AlgArrProc.h"
#include "ImgAlgos/GlobalMethods.h"

#include <string>
#include <iostream>
#include <cstdlib>  // std::rand()
#include <ctime>    // std::time()
#include <stdint.h> // for uint8_t, uint16_t etc.


using std::cout;
using std::endl;


//-----------------
// fill ndarray with constant
template <typename T, unsigned NDIM>
void fillConstNDArray(ndarray<T, NDIM>& nda, const T& c)
{
  for(typename ndarray<T,NDIM>::iterator it=nda.begin(); it!=nda.end(); ++it) *it = c;
}

//-----------------
// fill ndarray with random values in the range [0,100]
template <typename T, unsigned NDIM>
void fillRandonNDArray(ndarray<T, NDIM>& nda)
{
  std::srand(std::time(0)); // use current time as seed for random generator
  for(typename ndarray<T,NDIM>::iterator it=nda.begin(); it!=nda.end(); ++it) *it = std::rand() % 100;
}

//-----------------

int test01 ()
{
  typedef uint16_t mask_t;
  typedef float    data_t;
  //typedef uint32_t wind_t;
  typedef ImgAlgos::AlgArrProc::wind_t wind_t;

  cout << "Test of ImgAlgos::AlgArrProc\n";     

  /*
  unsigned shape_arr_wins[2] = {3, 5};
  size_t arr_wins[] = {0, 12, 112, 22, 222,
                       1, 14, 114, 24, 224,
                       3, 16, 116, 26, 226};
  */
  unsigned shape_arr_wins[2] = {3, 5};
  wind_t arr_wins[] = {0, 0, 185, 0, 388,
                       1, 0, 185, 0, 388,
                       3, 0, 185, 0, 388};
  ndarray<const wind_t,2> nda_winds(&arr_wins[0], shape_arr_wins);

  //ndarray<const size_t,2> nda_winds; // empty ndarray of windows

  unsigned shape3d[3] = {4, 185, 388};

  ndarray<data_t,3> data_nda(shape3d);  fillRandonNDArray<data_t,3>(data_nda);

  ndarray<mask_t,3> mask_nda(shape3d);  fillConstNDArray<mask_t,3>(mask_nda, 1);

  std::cout << "random data ndarray:\n" << data_nda << '\n';
  std::cout << "mask ndarray:\n" << mask_nda << '\n';

  unsigned pbits    = 0; // 0177777;
 
  //ImgAlgos::AlgArrProc* arrp = new ImgAlgos::AlgArrProc(&nda_winds, pbits);
  ImgAlgos::AlgArrProc* arrp = new ImgAlgos::AlgArrProc(nda_winds, pbits);
  arrp->printInputPars();

  data_t threshold = 50;
  unsigned npix = arrp->   numberOfPixAboveThr<data_t,3>(data_nda, mask_nda, threshold);
  double   asum = arrp->intensityOfPixAboveThr<data_t,3>(data_nda, mask_nda, threshold);

  std::cout << "Threshold: " << threshold 
            << "  npix above threshold: " << npix 
            << "  fraction:" << float(npix)/data_nda.size()
            << "  intensity:" << asum << '\n';


  ImgAlgos::printSizeOfTypes();
  std::cout << "Size Of Type:" 
            << "\nsizeof(char) = " << sizeof(char) << " with typeid(char).name(): " << typeid(char).name() << '\n'; 

  std::cout << "Size Of Type:" 
            << "\nsizeof(size_t) = " << sizeof(size_t) << " with typeid(size_t).name(): " << typeid(size_t).name() << '\n'; 

  return 0;
}

//-----------------

int main (int argc, char* argv[])
{  
  cout << "Number of input arguments = " << argc << endl; 
  // atoi(argv[1])==1) 
  
  test01();

  return 0;
}

//-----------------
