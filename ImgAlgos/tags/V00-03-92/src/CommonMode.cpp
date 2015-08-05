//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CommonMode...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CommonMode.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include <boost/shared_ptr.hpp>
//#include "PSEvt/EventId.h"
//#include "PSTime/Time.h"
//#include "psana/Module.h"
//#include "pdsdata/xtc/DetInfo.hh" // for srcToString( const Pds::Src& src )
//#include "PSCalib/Exceptions.h"   // for srcToString( const Pds::Src& src )

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;
using namespace ImgAlgos;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//--------------------
//--------------------

//--------------------
// Matthew Weaver's code from ami/event/FrameCalib.cc

/*
int median(ndarray<const uint32_t,2> d,
           unsigned& iLo, unsigned& iHi)
{
  unsigned* bins = 0;
  unsigned nbins = 0;

  while(1) {
    if (bins) delete[] bins;

    if (iLo>iHi) {
      printf("Warning: FrameCalib::median iLo,iHi arguments reversed [%d,%d]\n",
             iLo,iHi);
      unsigned v=iLo; iLo=iHi; iHi=v;
    }

    nbins = iHi-iLo+1;

    if (nbins>10000) {
      printf("Warning: FrameCalib::median too many bins [%d]\n",nbins);
      return -1;
    }

    bins  = new unsigned[nbins];
    memset(bins,0,nbins*sizeof(unsigned));

    for(unsigned j=0; j<d.shape()[0]; j++) {
      const uint32_t* data = &d[j][0];
      for(unsigned i=0; i<d.shape()[1]; i++)
        if (data[i] < iLo)
          bins[0]++;
        else if (data[i] >= iHi)
          bins[nbins-1]++;
        else
          bins[data[i]-iLo]++;
    }
        
    if (bins[0] > d.size()/2)
      if (iLo > nbins/4) 
        iLo -= nbins/4;
      else if (iLo > 0)
        iLo = 0;
      else {
        delete[] bins;
        return iLo;
      }
    else if (bins[nbins-1] > d.size()/2)
      iHi += nbins/4;
    else
      break;
  }
    
  unsigned i=1;
  int s=(d.size()-bins[0]-bins[nbins-1])/2;
  while( s>0 )
    s -= bins[i++];

  if (unsigned(-s) > bins[i-1]/2) i--;

  delete[] bins;

  return (iLo+i);
}


*/

//--------------------
//--------------------
//--------------------
//--------------------

//template void ImgAlgos::commonMode<double> (double*  data, const uint16_t* mask, const unsigned length, const double  threshold, const double  maxCorrection, double&  cm);
//template void ImgAlgos::commonMode<float>  (float*   data, const uint16_t* mask, const unsigned length, const float   threshold, const float   maxCorrection, float&   cm);
//template void ImgAlgos::commonMode<int32_t>(int32_t* data, const uint16_t* mask, const unsigned length, const int32_t threshold, const int32_t maxCorrection, int32_t& cm);
//template void ImgAlgos::commonMode<int16_t>(int16_t* data, const uint16_t* mask, const unsigned length, const int16_t threshold, const int16_t maxCorrection, int16_t& cm);

//--------------------

} // namespace ImgAlgos

