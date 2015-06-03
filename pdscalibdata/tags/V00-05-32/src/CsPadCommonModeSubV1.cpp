//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadCommonModeSubV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPadCommonModeSubV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "CsPadCommonModeSubV1";

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pdscalibdata {

//----------------
// Constructors --
//----------------
CsPadCommonModeSubV1::CsPadCommonModeSubV1 ()
  : m_mode(uint32_t(None))
{
  std::fill_n(m_data, int(DataSize), 0.0);
}

CsPadCommonModeSubV1::CsPadCommonModeSubV1 (const std::string& fname) 
  : m_mode(uint32_t(None))
{
  std::fill_n(m_data, int(DataSize), 0.0);
  // cpo: to make analysis easier for users, put in sensible cspad defaults
  m_mode   =1;   // Algorithm mode / number
  m_data[0]=50;  // Maximal allowed correction of the mean value to apply correction 
  m_data[1]=10;  // Maximal allowed value of the peak sigma to apply correction 
  m_data[2]=100; // Threshold on number of pixels per ADU bin to be used in peak finding algorithm
  
  // open file
  std::ifstream in(fname.c_str());
  if (in.good()) {
    // found file. over-ride defaults
    // read first number into a mode
    if (not (in >> m_mode)) {
      const std::string msg = "Common mode file does not have enough data: "+fname;
      MsgLogRoot(error, msg);
      throw std::runtime_error(msg);
    }

    // read whatever left into the array
    // TODO: some error checking, what if non-number appears in a file
    double* it = m_data;
    size_t count = 0;
    while(in and count != DataSize) {
      in >> *it++;
      ++ count;
    }
  }

  MsgLog(logger, trace, "CsPadCommonModeSubV1: mode=" << m_mode << " data=" << m_data[0] << "," << m_data[1] << "," << m_data[2]);
}

CsPadCommonModeSubV1::CsPadCommonModeSubV1 (CommonMode mode, const double data[DataSize])
  : m_mode(uint32_t(mode))
{
  std::copy(data, data+DataSize, m_data);
  MsgLog(logger, trace, "CsPadCommonModeSubV1: mode=" << m_mode << " data=" << m_data[0] << "," << m_data[1] << "," << m_data[2]);
}


//--------------
// Destructor --
//--------------
CsPadCommonModeSubV1::~CsPadCommonModeSubV1 ()
{
}

float 
CsPadCommonModeSubV1::findCommonMode(const int16_t* sdata,
                                     const float* peddata, 
                                     const  uint16_t *pixStatus, 
                                     unsigned ssize,
                                     int stride) const
{
  // do we even need it
  if (m_mode == None) return float(UnknownCM);

  // for now it does not make sense to calculate common mode
  // if pedestals are not known
  if (not peddata) return float(UnknownCM);
  
  // declare array for histogram
  const int low = -1000;
  const int high = 2000;
  const unsigned hsize = high-low;
  int hist[hsize];
  std::fill_n(hist, hsize, 0);
  unsigned long count = 0;
  
  // fill histogram
  for (unsigned c = 0, p = 0; c != ssize; ++ c, p += stride) {
    
    // ignore channels that re too noisy
    //if (pixStatus and (pixStatus[p] & CsPadPixelStatusV1::VeryHot)) continue;
    if (pixStatus and pixStatus[p]) continue; // all bad pixels (>0) are discarded
    
    // pixel value with pedestal subtracted, rounded to integer
    double dval = sdata[p] - peddata[p];
    int val = dval < 0 ? int(dval - 0.5) : int(dval + 0.5);
    
    // histogram bin
    unsigned bin = unsigned(val - low);
    
    // increment bin value if in range
    if (bin < hsize) {
      ++hist[bin] ;
      ++ count;
    }
      
  }

  MsgLog(logger, debug, "histo filled count = " << count);
  
  // analyze histogram now, first find peak position
  // as the position of the lowest bin with highest count 
  // larger than 100 and which has a bin somewhere on 
  // right side with count dropping by half
  int peakPos = -1;
  int peakCount = -1;
  int hmRight = hsize;
  int thresh = int(m_data[2]);
  if(thresh<=0) thresh=100;
  for (unsigned i = 0; i < hsize; ++ i ) {
    if (hist[i] > peakCount and hist[i] > thresh) {
      peakPos = i;
      peakCount = hist[i];
    } else if (peakCount > 0 and hist[i] <= peakCount/2) {
      hmRight = i;
      break;
    }
  }

  // did we find anything resembling
  if (peakPos < 0) {
    MsgLog(logger, debug, "peakPos = " << peakPos);
    return float(UnknownCM);
  }

  // find half maximum channel on left side
  int hmLeft = -1;
  for (int i = peakPos; hmLeft < 0 and i >= 0; -- i) {
    if(hist[i] <= peakCount/2) hmLeft = i;
  }
  MsgLog(logger, debug, "peakPos = " << peakPos << " peakCount = " << peakCount 
      << " hmLeft = " << hmLeft << " hmRight = " << hmRight);
  
  // full width at half maximum
  int fwhm = hmRight - hmLeft;
  double sigma = fwhm / 2.36;

  // calculate mean and sigma
  double mean = peakPos;
  for (int j = 0; j < 2; ++j) {
    int s0 = 0;
    double s1 = 0;
    double s2 = 0;
    int d = int(sigma*2+0.5);
    for (int i = std::max(0,peakPos-d); i < int(hsize) and i <= peakPos+d; ++ i) {
      s0 += hist[i];
      s1 += (i-mean)*hist[i];
      s2 += (i-mean)*(i-mean)*hist[i];
    }
    mean = mean + s1/s0;
    sigma = std::sqrt(s2/s0 - (s1/s0)*(s1/s0));
  }
  mean += low;
  
  MsgLog(logger, debug, "mean = " << mean << " sigma = " << sigma);

  // limit the values to some reasonable numbers
  if (mean > m_data[0] or sigma > m_data[1]) return float(UnknownCM);
  
  return mean;
}

} // namespace pdscalibdata
