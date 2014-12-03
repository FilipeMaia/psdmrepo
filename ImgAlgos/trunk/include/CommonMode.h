#ifndef IMGALGOS_COMMONMODE_H
#define IMGALGOS_COMMONMODE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CommonMode.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <fstream>   // ofstream
#include <iomanip>   // for setw, setfill
#include <sstream>   // for stringstream
#include <stdint.h>  // uint8_t, uint32_t, etc.
#include <iostream>
#include <cmath>
//#include <math.h>
//#include <stdio.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "MsgLogger/MsgLogger.h"
#include "ndarray/ndarray.h"

namespace ImgAlgos {

using namespace std;

//--------------------

 const static int UnknownCM = -10000; 

//--------------------
// Code from ami/event/FrameCalib.cc
// int median(ndarray<const uint32_t,2> d, unsigned& iLo, unsigned& iHi);

//--------------------
//--------------------

//--------------------
// Philip Hart's code from psalg/src/common_mode.cc:
//
// averages all values in array "data" for length "length"
// that are below "threshold".  If the magniture of the correction
// is greater than "maxCorrection" leave "data" unchanged, otherwise
// subtract the average in place.  
// mask is either a null pointer (in which case nothing is masked)
// or a list of values arranged like the data where non-zero means ignore

   template <typename T>
   void commonMode(T* data, const uint16_t* mask, const unsigned length, const T threshold, const T maxCorrection, T& cm) {
     // do dumbest thing possible to start - switch to median
     // do 2nd dumbest thing possible to start - switch to median
     cm = 0;
     T* tmp = data;
     double sum = 0;
     int nSummed = 0;
     for (unsigned col=0; col<length; col++) {
       T cval = *tmp++;
       T mval = (mask) ? *mask++ : 0;
       if (mval==0 && cval<threshold) {
         nSummed++;
         sum += cval;
       }
     }
     
     if (nSummed>0) {
       T mean = (T)(sum/nSummed);
       if (fabs(mean)<=maxCorrection) {
         cm = mean;
         tmp = data;
         for (unsigned col=0; col<length; col++) {
   	   *tmp++ -= cm;
         }
       }
     }
   }

//--------------------
//--------------------
// Philip Hart's code from psalg/src/common_mode.cc:
//
// finds median of values in array "data" for length "length"
// that are below "threshold".  If the magniture of the correction
// is greater than "maxCorrection" leave "data" unchanged, otherwise
// subtract the median in place.  
// mask is either a null pointer (in which case nothing is masked)
// or a list of values arranged like the data where non-zero means ignore

/*
template <typename T>
void commonModeMedian(const T* data, const uint16_t* mask, const unsigned length, const T threshold, const T maxCorrection, T& cm) {
  cm = 0;
  const T* tmp = data;

  const uint32_t lMax = 32768;//2**14*2;  // here I may be assuming data in ADU
  const uint32_t lHalfMax = 16384;//2**14;
  unsigned hist[lMax];
  memset(hist, 0, sizeof(unsigned)*lMax);
  int nSummed = 0;
  for (unsigned col=0; col<length; col++) {
    T cval = *tmp++;
    T mval = (mask) ? *mask++ : 0;
    if (mval==0 && cval<threshold) {
      nSummed++;
      unsigned bin = (int)cval+lHalfMax;
      // unsigned long?  check range or raise?
      hist[bin]++;
    }
  }

  if (nSummed==0) return;

  unsigned medianCount = (unsigned)ceil(nSummed/2.);
  unsigned histSum = 0;
  for (unsigned bin=0; bin<lMax; bin++) {
    histSum += hist[bin];
    if (histSum>=medianCount) {
      T median = (int)bin -  (int)lHalfMax;
      if (fabs(median)<=maxCorrection) {
        cm = median;
      }
      return;
    }
  }
}

*/

//--------------------

template <typename T>
void commonModeMedian(T* data, const uint16_t* mask, const unsigned length, const T threshold, const T maxCorrection, T& cm) {
  commonModeMedian((const T*)data, mask, length, threshold, maxCorrection, cm);
  if (cm != 0) {
    T* tmp = data;
    for (unsigned col=0; col<length; col++) {
      *tmp++ -= cm;
    }
  }
}

//--------------------
//--------------------
// This method was originally designed by Andy for CSPAD in pdscalibdata/src/CsPadCommonModeSubV1.cpp
// - data type int16_t is changed to T
// - subtract CM inside this module
// - pedestal is not subtracted in this algorithm; assume that it is already subtracted

  /**
   *  Find common mode for an CsPad  section.
   *  
   *  Function will return UnknownCM value if the calculation 
   *  cannot be performed (or need not be performed).
   *  
   *  @param pars   array[3] of control parameters; mean_max, sigma_max, threshold on number of pixels/ADC count
   *  @param sdata  pixel data
   *  @param pixStatus  pixel status data, can be zero pointer
   *  @param ssize  size of all above arrays
   *  @param stride increment for pixel indices
   */ 

   //float cm_corr =  findCommonMode<T>(pars, sdata, pixStatus, ssize, nSect); 

template <typename T>
float 
findCommonMode(const double* pars,
               T* sdata,
               const uint16_t *pixStatus, 
               unsigned ssize,
               int stride = 1
               )
{
  // do we even need it
  //if (m_mode == None) return float(UnknownCM);

  // for now it does not make sense to calculate common mode
  // if pedestals are not known
  // if (not peddata) return float(UnknownCM);
  
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
    //if (pixStatus and (pixStatus[p] & 1)) continue;
    if (pixStatus and pixStatus[p]) continue; // Discard  pixels with any status > 0
    
    // pixel value with pedestal subtracted, rounded to integer
    double dval = sdata[p]; // - peddata[p];
    int val = dval < 0 ? int(dval - 0.5) : int(dval + 0.5);
    
    // histogram bin
    unsigned bin = unsigned(val - low);
    
    // increment bin value if in range
    if (bin < hsize) {
      ++hist[bin] ;
      ++ count;
    }      
  }

  MsgLog("findCommonMode", debug, "histo filled count = " << count);
  
  // analyze histogram now, first find peak position
  // as the position of the lowest bin with highest count 
  // larger than 100 and which has a bin somewhere on 
  // right side with count dropping by half
  int peakPos = -1;
  int peakCount = -1;
  int hmRight = hsize;
  int thresh = int(pars[2]);
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
    MsgLog("findCommonMode", debug, "peakPos = " << peakPos);
    return float(UnknownCM);
  }

  // find half maximum channel on left side
  int hmLeft = -1;
  for (int i = peakPos; hmLeft < 0 and i >= 0; -- i) {
    if(hist[i] <= peakCount/2) hmLeft = i;
  }
  MsgLog("findCommonMode", debug, "peakPos = " << peakPos << " peakCount = " << peakCount 
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
  
  MsgLog("findCommonMode", debug, "mean = " << mean << " sigma = " << sigma);

  // limit the values to some reasonable numbers
  if (abs(mean) > pars[0] or sigma > pars[1]) return float(UnknownCM);
  
  //--------------------
  // subtract CM 
  for (unsigned c = 0, p = 0; c < ssize; ++ c, p += stride) {
        sdata[p] -= mean;
  } 

  return mean;
}




//--------------------
// Modified Philip Hart's commonMode (mean) code (see above) for 2-d region in data. 

template <typename T>
  void meanInRegion(  const double* pars
                    , ndarray<T,2>& data 
	            , ndarray<const uint16_t,2>& status
	            , const size_t& rowmin
                    , const size_t& colmin
                    , const size_t& nrows
                    , const size_t& ncols
                    , const size_t& srows = 1
                    , const size_t& scols = 1
                    ) {

     T threshold = pars[0];
     T maxcorr   = pars[1];

     double sumv = 0;
     int    sum1 = 0;
     bool   check_status = (status.data()) ? true : false;

     for (size_t r=rowmin; r<rowmin+nrows; r+=srows) { 
       for (size_t c=colmin; c<colmin+ncols; c+=scols) {
         T v = data[r][c];
         T s = (check_status) ? status[r][c] : 0;
         if (s==0 && v<threshold) {
           sumv += v;
           sum1 ++;
         }
       }
     }
     
     if (sum1>0) {
       T mean = (T)(sumv/sum1);

       std::cout << "  mean:" << mean
                 << "  threshold:" << threshold
	         << '\n';

       if (fabs(mean)<=maxcorr) {
         for (size_t r=rowmin; r<rowmin+nrows; r+=srows) { 
           for (size_t c=colmin; c<colmin+ncols; c+=scols) {
   	     data[r][c] -= mean;
           }
         }
       }
     }
}
//--------------------
// Median, similar to ami/event/FrameCalib::median, 
// but finding median for entire good statistics w/o discarding edge bins

  template <typename T>
  void medianInRegion(const double* pars
                    , ndarray<T,2>& data 
	            , ndarray<const uint16_t,2>& status
	            , const size_t& rowmin
                    , const size_t& colmin
                    , const size_t& nrows
                    , const size_t& ncols
                    , const size_t& srows = 1
                    , const size_t& scols = 1
		    , const unsigned& pbits=0
                    ) {

      int hint_range = (int)pars[1];
      int maxcorr    = (int)pars[2];

      bool check_status = (status.data()) ? true : false;

      // declare array for histogram
      int half_range = max(int(hint_range), 10);

      int low  = -half_range;
      int high =  half_range;
      unsigned* hist  = 0;
      int       nbins = 0;
      int       bin   = 0;
      unsigned  iter  = 0;
      unsigned  count = 0;

      while(1) { // loop continues untill the range does not contain a half of statistics

	  iter ++;

          nbins = high-low+1;
	  if (nbins>10000) {
            if (pbits & 4) MsgLog("medianInRegion", warning, "Too many bins " << nbins 
                                  << ", common mode correction is not allied");
	    return;
	  }

          if (hist) delete[] hist;
          hist = new unsigned[nbins];
          std::fill_n(hist, nbins, 0);
          count = 0;
      
          // fill histogram
          for (size_t r=rowmin; r<rowmin+nrows; r+=srows) { 
            for (size_t c=colmin; c<colmin+ncols; c+=scols) {
            
              // ignore pixels that are too noisy; discard pixels with any status > 0
              if (check_status && status[r][c]) continue;
            
              bin = int(data[r][c]) - low;            
              if      (bin < 1)     hist[0]++;
              else if (bin < nbins) hist[bin]++;
              else                  hist[nbins-1]++;
              count++;
            }
          }
          
          if (pbits & 2) {
              MsgLog("medianInRegion", info, "Iter:" << iter 
                                   << "  histo nbins:" << nbins 
                                   << "  low:" << low 
                                   << "  high:" << high 
                                   << "  count:" << count);
              for (int b=0; b<nbins; ++b) std::cout << " " << b << ":" << hist[b]; 
              std::cout  << '\n';
          }
    
          // do not apply correction if the number of good pixels is small
          if (count < 10) return; 

          if (hist[0]>count/2) {
	    if (maxcorr && low < -maxcorr) return; // do not apply cm correction
            low  -= nbins/4;
	  }
          else if (hist[nbins-1]>count/2) {
	    if (maxcorr && high > maxcorr) return; // do not apply cm correction
	    high += nbins/4;
	  }
	  else
            break; 
      } // while(1)

      int i=-1;
      int s = count/2;
      while( s>0 ) s -= hist[++i];

      if (unsigned(abs(-s)) > hist[i-1]/2) i--; // step back

      int icm = low+i+1; // +1 is an empiric shift of distribution to 0.

      if (maxcorr && abs(icm)>maxcorr) return; // do not apply cm correction

      T cm = (T)icm;
      if (pbits & 1) MsgLog("medianInRegion", info, "cm correction = " << cm);

      // Apply common mode correction to data
      for (size_t r=rowmin; r<rowmin+nrows; r+=srows) { 
        for (size_t c=colmin; c<colmin+ncols; c+=scols) {
          data[r][c] -= cm;
        }
      }
  }

//--------------------

} // namespace ImgAlgos

#endif // IMGALGOS_COMMONMODE_H
