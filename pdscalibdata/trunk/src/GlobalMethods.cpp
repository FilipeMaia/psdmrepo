//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GlobalMethods...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cmath> // for sqrt, atan2, etc.
#include <time.h>
#include <stdlib.h>     // getenv

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;
using namespace pdscalibdata;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pdscalibdata {

//----------------
// Constructors --
//----------------
GlobalMethods::GlobalMethods ()
{
}

//--------------
// Destructor --
//--------------
GlobalMethods::~GlobalMethods ()
{
}

//--------------------

std::string 
stringFromUint(unsigned number, unsigned width, char fillchar)
{
  stringstream ssNum; ssNum << setw(width) << setfill(fillchar) << number;
  return ssNum.str();
}

//--------------------

void 
printSizeOfTypes()
{
  std::cout << "Size Of Types:" 
            << "\nsizeof(bool    ) = " << sizeof(bool    ) << " with typeid(bool    ).name(): " << typeid(bool    ).name() 
            << "\nsizeof(uint8_t ) = " << sizeof(uint8_t ) << " with typeid(uint8_t ).name(): " << typeid(uint8_t ).name()  
            << "\nsizeof(uint16_t) = " << sizeof(uint16_t) << " with typeid(uint16_t).name(): " << typeid(uint16_t).name()  
            << "\nsizeof(uint32_t) = " << sizeof(uint32_t) << " with typeid(uint32_t).name(): " << typeid(uint32_t).name()  
            << "\nsizeof(int     ) = " << sizeof(int     ) << " with typeid(int     ).name(): " << typeid(int     ).name()  
            << "\nsizeof(int16_t ) = " << sizeof(int16_t ) << " with typeid(int16_t ).name(): " << typeid(int16_t ).name()  
            << "\nsizeof(int32_t ) = " << sizeof(int32_t ) << " with typeid(int32_t ).name(): " << typeid(int32_t ).name()  
            << "\nsizeof(float   ) = " << sizeof(float   ) << " with typeid(float   ).name(): " << typeid(float   ).name()  
            << "\nsizeof(double  ) = " << sizeof(double  ) << " with typeid(double  ).name(): " << typeid(double  ).name()  
            << "\n\n";
}


//--------------------

DATA_TYPE 
enumDataTypeForString(std::string str_type)
{
    if      (str_type == "float"   ) return FLOAT;   
    else if (str_type == "double"  ) return DOUBLE;  
    else if (str_type == "short"   ) return SHORT;   
    else if (str_type == "unsigned") return UNSIGNED;
    else if (str_type == "int"     ) return INT;     
    else if (str_type == "int16_t" ) return INT16;   
    else if (str_type == "int32_t" ) return INT32;   
    else if (str_type == "uint"    ) return UINT;    
    else if (str_type == "uint8_t" ) return UINT8;   
    else if (str_type == "uint16_t") return UINT16;  
    else if (str_type == "uint32_t") return UINT32;  
    else return NONIMPL;
}

//--------------------

std::string strDataTypeForEnum(DATA_TYPE enum_type)
{
    if      (enum_type == FLOAT   ) return std::string("float");  
    else if (enum_type == DOUBLE  ) return std::string("double"); 
    else if (enum_type == SHORT   ) return std::string("short");  
    else if (enum_type == UNSIGNED) return std::string("unsigned");
    else if (enum_type == INT     ) return std::string("int");    
    else if (enum_type == INT16   ) return std::string("int16_t");
    else if (enum_type == INT32   ) return std::string("int32_t");
    else if (enum_type == UINT    ) return std::string("unsigned");   
    else if (enum_type == UINT8   ) return std::string("uint8_t");
    else if (enum_type == UINT16  ) return std::string("uint16_t");
    else if (enum_type == UINT32  ) return std::string("uint32_t");
    else return std::string("non-implemented");
}

//--------------------

float 
findCommonMode(const double* pars,
               const int16_t* sdata,
               const float* peddata, 
               const uint16_t *pixStatus, 
               unsigned ssize,
               int stride
               )
{
  // do we even need it
  //if (m_mode == None) return float(UnknownCM);

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
    if (pixStatus and (pixStatus[p] & 1)) continue;
    
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
  if (mean > pars[0] or sigma > pars[1]) return float(UnknownCM);
  
  return mean;
}

//--------------------

std::string strTimeStamp(const std::string& format)
{
  time_t  time_sec;
  time ( &time_sec );
  struct tm* timeinfo; timeinfo = localtime ( &time_sec );
  char c_time_buf[32]; strftime(c_time_buf, 32, format.c_str(), timeinfo);
  return std::string (c_time_buf);
}

//--------------------

std::string strEnvVar(const std::string& str)
{
  char* var; var = getenv (str.c_str());
  if (var!=NULL) return std::string (var);
  else           return str + " IS NOT DEFINED!";
}


//--------------------
//--------------------
//--------------------
} // namespace pdscalibdata
