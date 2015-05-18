//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadFilterV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPadFilterV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <fstream>
#include <stdexcept>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  const char logger[] = "CsPadFilterV1";
  
  // Predicate for counting 
  struct HigherThan {
    HigherThan(int min) : m_min(min) {}
    
    bool operator()(int val) const { return val > m_min; }
    
    int m_min;
  };
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pdscalibdata {

//----------------
// Constructors --
//----------------
CsPadFilterV1::CsPadFilterV1 ()
  : m_mode(uint32_t(None))
{
  std::fill_n(m_data, int(DataSize), 0.0);
}

CsPadFilterV1::CsPadFilterV1 (const std::string& fname) 
  : m_mode(uint32_t(None))
{
  std::fill_n(m_data, int(DataSize), 0.0);
  
  // open file
  std::ifstream in(fname.c_str());
  if (not in.good()) {
    const std::string msg = "Failed to open cspad filter file: "+fname;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  // read first number into a mode
  if (not (in >> m_mode)) {
    const std::string msg = "cspad filter file does not have enough data: "+fname;
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

  MsgLog(logger, trace, "CsPadFilterV1: mode=" << m_mode << " data=" << m_data[0] << "," << m_data[1]);
  
}

CsPadFilterV1::CsPadFilterV1 (FilterMode mode, const double data[DataSize]) 
  : m_mode(uint32_t(mode))
{
  std::copy(data, data+DataSize, m_data);
  MsgLog(logger, trace, "CsPadFilterV1: mode=" << m_mode << " data=" << m_data[0] << "," << m_data[1]);
}

//--------------
// Destructor --
//--------------
CsPadFilterV1::~CsPadFilterV1 ()
{
}

bool
CsPadFilterV1::filter(const ndarray<const int16_t, 3>& pixelData) const
{
  if (m_mode == None) return true;
  
  unsigned count = std::count_if(pixelData.begin(), pixelData.end(), ::HigherThan(int(m_data[0])));
  
  MsgLog(logger, debug, "CsPadFilterV1::filter - " << count << " pixels above " << m_data[0]);
  
  if (m_data[1] < 0) {
    // m_data[0] is a percentage
    return count > -m_data[1]/100*pixelData.size();
  } else {
    // m_data[0] is absolute pixel count
    return count > m_data[1];
  }

}

bool
CsPadFilterV1::filter(const ndarray<const int16_t, 3>& pixelData, const ndarray<const uint16_t, 3>& pixelStatus) const
{
  if (m_mode == None) return true;

  if (pixelData.size() != pixelStatus.size()) {
    MsgLog(logger, debug, "CsPadFilterV1::filter - pixel data array and pixel status array have different sizes");
    return false;
  }

  // calculate number of pixels exceeding m_data[0] threshold,
  // only take pixels with good status (=0)
  unsigned count = 0;
  int threshold = int(m_data[0]);
  typedef ndarray<const int16_t, 3>::iterator Iter;
  ndarray<const uint16_t, 3>::iterator stat_iter = pixelStatus.begin();
  for (Iter data_iter = pixelData.begin(); data_iter != pixelData.end(); ++ data_iter, ++ stat_iter) {
    if (*stat_iter == 0 and *data_iter > threshold) ++ count;
  }

  MsgLog(logger, debug, "CsPadFilterV1::filter - " << count << " pixels above " << m_data[0]);

  if (m_data[1] < 0) {
    // m_data[0] is a percentage
    return count > -m_data[1]/100*pixelData.size();
  } else {
    // m_data[0] is absolute pixel count
    return count > m_data[1];
  }
}

} // namespace pdscalibdata
