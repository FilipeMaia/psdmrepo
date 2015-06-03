//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2PixelGainV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPad2x2PixelGainV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <stdexcept>
#include <fstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pdscalibdata {

//----------------
// Constructors --
//----------------
CsPad2x2PixelGainV1::CsPad2x2PixelGainV1 ()
{
  // fill all pedestals with zeros
  std::fill_n(m_pixelGains, int(Size), pixelGain_t(0));
}

CsPad2x2PixelGainV1::CsPad2x2PixelGainV1 (const std::string& fname)
{
  // open file
  std::ifstream in(fname.c_str());
  if (not in.good()) {
    const std::string msg = "Failed to open pixel gain file: "+fname;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  // read all numbers
  pixelGain_t* it = m_pixelGains;
  size_t count = 0;
  while(in and count != Size) {
    in >> *it++;
    ++ count;
  }

  // check that we read whole array
  if (count != Size) {
    const std::string msg = "Pixel gain file does not have enough data: "+fname;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  // and no data left after we finished reading
  float tmp ;
  if ( in >> tmp ) {
    ++ count;
    const std::string msg = "Pixel gain file has extra data: "+fname;
    MsgLogRoot(error, msg);
    MsgLogRoot(error, "read " << count << " numbers, expecting " << Size );
    throw std::runtime_error(msg);
  }
}

//--------------
// Destructor --
//--------------
CsPad2x2PixelGainV1::~CsPad2x2PixelGainV1 ()
{
}

} // namespace pdscalibdata
