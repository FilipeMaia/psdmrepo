//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2PixelStatusV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPad2x2PixelStatusV1.h"

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
CsPad2x2PixelStatusV1::CsPad2x2PixelStatusV1 ()
{
  // fill all status codes with zeros
  std::fill_n(m_status, int(Size), status_t(0));
}

CsPad2x2PixelStatusV1::CsPad2x2PixelStatusV1 (const std::string& fname)
{
  // open file
  std::ifstream in(fname.c_str());
  if (not in.good()) {
    const std::string msg = "Failed to open pixel status file: "+fname;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  // read all numbers
  status_t* it = m_status;
  size_t count = 0;
  while(in and count != Size) {
    in >> *it++;
    ++ count;
  }

  // check that we read whole array
  if (count != Size) {
    const std::string msg = "Pixel status file does not have enough data: "+fname;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  // and no data left after we finished reading
  float tmp ;
  if ( in >> tmp ) {
    ++ count;
    const std::string msg = "Pixel status file has extra data: "+fname;
    MsgLogRoot(error, msg);
    MsgLogRoot(error, "read " << count << " numbers, expecting " << Size );
    throw std::runtime_error(msg);
  }
}

//--------------
// Destructor --
//--------------
CsPad2x2PixelStatusV1::~CsPad2x2PixelStatusV1 ()
{
}

} // namespace pdscalibdata
