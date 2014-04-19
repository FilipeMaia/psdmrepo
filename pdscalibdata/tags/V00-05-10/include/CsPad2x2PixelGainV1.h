#ifndef PDSCALIBDATA_CSPAD2X2PIXELGAINV1_H
#define PDSCALIBDATA_CSPAD2X2PIXELGAINV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2PixelGainV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/psddl/cspad2x2.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/// @addtogroup pdscalibdata

/**
 *  @ingroup pdscalibdata
 *
 *  @brief Pedestals data for CsPad2x2::ElementV1.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class CsPad2x2PixelGainV1 : boost::noncopyable {
public:

  enum { Sections = 2 };
  enum { Columns = Pds::CsPad2x2::ColumnsPerASIC };
  enum { Rows = Pds::CsPad2x2::MaxRowsPerASIC*2 };
  enum { Size = Sections*Columns*Rows };

  typedef float pixelGain_t;

  // Default constructor
  CsPad2x2PixelGainV1 () ;

  // read pedestals from file
  CsPad2x2PixelGainV1 (const std::string& fname) ;

  // Destructor
  ~CsPad2x2PixelGainV1 () ;

  // access pedestal data
  ndarray<pixelGain_t, 3> pixelGains() const {
    return make_ndarray(m_pixelGains, Columns, Rows, Sections);
  }

protected:

private:

  // Data members
  mutable pixelGain_t m_pixelGains[Size];

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPAD2X2PIXELGAINV1_H
