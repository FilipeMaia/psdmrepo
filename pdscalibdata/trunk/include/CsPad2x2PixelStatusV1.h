#ifndef PDSCALIBDATA_CSPAD2X2PIXELSTATUSV1_H
#define PDSCALIBDATA_CSPAD2X2PIXELSTATUSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2PixelStatusV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>
#include <stdint.h>

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
 *  @brief Pixel status data for CsPad2x2::ElementV1.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class CsPad2x2PixelStatusV1 : boost::noncopyable {
public:

  enum { Sections = 2 };
  enum { Columns = Pds::CsPad2x2::ColumnsPerASIC };
  enum { Rows = Pds::CsPad2x2::MaxRowsPerASIC*2 };
  enum { Size = Sections*Columns*Rows };

  // NOTE: Using the same codes as in CspadCorrector
  enum PixelStatus {
    VeryHot=1,
    Hot=2,
    Cold=4
  };

  typedef uint16_t status_t;

  /// Default constructor, all pixel codes set to 0
  CsPad2x2PixelStatusV1 () ;

  /**
   *  Read all codes from file.
   *
   *  @throw std::exception
   */
  CsPad2x2PixelStatusV1 (const std::string& fname) ;

  // Destructor
  ~CsPad2x2PixelStatusV1 () ;

  // access status data
  ndarray<status_t, 3> status() const {
    return make_ndarray(m_status, Columns, Rows, Sections);
  }

protected:

private:

  // Data members
  mutable status_t m_status[Size];

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPAD2X2PIXELSTATUSV1_H
