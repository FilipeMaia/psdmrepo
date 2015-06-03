#ifndef PDSCALIBDATA_PNCCDPIXELSTATUSV1_H
#define PDSCALIBDATA_PNCCDPIXELSTATUSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdPixelStatusV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "pdscalibdata/PnccdBaseV1.h" // Segs, Rows, Cols etc.

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/psddl/pnccd.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class PnccdPixelStatusV1: public PnccdBaseV1 {
public:

  typedef uint16_t pars_t;

  /// Default constructor
  PnccdPixelStatusV1 () ;
  
  /// Read parameters from file
  PnccdPixelStatusV1 (const std::string& fname) ;

  /// Destructor
  ~PnccdPixelStatusV1 (){}

  /// Access parameters
  ndarray<pars_t, 3> pixel_status() const {
    return make_ndarray(m_pars, Segs, Rows, Cols);
  }

  /// Print array of parameters
  void  print();

protected:

private:

  /// Data members  
  mutable pars_t m_pars[Size];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  PnccdPixelStatusV1 ( const PnccdPixelStatusV1& ) ;
  PnccdPixelStatusV1& operator = ( const PnccdPixelStatusV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_PNCCDPIXELSTATUSV1_H
