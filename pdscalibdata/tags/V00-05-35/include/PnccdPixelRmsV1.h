#ifndef PDSCALIBDATA_PNCCDPIXELRMSV1_H
#define PDSCALIBDATA_PNCCDPIXELRMSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdPixelRmsV1.
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

class PnccdPixelRmsV1: public PnccdBaseV1 {
public:

  typedef float pars_t;

  /// Default constructor
  PnccdPixelRmsV1 () ;
  
  /// Read parameters from file
  PnccdPixelRmsV1 (const std::string& fname) ;

  /// Destructor
  ~PnccdPixelRmsV1 (){}

  /// Access parameters
  ndarray<pars_t, 3> pixel_rms() const {
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
  PnccdPixelRmsV1 ( const PnccdPixelRmsV1& ) ;
  PnccdPixelRmsV1& operator = ( const PnccdPixelRmsV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_PNCCDPIXELRMSV1_H
