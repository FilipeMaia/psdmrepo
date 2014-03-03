#ifndef PDSCALIBDATA_CSPADBASEV2_H
#define PDSCALIBDATA_CSPADBASEV2_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id: CsPadBaseV2.h 1 2014-03-01 18:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class CsPadBaseV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
// #include "ndarray/ndarray.h"
// #include "pdsdata/psddl/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *  class CsPadBaseV2 contains common parameters and methods (if any) for CSPAD. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: CsPadBaseV2.cpp 2014-03-01 18:00:00Z dubrovin@SLAC.STANFORD.EDU $
 *
 *  @author Mikhail Dubrovin
 */

class CsPadBaseV2 {
public:

  const static size_t   Ndim = 4; 
  const static size_t   Quads= 4; 
  const static size_t   Segs = 8; 
  const static size_t   Rows = 185; 
  const static size_t   Cols = 388; 
  const static size_t   Size = Quads*Segs*Rows*Cols; 
  
  // Default constructor
  CsPadBaseV2 () {};
  
  // Destructor
  ~CsPadBaseV2 () {};

protected:

private:

  // Copy constructor and assignment are disabled by default
  CsPadBaseV2 ( const CsPadBaseV2& ) ;
  CsPadBaseV2& operator = ( const CsPadBaseV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADBASEV2_H
