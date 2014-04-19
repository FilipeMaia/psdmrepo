#ifndef PDSCALIBDATA_CSPAD2X2BASEV2_H
#define PDSCALIBDATA_CSPAD2X2BASEV2_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//	Class CsPad2x2BaseV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
// #include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
// #include "ndarray/ndarray.h"
// #include "pdsdata/psddl/cspad2x2.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *  class CsPad2x2BaseV2 contains common parameters and methods (if any) for CSPAD. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class CsPad2x2BaseV2 {
public:

  const static size_t   Ndim = 3; 
  const static size_t   Segs = 2; 
  const static size_t   Rows = 185; 
  const static size_t   Cols = 388; 
  const static size_t   Size = Rows*Cols*Segs; 
  
  // Default constructor
  CsPad2x2BaseV2 () {};
  
  // Destructor
  ~CsPad2x2BaseV2 () {};

protected:

private:

  // Copy constructor and assignment are disabled by default
  CsPad2x2BaseV2 ( const CsPad2x2BaseV2& ) ;
  CsPad2x2BaseV2& operator = ( const CsPad2x2BaseV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPAD2X2BASEV2_H
