#ifndef PDSCALIBDATA_PNCCDBASEV1_H
#define PDSCALIBDATA_PNCCDBASEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadBaseV1.
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
 *  class PnccdBaseV1 contains common parameters and methods (if any) for pnCCD. 
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

class PnccdBaseV1 {
public:

  const static size_t   Ndim = 3; 
  const static size_t   Segs = 4; 
  const static size_t   Rows = 512; 
  const static size_t   Cols = 512; 
  const static size_t   Size = Segs*Rows*Cols; 
  
  // Default constructor
  PnccdBaseV1 () {};
  
  // Destructor
  ~PnccdBaseV1 () {};

protected:

private:

  // Copy constructor and assignment are disabled by default
  PnccdBaseV1 ( const PnccdBaseV1& ) ;
  PnccdBaseV1& operator = ( const PnccdBaseV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_PNCCDBASEV1_H
