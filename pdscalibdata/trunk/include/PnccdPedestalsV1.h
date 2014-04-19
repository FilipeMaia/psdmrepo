#ifndef PDSCALIBDATA_PNCCDPEDESTALSV1_H
#define PDSCALIBDATA_PNCCDPEDESTALSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdPedestalsV1.
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

class PnccdPedestalsV1: public PnccdBaseV1 {
public:

  typedef float pars_t;

  /// Default constructor
  PnccdPedestalsV1 () ;
  
  /// Read pedestals from file
  PnccdPedestalsV1 (const std::string& fname) ;

  /// Destructor
  ~PnccdPedestalsV1 (){}

  /// Access pedestal data
  ndarray<pars_t, 3> pedestals() const {
    return make_ndarray(m_pars, Segs, Rows, Cols);
  }

  /// Print part of pedestal array
  void  print();

protected:

private:

  /// Data members  
  mutable pars_t m_pars[Size];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  PnccdPedestalsV1 ( const PnccdPedestalsV1& ) ;
  PnccdPedestalsV1& operator = ( const PnccdPedestalsV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_PNCCDPEDESTALSV1_H
