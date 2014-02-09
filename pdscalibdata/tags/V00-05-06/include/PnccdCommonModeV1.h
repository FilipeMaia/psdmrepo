#ifndef PDSCALIBDATA_PNCCDCOMMONMODEV1_H
#define PDSCALIBDATA_PNCCDCOMMONMODEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdCommonModeV1.h 1 2014-01-28 18:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdCommonModeV1.
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
 *  @version $Id: PnccdCommonModeV1.cpp 2014-01-28 18:00:00Z dubrovin@SLAC.STANFORD.EDU $
 *
 *  @author Mikhail Dubrovin
 */

class PnccdCommonModeV1: public PnccdBaseV1 {
public:

  const static size_t CMSize = 16; 
 
  typedef double pars_t;

  /// Default constructor
  PnccdCommonModeV1 () ;
  
  /// Read parameters from file
  PnccdCommonModeV1 (const std::string& fname) ;

  /// Destructor
  ~PnccdCommonModeV1 (){}

  /// Access parameters
  ndarray<pars_t, 1> common_mode() const {
    return make_ndarray(m_pars, CMSize);
  }

  /// Print array of parameters
  void  print();

protected:

private:

  /// Data members  
  mutable pars_t m_pars[CMSize];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  PnccdCommonModeV1 ( const PnccdCommonModeV1& ) ;
  PnccdCommonModeV1& operator = ( const PnccdCommonModeV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_PNCCDCOMMONMODEV1_H
