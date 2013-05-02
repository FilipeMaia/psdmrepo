#ifndef PDSCALIBDATA_CSPADBEAMINTERSECTV1_H
#define PDSCALIBDATA_CSPADBEAMINTERSECTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadBeamIntersectV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <cstddef>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
// #include "psddl_psana/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *  Gets, holds, and provides an access to the 3-vector for CSPAD users.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CsPadBeamIntersectV1  {
public:

  enum { NUMBER_OF_PARAMETERS = 3 };

  CsPadBeamIntersectV1( const std::vector<double> v_parameters );
  double  getVectorEl(std::size_t i){ return m_beam_intersect[i]; };
  double* getVector(){ return &m_beam_intersect[0]; };

  void  print();

  // Default constructor
  CsPadBeamIntersectV1 () ;

  // Destructor
  virtual ~CsPadBeamIntersectV1 () ;

protected:

private:

  // Data members
  // 3-vector, defined by users for CSPAD position
  double m_beam_intersect[NUMBER_OF_PARAMETERS];
  
  // Copy constructor and assignment are disabled by default
  CsPadBeamIntersectV1 ( const CsPadBeamIntersectV1& ) ;
  CsPadBeamIntersectV1& operator = ( const CsPadBeamIntersectV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADBEAMINTERSECTV1_H
