#ifndef PDSCALIBDATA_CSPADBEAMVECTORV1_H
#define PDSCALIBDATA_CSPADBEAMVECTORV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadBeamVectorV1.
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

class CsPadBeamVectorV1  {
public:

  enum { NUMBER_OF_PARAMETERS = 3 };

  CsPadBeamVectorV1( const std::vector<double> v_parameters );
  double  getVectorEl(std::size_t i){ return m_beam_vector[i]; };
  double* getVector(){ return &m_beam_vector[0]; };

  void  print();

  // Default constructor
  CsPadBeamVectorV1 () ;

  // Destructor
  virtual ~CsPadBeamVectorV1 () ;

protected:

private:

  // Data members
  // 3-vector, defined by users for CSPAD position
  double m_beam_vector[NUMBER_OF_PARAMETERS];
  
  // Copy constructor and assignment are disabled by default
  CsPadBeamVectorV1 ( const CsPadBeamVectorV1& ) ;
  CsPadBeamVectorV1& operator = ( const CsPadBeamVectorV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADBEAMVECTORV1_H
