#ifndef PDSCALIBDATA_CALIBPARSQUADROTATIONV1_H
#define PDSCALIBDATA_CALIBPARSQUADROTATIONV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsQuadRotationV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *  Gets, holds, and provides an access to the 4 quad 
 *  rotation angles (degree) of the CSPad
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

class CalibParsQuadRotationV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NUMBER_OF_PARAMETERS = 4 };

  CalibParsQuadRotationV1( const std::vector<double> v_parameters );
  double getQuadRotation(size_t quad){ return m_quad_rotation[quad]; };
  void  print();

  // Default constructor
  CalibParsQuadRotationV1 () ;

  // Destructor
  virtual ~CalibParsQuadRotationV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) nominal rotation angles (0,90,180,270)
  double m_quad_rotation[NQuad];
  
  // Copy constructor and assignment are disabled by default
  CalibParsQuadRotationV1 ( const CalibParsQuadRotationV1& ) ;
  CalibParsQuadRotationV1& operator = ( const CalibParsQuadRotationV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSQUADROTATIONV1_H
