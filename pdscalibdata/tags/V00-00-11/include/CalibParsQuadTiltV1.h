#ifndef PDSCALIBDATA_CALIBPARSQUADTILTV1_H
#define PDSCALIBDATA_CALIBPARSQUADTILTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsQuadTiltV1.
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
 *  tilt angles (degree) of the CSPad
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

class CalibParsQuadTiltV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NUMBER_OF_PARAMETERS = 4 };

  CalibParsQuadTiltV1( const std::vector<double> v_parameters );
  double getQuadTilt(size_t quad){ return m_quad_tilt[quad]; };
  void  print();

  // Default constructor
  CalibParsQuadTiltV1 () ;

  // Destructor
  virtual ~CalibParsQuadTiltV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) tilt angles from optical measurements
  double m_quad_tilt[NQuad];  

  // Copy constructor and assignment are disabled by default
  CalibParsQuadTiltV1 ( const CalibParsQuadTiltV1& ) ;
  CalibParsQuadTiltV1& operator = ( const CalibParsQuadTiltV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSQUADTILTV1_H
