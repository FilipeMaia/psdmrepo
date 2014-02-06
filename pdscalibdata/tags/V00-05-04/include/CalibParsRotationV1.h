#ifndef PDSCALIBDATA_CALIBPARSROTATIONV1_H
#define PDSCALIBDATA_CALIBPARSROTATIONV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsRotationV1.
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
 *  Gets, holds, and provides an access to the 32 (4 quad x 8 2x1-sectors) 
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

class CalibParsRotationV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NSect = Psana::CsPad::SectorsPerQuad};
  enum { NUMBER_OF_PARAMETERS = 32 };

  CalibParsRotationV1( const std::vector<double> v_parameters );
  double getRotation(size_t quad, size_t sect){ return m_rotation[quad][sect]; };
  void  print();

  // Default constructor
  CalibParsRotationV1 () ;

  // Destructor
  virtual ~CalibParsRotationV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) nominal rotation angles (0,90,180,270)
  double m_rotation[NQuad][NSect];  

  // Copy constructor and assignment are disabled by default
  CalibParsRotationV1 ( const CalibParsRotationV1& ) ;
  CalibParsRotationV1& operator = ( const CalibParsRotationV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSROTATIONV1_H
