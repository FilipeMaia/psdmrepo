#ifndef PDSCALIBDATA_CALIBPARSCENTERV1_H
#define PDSCALIBDATA_CALIBPARSCENTERV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsCenterV1.
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
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CalibParsCenterV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NSect = Psana::CsPad::SectorsPerQuad};
  enum { NUMBER_OF_PARAMETERS = 96 };
  CalibParsCenterV1( const std::vector<float> v_parameters );
  float getCenterX(size_t quad, size_t sect){ return m_center_x[quad][sect]; };
  float getCenterY(size_t quad, size_t sect){ return m_center_y[quad][sect]; };
  float getCenterZ(size_t quad, size_t sect){ return m_center_z[quad][sect]; };
  void  print();

  // Default constructor
  CalibParsCenterV1 () ;

  // Destructor
  virtual ~CalibParsCenterV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) center coordinates from optical measurements
  float m_center_x[NQuad][NSect];
  float m_center_y[NQuad][NSect];
  float m_center_z[NQuad][NSect];  

  // Copy constructor and assignment are disabled by default
  CalibParsCenterV1 ( const CalibParsCenterV1& ) ;
  CalibParsCenterV1& operator = ( const CalibParsCenterV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSCENTERV1_H
