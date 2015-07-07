#ifndef PDSCALIBDATA_CALIBPARSCENTERCORRV1_H
#define PDSCALIBDATA_CALIBPARSCENTERCORRV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsCenterCorrV1.
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
 *  Gets, holds, and provides an access to the 2x1 center correction (in pixel size):
 *  (x,y,z) * (4 quads) * (8 2x1-sectors) of the CSpad
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

class CalibParsCenterCorrV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NSect = Psana::CsPad::SectorsPerQuad};
  enum { NUMBER_OF_PARAMETERS = 96 };

  CalibParsCenterCorrV1( const std::vector<double> v_parameters );
  double getCenterCorrX(size_t quad, size_t sect){ return m_center_corr_x[quad][sect]; };
  double getCenterCorrY(size_t quad, size_t sect){ return m_center_corr_y[quad][sect]; };
  double getCenterCorrZ(size_t quad, size_t sect){ return m_center_corr_z[quad][sect]; };
  void  print();

  // Default constructor
  CalibParsCenterCorrV1 () ;

  // Destructor
  virtual ~CalibParsCenterCorrV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) center coordinate corrections from my semi-manual alignment
  double m_center_corr_x[NQuad][NSect];
  double m_center_corr_y[NQuad][NSect];
  double m_center_corr_z[NQuad][NSect];

  // Copy constructor and assignment are disabled by default
  CalibParsCenterCorrV1 ( const CalibParsCenterCorrV1& ) ;
  CalibParsCenterCorrV1& operator = ( const CalibParsCenterCorrV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSCENTERCORRV1_H
