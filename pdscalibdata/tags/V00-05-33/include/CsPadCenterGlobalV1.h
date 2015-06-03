#ifndef PDSCALIBDATA_CSPADCENTERGLOBALV1_H
#define PDSCALIBDATA_CSPADCENTERGLOBALV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadCenterGlobalV1.
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
 *  Gets, holds, and provides an access to the 2x1 center (in pixel size):
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

class CsPadCenterGlobalV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NSect = Psana::CsPad::SectorsPerQuad};
  enum { NUMBER_OF_PARAMETERS = 96 };

  CsPadCenterGlobalV1( const std::vector<double> v_parameters );
  double getCenterX(size_t quad, size_t sect){ return m_center_x[quad][sect]; };
  double getCenterY(size_t quad, size_t sect){ return m_center_y[quad][sect]; };
  double getCenterZ(size_t quad, size_t sect){ return m_center_z[quad][sect]; };
  void  print();

  // Default constructor
  CsPadCenterGlobalV1 () ;

  // Destructor
  virtual ~CsPadCenterGlobalV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) center coordinates from optical measurements
  double m_center_x[NQuad][NSect];
  double m_center_y[NQuad][NSect];
  double m_center_z[NQuad][NSect];  

  // Copy constructor and assignment are disabled by default
  CsPadCenterGlobalV1 ( const CsPadCenterGlobalV1& ) ;
  CsPadCenterGlobalV1& operator = ( const CsPadCenterGlobalV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADCENTERGLOBALV1_H
