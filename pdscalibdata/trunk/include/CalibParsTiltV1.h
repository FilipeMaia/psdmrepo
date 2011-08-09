#ifndef PDSCALIBDATA_CALIBPARSTILTV1_H
#define PDSCALIBDATA_CALIBPARSTILTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsTiltV1.
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

class CalibParsTiltV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NSect = Psana::CsPad::SectorsPerQuad};
  enum { NUMBER_OF_PARAMETERS = 32 };

  CalibParsTiltV1( const std::vector<double> v_parameters );
  double getTilt(size_t quad, size_t sect){ return m_tilt[quad][sect]; };
  void  print();

  // Default constructor
  CalibParsTiltV1 () ;

  // Destructor
  virtual ~CalibParsTiltV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) tilt angles from optical measurements
  double m_tilt[NQuad][NSect];  

  // Copy constructor and assignment are disabled by default
  CalibParsTiltV1 ( const CalibParsTiltV1& ) ;
  CalibParsTiltV1& operator = ( const CalibParsTiltV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSTILTV1_H
