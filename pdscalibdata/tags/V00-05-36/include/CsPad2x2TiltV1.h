#ifndef PDSCALIBDATA_CSPAD2X2TILTV1_H
#define PDSCALIBDATA_CSPAD2X2TILTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2TiltV1.
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
 *  Gets, holds, and provides an access to the 2 of 2x1-sensors 
 *  tilt angles (degree) of the CSPad2x2
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

class CsPad2x2TiltV1  {
public:

  enum { NSect = 2 };
  enum { NUMBER_OF_PARAMETERS = 2 };

  CsPad2x2TiltV1( const std::vector<double> v_parameters );
  double getTilt(size_t sect){ return m_tilt[sect]; };
  void  print();

  // Default constructor
  CsPad2x2TiltV1 () ;

  // Destructor
  virtual ~CsPad2x2TiltV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) tilt angles from optical measurements
  double m_tilt[NSect];  

  // Copy constructor and assignment are disabled by default
  CsPad2x2TiltV1 ( const CsPad2x2TiltV1& ) ;
  CsPad2x2TiltV1& operator = ( const CsPad2x2TiltV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPAD2X2TILTV1_H
