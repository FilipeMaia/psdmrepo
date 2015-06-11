#ifndef PDSCALIBDATA_CSPAD2X2CENTERV1_H
#define PDSCALIBDATA_CSPAD2X2CENTERV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2CenterV1.
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
 *  (x,y,z) * (2 2x1-sensors) of the CSPad2x2
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

class CsPad2x2CenterV1  {
public:

  enum { NSect = 2 };
  enum { NUMBER_OF_PARAMETERS = 6 };

  CsPad2x2CenterV1( const std::vector<double> v_parameters );
  double getCenterX(size_t sect){ return m_center_x[sect]; };
  double getCenterY(size_t sect){ return m_center_y[sect]; };
  double getCenterZ(size_t sect){ return m_center_z[sect]; };
  void  print();

  // Default constructor
  CsPad2x2CenterV1 () ;

  // Destructor
  virtual ~CsPad2x2CenterV1 () ;

protected:

private:

  // Data members
  // Segment (2x1) center coordinates from optical measurements
  double m_center_x[NSect];
  double m_center_y[NSect];
  double m_center_z[NSect];  

  // Copy constructor and assignment are disabled by default
  CsPad2x2CenterV1 ( const CsPad2x2CenterV1& ) ;
  CsPad2x2CenterV1& operator = ( const CsPad2x2CenterV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPAD2X2CENTERV1_H
