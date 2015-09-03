#ifndef PDSCALIBDATA_CALIBPARSOFFSETV1_H
#define PDSCALIBDATA_CALIBPARSOFFSETV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsOffsetV1.
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
 *  Gets, holds, and provides an access to the x,y,z of 4 quads 
 *  offsets (in pixel size) of the CSPad
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

class CalibParsOffsetV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NUMBER_OF_PARAMETERS = 12 };

  CalibParsOffsetV1( const std::vector<double>& v_parameters );
  double getOffsetX(size_t quad){ return m_offset_x[quad]; };
  double getOffsetY(size_t quad){ return m_offset_y[quad]; };
  double getOffsetZ(size_t quad){ return m_offset_z[quad]; };
  void  print();

  // Default constructor
  CalibParsOffsetV1 () ;

  // Destructor
  virtual ~CalibParsOffsetV1 () ;

protected:

private:

  // Data members
  // Offsets of four quads in the detector
  double m_offset_x[NQuad];
  double m_offset_y[NQuad];
  double m_offset_z[NQuad];  

  // Copy constructor and assignment are disabled by default
  CalibParsOffsetV1 ( const CalibParsOffsetV1& ) ;
  CalibParsOffsetV1& operator = ( const CalibParsOffsetV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSOFFSETV1_H
