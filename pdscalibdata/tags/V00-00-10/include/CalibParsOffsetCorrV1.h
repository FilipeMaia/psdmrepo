#ifndef PDSCALIBDATA_CALIBPARSOFFSETCORRV1_H
#define PDSCALIBDATA_CALIBPARSOFFSETCORRV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsOffsetCorrV1.
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
 *  offset corrections (in pixel size) of the CSPad
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

class CalibParsOffsetCorrV1  {
public:

  enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
  enum { NUMBER_OF_PARAMETERS = 12 };

  CalibParsOffsetCorrV1( const std::vector<float> v_parameters );
  float getOffsetCorrX(size_t quad){ return m_offset_corr_x[quad]; };
  float getOffsetCorrY(size_t quad){ return m_offset_corr_y[quad]; };
  float getOffsetCorrZ(size_t quad){ return m_offset_corr_z[quad]; };
  void  print();

  // Default constructor
  CalibParsOffsetCorrV1 () ;

  // Destructor
  virtual ~CalibParsOffsetCorrV1 () ;

protected:

private:

  // Data members
  // Offsets of four quad corrections in the detector
  float m_offset_corr_x[NQuad];
  float m_offset_corr_y[NQuad];
  float m_offset_corr_z[NQuad];

  // Copy constructor and assignment are disabled by default
  CalibParsOffsetCorrV1 ( const CalibParsOffsetCorrV1& ) ;
  CalibParsOffsetCorrV1& operator = ( const CalibParsOffsetCorrV1& ) ;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSOFFSETCORRV1_H
