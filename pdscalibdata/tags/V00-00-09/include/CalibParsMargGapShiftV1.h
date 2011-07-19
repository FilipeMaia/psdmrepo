#ifndef PDSCALIBDATA_CALIBPARSMARGGAPSHIFTV1_H
#define PDSCALIBDATA_CALIBPARSMARGGAPSHIFTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsMargGapShiftV1.
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

class CalibParsMargGapShiftV1  {
public:

  enum { NUMBER_OF_PARAMETERS = 12 };
  CalibParsMargGapShiftV1( const std::vector<float> v_parameters );
  float getQuadMargX () { return m_quad_marg_x; };
  float getQuadMargY () { return m_quad_marg_y; };
  float getQuadMargZ () { return m_quad_marg_z; };

  float getMargX () { return m_marg_x; };
  float getMargY () { return m_marg_y; };
  float getMargZ () { return m_marg_z; };

  float getGapX  () { return m_gap_x; };
  float getGapY  () { return m_gap_y; };
  float getGapZ  () { return m_gap_z; };

  float getShiftX() { return m_shift_x; };
  float getShiftY() { return m_shift_y; };
  float getShiftZ() { return m_shift_z; };
  void  print();

  // Default constructor
  CalibParsMargGapShiftV1 () ;

  // Destructor
  virtual ~CalibParsMargGapShiftV1 () ;

protected:

private:

  // Data members
  // Quad margine, CSPad margine, gap, and shift of/between four quads in the detector
  float m_quad_marg_x;
  float m_quad_marg_y;
  float m_quad_marg_z;

  float m_marg_x;
  float m_marg_y;
  float m_marg_z;

  float m_gap_x;
  float m_gap_y;
  float m_gap_z;

  float m_shift_x;
  float m_shift_y;
  float m_shift_z;  


  // Copy constructor and assignment are disabled by default
  CalibParsMargGapShiftV1 ( const CalibParsMargGapShiftV1& ) ;
  CalibParsMargGapShiftV1& operator = ( const CalibParsMargGapShiftV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSMARGGAPSHIFTV1_H
