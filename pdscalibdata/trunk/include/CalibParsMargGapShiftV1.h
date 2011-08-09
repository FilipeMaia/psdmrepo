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
 *  Gets, holds, and provides an access to the x,y,z of
 *  1) the common margin of 2x1 in quad, 
 *  2) the margin of the quad in CSPad,
 *  3) gap between quads in CSPad, 
 *  4) relative shift of quads in CSPad.
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

class CalibParsMargGapShiftV1  {
public:

  enum { NUMBER_OF_PARAMETERS = 12 };

  CalibParsMargGapShiftV1( const std::vector<double> v_parameters );
  double getQuadMargX () { return m_quad_marg_x; };
  double getQuadMargY () { return m_quad_marg_y; };
  double getQuadMargZ () { return m_quad_marg_z; };

  double getMargX () { return m_marg_x; };
  double getMargY () { return m_marg_y; };
  double getMargZ () { return m_marg_z; };

  double getGapX  () { return m_gap_x; };
  double getGapY  () { return m_gap_y; };
  double getGapZ  () { return m_gap_z; };

  double getShiftX() { return m_shift_x; };
  double getShiftY() { return m_shift_y; };
  double getShiftZ() { return m_shift_z; };
  void  print();

  // Default constructor
  CalibParsMargGapShiftV1 () ;

  // Destructor
  virtual ~CalibParsMargGapShiftV1 () ;

protected:

private:

  // Data members
  // Quad margine, CSPad margine, gap, and shift of/between four quads in the detector
  double m_quad_marg_x;
  double m_quad_marg_y;
  double m_quad_marg_z;

  double m_marg_x;
  double m_marg_y;
  double m_marg_z;

  double m_gap_x;
  double m_gap_y;
  double m_gap_z;

  double m_shift_x;
  double m_shift_y;
  double m_shift_z;  


  // Copy constructor and assignment are disabled by default
  CalibParsMargGapShiftV1 ( const CalibParsMargGapShiftV1& ) ;
  CalibParsMargGapShiftV1& operator = ( const CalibParsMargGapShiftV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CALIBPARSMARGGAPSHIFTV1_H
