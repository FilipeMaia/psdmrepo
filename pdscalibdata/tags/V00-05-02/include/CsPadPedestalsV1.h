#ifndef PDSCALIBDATA_CSPADPEDESTALSV1_H
#define PDSCALIBDATA_CSPADPEDESTALSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPedestalsV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/psddl/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPadPedestalsV1  {
public:

  enum { Quads = Pds::CsPad::MaxQuadsPerSensor };
  enum { Sections = Pds::CsPad::ASICsPerQuad/2 };
  enum { Columns = Pds::CsPad::ColumnsPerASIC };
  enum { Rows = Pds::CsPad::MaxRowsPerASIC*2 };
  enum { Size = Quads*Sections*Columns*Rows };
  
  typedef float pedestal_t;
  
  // Default constructor
  CsPadPedestalsV1 () ;
  
  // read pedestals from file
  CsPadPedestalsV1 (const std::string& fname) ;

  // Destructor
  ~CsPadPedestalsV1 () ;

  // access pedestal data
  ndarray<pedestal_t, 4> pedestals() const {
    return make_ndarray(m_pedestals, Quads, Sections, Columns, Rows);
  }

protected:

private:

  // Data members  
  mutable pedestal_t m_pedestals[Size];

  // Copy constructor and assignment are disabled by default
  CsPadPedestalsV1 ( const CsPadPedestalsV1& ) ;
  CsPadPedestalsV1& operator = ( const CsPadPedestalsV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADPEDESTALSV1_H
