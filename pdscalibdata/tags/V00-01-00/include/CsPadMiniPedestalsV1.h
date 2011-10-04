#ifndef PDSCALIBDATA_CSPADMINIPEDESTALSV1_H
#define PDSCALIBDATA_CSPADMINIPEDESTALSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadMiniPedestalsV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/cspad/Detector.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/// @addtogroup pdscalibdata

/**
 *  @ingroup pdscalibdata
 *
 *  @brief Pedestals data for CsPad::MiniElementV1.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class CsPadMiniPedestalsV1 : boost::noncopyable {
public:

  enum { Sections = 2 };
  enum { Columns = Pds::CsPad::ColumnsPerASIC };
  enum { Rows = Pds::CsPad::MaxRowsPerASIC*2 };
  enum { Size = Sections*Columns*Rows };

  typedef float pedestal_t;
  // organization of data in MiniElement is quite different from
  // regular CsPad elements
  typedef pedestal_t Pedestals[Columns][Rows][Sections];

  // Default constructor
  CsPadMiniPedestalsV1 () ;

  // read pedestals from file
  CsPadMiniPedestalsV1 (const std::string& fname) ;

  // Destructor
  ~CsPadMiniPedestalsV1 () ;

  // access pedestal data
  const Pedestals& pedestals() const { return m_pedestals; }

protected:

private:

  // Data members
  Pedestals m_pedestals;

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADMINIPEDESTALSV1_H
