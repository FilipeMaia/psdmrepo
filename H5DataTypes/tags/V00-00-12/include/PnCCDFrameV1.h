#ifndef H5DATATYPES_PNCCDFRAMEV1_H
#define H5DATATYPES_PNCCDFRAMEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDFrameV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/pnCCD/ConfigV1.hh"
#include "pdsdata/pnCCD/FrameV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

/**
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

struct PnCCDFrameV1_Data {
  uint32_t specialWord;
  uint32_t frameNumber;
  uint32_t timeStampHi;
  uint32_t timeStampLo;
};

class PnCCDFrameV1  {
public:

  typedef Pds::PNCCD::FrameV1 XtcType ;
  typedef Pds::PNCCD::ConfigV1 ConfigXtcType ;

  // Default constructor
  PnCCDFrameV1 () {}
  PnCCDFrameV1 ( const XtcType& frame ) ;

  static hdf5pp::Type stored_type( const ConfigXtcType& config ) ;
  static hdf5pp::Type native_type( const ConfigXtcType& config ) ;

  static hdf5pp::Type stored_data_type( const ConfigXtcType& config ) ;

protected:

private:

  PnCCDFrameV1_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_PNCCDFRAMEV1_H
