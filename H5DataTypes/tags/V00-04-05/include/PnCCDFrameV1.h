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
#include "pdsdata/pnCCD/ConfigV2.hh"
#include "pdsdata/pnCCD/FrameV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::PNCCD::FrameV1
//
class PnCCDFrameV1  {
public:

  typedef Pds::PNCCD::FrameV1 XtcType ;
  typedef Pds::PNCCD::ConfigV1 ConfigXtcType1 ;
  typedef Pds::PNCCD::ConfigV2 ConfigXtcType2 ;

  // Default constructor
  PnCCDFrameV1 () {}
  PnCCDFrameV1 ( const XtcType& frame ) ;

  static hdf5pp::Type stored_type( const ConfigXtcType1& config ) ;
  static hdf5pp::Type native_type( const ConfigXtcType1& config ) ;
  static hdf5pp::Type stored_type( const ConfigXtcType2& config ) ;
  static hdf5pp::Type native_type( const ConfigXtcType2& config ) ;

  static hdf5pp::Type stored_data_type( const ConfigXtcType1& config ) ;
  static hdf5pp::Type stored_data_type( const ConfigXtcType2& config ) ;

protected:

  static hdf5pp::Type native_type( unsigned numlinks ) ;

private:

  uint32_t specialWord;
  uint32_t frameNumber;
  uint32_t timeStampHi;
  uint32_t timeStampLo;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_PNCCDFRAMEV1_H
