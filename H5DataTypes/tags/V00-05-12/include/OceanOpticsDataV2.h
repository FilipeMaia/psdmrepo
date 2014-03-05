#ifndef H5DATATYPES_OCEANOPTICSDATAV2_H
#define H5DATATYPES_OCEANOPTICSDATAV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsDataV2.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "H5DataTypes/XtcClockTime.h"
#include "hdf5pp/Type.h"
#include "pdsdata/psddl/oceanoptics.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::OceanOptics::DataV2
//

class OceanOpticsDataV2  {
public:

  typedef Pds::OceanOptics::DataV2 XtcType ;

  OceanOpticsDataV2 () {}
  OceanOpticsDataV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type() ;
  static hdf5pp::Type stored_corrected_data_type() ;

private:

  uint64_t     frameCounter;
  uint64_t     numDelayedFrames;
  uint64_t     numDiscardFrames;
  XtcClockTime timeFrameStart;
  XtcClockTime timeFrameFirstData;
  XtcClockTime timeFrameEnd;
  int8_t       numSpectraInData;
  int8_t       numSpectraInQueue;
  int8_t       numSpectraUnused;
  double       durationOfFrame;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_OCEANOPTICSDATAV2_H
