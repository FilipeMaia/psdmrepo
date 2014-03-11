//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/OceanOpticsConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
OceanOpticsConfigV2::OceanOpticsConfigV2 ( const Pds::OceanOptics::ConfigV2& data )
  : exposureTime(data.exposureTime())
  , deviceType(data.deviceType())
  , strayLightConstant(data.strayLightConstant())
{
  const ndarray<const double, 1>& waveLenCalib = data.waveLenCalib();
  std::copy(waveLenCalib.begin(), waveLenCalib.end(), this->waveLenCalib);
  const ndarray<const double, 1>& nonlinCorrect = data.nonlinCorrect();
  std::copy(nonlinCorrect.begin(), nonlinCorrect.end(), this->nonlinCorrect);
}

OceanOpticsConfigV2::~OceanOpticsConfigV2()
{
}


hdf5pp::Type
OceanOpticsConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
OceanOpticsConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<OceanOpticsConfigV2>() ;
  confType.insert_native<float>("exposureTime", offsetof(OceanOpticsConfigV2, exposureTime));
  confType.insert_native<int32_t>("deviceType", offsetof(OceanOpticsConfigV2, deviceType));
  confType.insert_native<double>("waveLenCalib", offsetof(OceanOpticsConfigV2, waveLenCalib), 4);
  confType.insert_native<double>("nonlinCorrect", offsetof(OceanOpticsConfigV2, nonlinCorrect), 8);
  confType.insert_native<double>("strayLightConstant", offsetof(OceanOpticsConfigV2, strayLightConstant));

  return confType ;
}

void
OceanOpticsConfigV2::store( const Pds::OceanOptics::ConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  OceanOpticsConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}


} // namespace H5DataTypes
