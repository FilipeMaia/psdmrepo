//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/OceanOpticsConfigV1.h"

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
OceanOpticsConfigV1::OceanOpticsConfigV1 ( const Pds::OceanOptics::ConfigV1& data )
  : exposureTime(data.exposureTime())
  , strayLightConstant(data.strayLightConstant())
{
  const ndarray<const double, 1>& waveLenCalib = data.waveLenCalib();
  std::copy(waveLenCalib.begin(), waveLenCalib.end(), this->waveLenCalib);
  const ndarray<const double, 1>& nonlinCorrect = data.nonlinCorrect();
  std::copy(nonlinCorrect.begin(), nonlinCorrect.end(), this->nonlinCorrect);
}

OceanOpticsConfigV1::~OceanOpticsConfigV1()
{
}


hdf5pp::Type
OceanOpticsConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
OceanOpticsConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<OceanOpticsConfigV1>() ;
  confType.insert_native<float>("exposureTime", offsetof(OceanOpticsConfigV1, exposureTime));
  confType.insert_native<double>("waveLenCalib", offsetof(OceanOpticsConfigV1, waveLenCalib), 4);
  confType.insert_native<double>("nonlinCorrect", offsetof(OceanOpticsConfigV1, nonlinCorrect), 8);
  confType.insert_native<double>("strayLightConstant", offsetof(OceanOpticsConfigV1, strayLightConstant));

  return confType ;
}

void
OceanOpticsConfigV1::store( const Pds::OceanOptics::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  OceanOpticsConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}


} // namespace H5DataTypes
