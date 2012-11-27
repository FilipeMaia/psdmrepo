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
#include "hdf5pp/ArrayType.h"
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
  for (int i = 0; i != 4; ++ i) {
    waveLenCalib[i] = data.waveLenCalib(i);
  }
  for (int i = 0; i != 8; ++ i) {
    nonlinCorrect[i] = data.nonlinCorrect(i);
  }
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
  hdf5pp::ArrayType wlCalibType = hdf5pp::ArrayType::arrayType<double>(4) ;
  hdf5pp::ArrayType nlCalibType = hdf5pp::ArrayType::arrayType<double>(8) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<OceanOpticsConfigV1>() ;
  confType.insert_native<float>("exposureTime", offsetof(OceanOpticsConfigV1, exposureTime));
  confType.insert("waveLenCalib", offsetof(OceanOpticsConfigV1, waveLenCalib), wlCalibType);
  confType.insert("nonlinCorrect", offsetof(OceanOpticsConfigV1, nonlinCorrect), nlCalibType);
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
