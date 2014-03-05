//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PimaxConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PimaxConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

PimaxConfigV1::PimaxConfigV1 ( const Pds::Pimax::ConfigV1& data )
  : width(data.width())
  , height(data.height())
  , orgX(data.orgX())
  , orgY(data.orgY())
  , binX(data.binX())
  , binY(data.binY())
  , exposureTime(data.exposureTime())
  , coolingTemp(data.coolingTemp())
  , readoutSpeed(data.readoutSpeed())
  , gainIndex(data.gainIndex())
  , intesifierGain(data.intesifierGain())
  , gateDelay(data.gateDelay())
  , gateWidth(data.gateWidth())
  , maskedHeight(data.maskedHeight())
  , kineticHeight(data.kineticHeight())
  , vsSpeed(data.vsSpeed())
  , infoReportInterval(data.infoReportInterval())
  , exposureEventCode(data.exposureEventCode())
  , numIntegrationShots(data.numIntegrationShots())
  , frameSize(data.frameSize())
  , numPixelsX(data.numPixelsX())
  , numPixelsY(data.numPixelsY())
  , numPixels(data.numPixels())
{
}

hdf5pp::Type
PimaxConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PimaxConfigV1::native_type()
{
  typedef PimaxConfigV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("width", offsetof(DsType, width), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("height", offsetof(DsType, height), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("orgX", offsetof(DsType, orgX), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("orgY", offsetof(DsType, orgY), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("binX", offsetof(DsType, binX), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("binY", offsetof(DsType, binY), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("exposureTime", offsetof(DsType, exposureTime), hdf5pp::TypeTraits<float>::native_type());
  type.insert("coolingTemp", offsetof(DsType, coolingTemp), hdf5pp::TypeTraits<float>::native_type());
  type.insert("readoutSpeed", offsetof(DsType, readoutSpeed), hdf5pp::TypeTraits<float>::native_type());
  type.insert("gainIndex", offsetof(DsType, gainIndex), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("intesifierGain", offsetof(DsType, intesifierGain), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("gateDelay", offsetof(DsType, gateDelay), hdf5pp::TypeTraits<double>::native_type());
  type.insert("gateWidth", offsetof(DsType, gateWidth), hdf5pp::TypeTraits<double>::native_type());
  type.insert("maskedHeight", offsetof(DsType, maskedHeight), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("kineticHeight", offsetof(DsType, kineticHeight), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("vsSpeed", offsetof(DsType, vsSpeed), hdf5pp::TypeTraits<float>::native_type());
  type.insert("infoReportInterval", offsetof(DsType, infoReportInterval), hdf5pp::TypeTraits<int16_t>::native_type());
  type.insert("exposureEventCode", offsetof(DsType, exposureEventCode), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("numIntegrationShots", offsetof(DsType, numIntegrationShots), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("frameSize", offsetof(DsType, frameSize), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numPixelsX", offsetof(DsType, numPixelsX), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numPixelsY", offsetof(DsType, numPixelsY), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numPixels", offsetof(DsType, numPixels), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

void
PimaxConfigV1::store( const Pds::Pimax::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PimaxConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
