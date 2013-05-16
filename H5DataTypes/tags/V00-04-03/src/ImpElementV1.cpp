//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImpElementV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/ImpElementV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

ImpSample::ImpSample ( const XtcType& data )
{
  for (unsigned i = 0; i != 4; ++ i) {
    channels[i] = const_cast<XtcType&>(data).channel(i);
  }
}

hdf5pp::Type
ImpSample::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ImpSample::native_type()
{
  typedef ImpSample DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hsize_t _array_type_channels_shape[] = { 4 };
  hdf5pp::ArrayType _array_type_channels = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<uint32_t>::native_type(), 1, _array_type_channels_shape);
  type.insert("channels", offsetof(DsType, channels), _array_type_channels);
  return type;
}




ImpLaneStatus::ImpLaneStatus ( const XtcType& data )
  : linkErrCount(data.usLinkErrCount)
  , linkDownCount(data.usLinkDownCount)
  , cellErrCount(data.usCellErrCount)
  , rxCount(data.usRxCount)
  , locLinked(data.usLocLinked)
  , remLinked(data.usRemLinked)
  , zeros(data.zeros)
  , powersOkay(data.powersOkay)
{
}

hdf5pp::Type
ImpLaneStatus::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ImpLaneStatus::native_type()
{
  typedef ImpLaneStatus DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("linkErrCount", offsetof(DsType, linkErrCount), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("linkDownCount", offsetof(DsType, linkDownCount), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("cellErrCount", offsetof(DsType, cellErrCount), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("rxCount", offsetof(DsType, rxCount), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("locLinked", offsetof(DsType, locLinked), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("remLinked", offsetof(DsType, remLinked), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("zeros", offsetof(DsType, zeros), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("powersOkay", offsetof(DsType, powersOkay), hdf5pp::TypeTraits<uint8_t>::native_type());
  return type;
}




ImpElementV1::ImpElementV1 ( const XtcType& data )
  : vc(const_cast<XtcType&>(data).vc()) 
  , lane(const_cast<XtcType&>(data).lane()) 
  , frameNumber(const_cast<XtcType&>(data).frameNumber())
  , ticks(const_cast<XtcType&>(data).ticks())
  , fiducials(const_cast<XtcType&>(data).fiducials())
  , range(const_cast<XtcType&>(data).range())
  , laneStatus(const_cast<XtcType&>(data).laneStatus())
{
}

hdf5pp::Type
ImpElementV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ImpElementV1::native_type()
{
  typedef ImpElementV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("vc", offsetof(DsType, vc), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("lane", offsetof(DsType, lane), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("frameNumber", offsetof(DsType, frameNumber), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("ticks", offsetof(DsType, ticks), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("fiducials", offsetof(DsType, fiducials), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("range", offsetof(DsType, range), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("laneStatus", offsetof(DsType, laneStatus), hdf5pp::TypeTraits<ImpLaneStatus>::native_type());
  return type;
}

hdf5pp::Type
ImpElementV1::stored_data_type(uint32_t nSamples)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<ImpSample>::native_type() ;

  hsize_t dims[] = { nSamples } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 1, dims );
}

} // namespace H5DataTypes
