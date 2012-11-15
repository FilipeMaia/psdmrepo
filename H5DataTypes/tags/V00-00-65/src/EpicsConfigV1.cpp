//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsConfigV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EpicsConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  hdf5pp::Type _strType( size_t size )
  {
    hdf5pp::Type strType = hdf5pp::Type::Copy(H5T_C_S1);
    strType.set_size(size);
    return strType;
  }


  hdf5pp::Type _pvConfigDescType()
  {
    static hdf5pp::Type strType = _strType(Pds::Epics::PvConfigV1::iMaxPvDescLength);
    return strType;
  }

}

//      ----------------------------------------
//      -- Public Function Member Definitions --
//      ----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
EpicsPvConfigV1::EpicsPvConfigV1(const Pds::Epics::PvConfigV1& pvConfig)
  : pvId(pvConfig.iPvId)
  , interval(pvConfig.fInterval)
{
  std::copy(pvConfig.sPvDesc, pvConfig.sPvDesc+iMaxPvDescLength, description);
}

hdf5pp::Type
EpicsPvConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EpicsPvConfigV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<EpicsPvConfigV1>() ;
  type.insert_native<int16_t>( "pvId", offsetof(EpicsPvConfigV1, pvId) ) ;
  type.insert( "description", offsetof(EpicsPvConfigV1, description), _pvConfigDescType() ) ;
  type.insert_native<float>( "interval", offsetof(EpicsPvConfigV1, interval) ) ;

  return type ;
}


EpicsConfigV1::EpicsConfigV1(const XtcType& data)
  : numPv(data.getNumPv())
{
}

hdf5pp::Type
EpicsConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EpicsConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EpicsConfigV1>() ;
  confType.insert_native<uint32_t>("numPv", offsetof(EpicsConfigV1, numPv)) ;

  return confType ;
}

void
EpicsConfigV1::store(const XtcType& config, hdf5pp::Group grp)
{
  // make scalar data set for main object
  EpicsConfigV1 data(config);
  storeDataObject(data, "config", grp);

  // pvconfig data
  const uint32_t numPv = config.getNumPv();
  EpicsPvConfigV1 pvConfigs[numPv];
  for (uint32_t i = 0; i < numPv; ++ i) {
    pvConfigs[i] = EpicsPvConfigV1(*config.getPvConfig(i));
  }
  storeDataObjects(numPv, pvConfigs, "pvConfig", grp);
}


} // namespace H5DataTypes
