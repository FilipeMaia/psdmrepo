//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlDataConfigV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/ControlDataConfigV3.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/ControlDataConfigV2.h"
#include "H5DataTypes/ControlDataConfigV1.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  hdf5pp::Type _strType( size_t size )
  {
    hdf5pp::Type strType = hdf5pp::Type::Copy(H5T_C_S1);
    strType.set_size( size ) ;
    return strType ;
  }


  hdf5pp::Type _pvLblNameType()
  {
    static hdf5pp::Type strType = _strType( Pds::ControlData::PVLabel::NameSize );
    return strType ;
  }

  hdf5pp::Type _pvLblValueType()
  {
    static hdf5pp::Type strType = _strType( Pds::ControlData::PVLabel::ValueSize );
    return strType ;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

ControlDataConfigV3::ControlDataConfigV3 ( const XtcType& data )
  : uses_l3t_events(data.uses_l3t_events())
  , uses_duration(data.uses_duration())
  , uses_events(data.uses_events())
  , duration(data.duration())
  , events(data.events())
  , npvControls(data.npvControls())
  , npvMonitors(data.npvMonitors())
  , npvLabels(data.npvLabels())
{
}

hdf5pp::Type
ControlDataConfigV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ControlDataConfigV3::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<ControlDataConfigV3>() ;
  confType.insert_native<uint8_t>( "uses_l3t_events", offsetof(ControlDataConfigV3,uses_l3t_events) ) ;
  confType.insert_native<uint8_t>( "uses_duration", offsetof(ControlDataConfigV3,uses_duration) ) ;
  confType.insert_native<uint8_t>( "uses_events", offsetof(ControlDataConfigV3,uses_events) ) ;
  confType.insert_native<XtcClockTime>( "duration", offsetof(ControlDataConfigV3,duration) ) ;
  confType.insert_native<uint32_t>( "events", offsetof(ControlDataConfigV3,events) ) ;
  confType.insert_native<uint32_t>( "npvControls", offsetof(ControlDataConfigV3,npvControls) ) ;
  confType.insert_native<uint32_t>( "npvMonitors", offsetof(ControlDataConfigV3,npvMonitors) ) ;
  confType.insert_native<uint32_t>( "npvLabels", offsetof(ControlDataConfigV3,npvLabels) ) ;

  return confType ;
}

void
ControlDataConfigV3::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  ControlDataConfigV3 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // pvcontrols data
  const ndarray<const Pds::ControlData::PVControl, 1>& in_pvControls = config.pvControls();
  const uint32_t npvControls = config.npvControls() ;
  ControlDataPVControlV1 pvControls[npvControls] ;
  for ( uint32_t i = 0 ; i < npvControls ; ++ i ) {
    pvControls[i] = ControlDataPVControlV1( in_pvControls[i] ) ;
  }
  storeDataObjects ( npvControls, pvControls, "pvControls", grp ) ;

  // pvmonitors data
  const ndarray<const Pds::ControlData::PVMonitor, 1>& in_pvMonitors = config.pvMonitors();
  const uint32_t npvMonitors = config.npvMonitors() ;
  ControlDataPVMonitorV1 pvMonitors[npvMonitors] ;
  for ( uint32_t i = 0 ; i < npvMonitors ; ++ i ) {
    pvMonitors[i] = ControlDataPVMonitorV1( in_pvMonitors[i] ) ;
  }
  storeDataObjects ( npvMonitors, pvMonitors, "pvMonitors", grp ) ;

  // pvlabels data
  const ndarray<const Pds::ControlData::PVLabel, 1>& in_pvLabels = config.pvLabels();
  const uint32_t npvLabels = config.npvLabels() ;
  ControlDataPVLabelV1 pvLabels[npvLabels] ;
  for ( uint32_t i = 0 ; i < npvLabels ; ++ i ) {
    pvLabels[i] = ControlDataPVLabelV1( in_pvLabels[i] ) ;
  }
  storeDataObjects ( npvLabels, pvLabels, "pvLabels", grp ) ;

}


} // namespace H5DataTypes
