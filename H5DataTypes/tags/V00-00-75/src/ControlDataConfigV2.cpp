//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlDataConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/ControlDataConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
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

//----------------
// Constructors --
//----------------
ControlDataPVLabelV1::ControlDataPVLabelV1 ( const Pds::ControlData::PVLabel& plabel )
{
  strcpy( name, plabel.name() ) ;
  strcpy( value, plabel.value() ) ;
}

hdf5pp::Type
ControlDataPVLabelV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ControlDataPVLabelV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<ControlDataPVLabelV1>() ;
  type.insert( "name", offsetof(ControlDataPVLabelV1, name), _pvLblNameType() ) ;
  type.insert( "value", offsetof(ControlDataPVLabelV1, value), _pvLblValueType() ) ;

  return type ;
}


ControlDataConfigV2::ControlDataConfigV2 ( const XtcType& data )
  : uses_duration(data.uses_duration())
  , uses_events(data.uses_events())
  , duration(data.duration())
  , events(data.events())
  , npvControls(data.npvControls())
  , npvMonitors(data.npvMonitors())
  , npvLabels(data.npvLabels())
{
}

hdf5pp::Type
ControlDataConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ControlDataConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<ControlDataConfigV2>() ;
  confType.insert_native<uint8_t>( "uses_duration", offsetof(ControlDataConfigV2,uses_duration) ) ;
  confType.insert_native<uint8_t>( "uses_events", offsetof(ControlDataConfigV2,uses_events) ) ;
  confType.insert_native<XtcClockTime>( "duration", offsetof(ControlDataConfigV2,duration) ) ;
  confType.insert_native<uint32_t>( "events", offsetof(ControlDataConfigV2,events) ) ;
  confType.insert_native<uint32_t>( "npvControls", offsetof(ControlDataConfigV2,npvControls) ) ;
  confType.insert_native<uint32_t>( "npvMonitors", offsetof(ControlDataConfigV2,npvMonitors) ) ;

  return confType ;
}

void
ControlDataConfigV2::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  ControlDataConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // pvcontrols data
  const uint32_t npvControls = config.npvControls() ;
  ControlDataPVControlV1 pvControls[npvControls] ;
  for ( uint32_t i = 0 ; i < npvControls ; ++ i ) {
    pvControls[i] = ControlDataPVControlV1( config.pvControl(i) ) ;
  }
  storeDataObjects ( npvControls, pvControls, "pvControls", grp ) ;

  // pvmonitors data
  const uint32_t npvMonitors = config.npvMonitors() ;
  ControlDataPVMonitorV1 pvMonitors[npvMonitors] ;
  for ( uint32_t i = 0 ; i < npvMonitors ; ++ i ) {
    pvMonitors[i] = ControlDataPVMonitorV1( config.pvMonitor(i) ) ;
  }
  storeDataObjects ( npvMonitors, pvMonitors, "pvMonitors", grp ) ;

  // pvlabels data
  const uint32_t npvLabels = config.npvLabels() ;
  ControlDataPVLabelV1 pvLabels[npvLabels] ;
  for ( uint32_t i = 0 ; i < npvLabels ; ++ i ) {
    pvLabels[i] = ControlDataPVLabelV1( config.pvLabel(i) ) ;
  }
  storeDataObjects ( npvLabels, pvLabels, "pvLabels", grp ) ;

}


} // namespace H5DataTypes
