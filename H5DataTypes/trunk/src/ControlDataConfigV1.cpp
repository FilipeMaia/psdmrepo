//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlDataConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/ControlDataConfigV1.h"

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
    strType.set_size( size ) ;
    return strType ;
  }


  hdf5pp::Type _pvCtrlNameType()
  {
    static hdf5pp::Type strType = _strType( Pds::ControlData::PVControl::NameSize );
    return strType ;
  }

  hdf5pp::Type _pvMonNameType()
  {
    static hdf5pp::Type strType = _strType( Pds::ControlData::PVMonitor::NameSize );
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
ControlDataPVControlV1::ControlDataPVControlV1 ( const Pds::ControlData::PVControl& pconfig )
{
  strcpy( m_data.name, pconfig.name() ) ;
  m_data.index = pconfig.index() ;
  m_data.value = pconfig.value() ;
}

hdf5pp::Type
ControlDataPVControlV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ControlDataPVControlV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< ControlDataPVControlV1_Data >() ;
  type.insert( "name", offsetof(ControlDataPVControlV1_Data,name), _pvCtrlNameType() ) ;
  type.insert_native<int32_t>( "index", offsetof(ControlDataPVControlV1_Data,index) ) ;
  type.insert_native<double>( "value", offsetof(ControlDataPVControlV1_Data,value) ) ;

  return type ;
}

ControlDataPVMonitorV1::ControlDataPVMonitorV1 ( const Pds::ControlData::PVMonitor& pconfig )
{
  strcpy( m_data.name, pconfig.name() ) ;
  m_data.index = pconfig.index() ;
  m_data.loValue = pconfig.loValue() ;
  m_data.hiValue = pconfig.hiValue() ;
}

hdf5pp::Type
ControlDataPVMonitorV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ControlDataPVMonitorV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< ControlDataPVMonitorV1_Data >() ;
  type.insert( "name", offsetof(ControlDataPVMonitorV1_Data,name), _pvMonNameType() ) ;
  type.insert_native<int32_t>( "index", offsetof(ControlDataPVMonitorV1_Data,index) ) ;
  type.insert_native<double>( "loValue", offsetof(ControlDataPVMonitorV1_Data,loValue) ) ;
  type.insert_native<double>( "hiValue", offsetof(ControlDataPVMonitorV1_Data,hiValue) ) ;

  return type ;
}


ControlDataConfigV1::ControlDataConfigV1 ( const XtcType& data )
{
  m_data.uses_duration = data.uses_duration() ;
  m_data.uses_events = data.uses_events() ;
  m_data.duration.seconds = data.duration().seconds() ;
  m_data.duration.nanoseconds = data.duration().nanoseconds() ;
  m_data.events = data.events() ;
  m_data.npvControls = data.npvControls() ;
  m_data.npvMonitors = data.npvMonitors() ;
}

hdf5pp::Type
ControlDataConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ControlDataConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<ControlDataConfigV1_Data>() ;
  confType.insert_native<uint8_t>( "uses_duration", offsetof(ControlDataConfigV1_Data,uses_duration) ) ;
  confType.insert_native<uint8_t>( "uses_events", offsetof(ControlDataConfigV1_Data,uses_events) ) ;
  confType.insert_native<XtcClockTime>( "duration", offsetof(ControlDataConfigV1_Data,duration) ) ;
  confType.insert_native<uint32_t>( "events", offsetof(ControlDataConfigV1_Data,events) ) ;
  confType.insert_native<uint32_t>( "npvControls", offsetof(ControlDataConfigV1_Data,npvControls) ) ;
  confType.insert_native<uint32_t>( "npvMonitors", offsetof(ControlDataConfigV1_Data,npvMonitors) ) ;

  return confType ;
}

void
ControlDataConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  ControlDataConfigV1 data ( config ) ;
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

}


} // namespace H5DataTypes
