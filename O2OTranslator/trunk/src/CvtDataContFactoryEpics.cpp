//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtDataContFactoryEpics...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CvtDataContFactoryEpics.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  hdf5pp::Type _timeType()
  {
    // embedded EPICS timestamp type
    static hdf5pp::CompoundType timeType = hdf5pp::CompoundType::compoundType<Pds::Epics::epicsTimeStamp>() ;
    static bool init = false ;
    if ( not init ) {
      timeType.insert_native<uint32_t>( "secPastEpoch", offsetof(Pds::Epics::epicsTimeStamp,secPastEpoch) ) ;
      timeType.insert_native<uint32_t>( "nsec", offsetof(Pds::Epics::epicsTimeStamp,nsec) ) ;
      init = true ;
    }
    return timeType ;
  }

  hdf5pp::Type _strType( size_t size )
  {
    hdf5pp::Type strType = hdf5pp::Type::Copy(H5T_C_S1);
    strType.set_size( size ) ;
    return strType ;
  }


  hdf5pp::Type _pvnameType()
  {
    static hdf5pp::Type pvnameType = _strType( sizeof( ((Pds::EpicsPvCtrlHeader*)0)->sPvName ) );
    return pvnameType ;
  }

  hdf5pp::Type _unitsType()
  {
    static hdf5pp::Type unitsType = _strType( MAX_UNITS_SIZE ) ;
    return unitsType ;
  }

  hdf5pp::Type _enumStrType()
  {
    static hdf5pp::Type enumStrType = _strType( MAX_ENUM_STRING_SIZE ) ;
    return enumStrType ;
  }

  hdf5pp::Type _strValueType()
  {
    static hdf5pp::Type valueStrType = _strType( MAX_STRING_SIZE ) ;
    return valueStrType ;
  }


  template <int iDbrType>
  size_t structSize ( const Pds::EpicsPvHeader& pv )
  {
    size_t n = pv.iNumElements ;
    if ( dbr_type_is_CTRL(pv.iDbrType) ) {
      typedef Pds::EpicsPvCtrl<iDbrType> Struct ;
      return sizeof(Struct) + (n-1)*sizeof(typename Struct::TDbrOrg) ;
    } else if ( dbr_type_is_TIME(pv.iDbrType) ) {
      typedef Pds::EpicsPvTime<iDbrType> Struct ;
      return sizeof(Struct) + (n-1)*sizeof(typename Struct::TDbrOrg) ;
    } else {
      // suppress warning
      return 0 ;
    }
  }


  template <typename Struct, typename Field>
  void
  defineCtrlFields( hdf5pp::CompoundType& type )
  {
    type.insert( "pvname", offsetof(Struct,sPvName), _pvnameType() ) ;

    type.insert_native<int16_t>( "status", offsetof(Struct,status) ) ;
    type.insert_native<int16_t>( "severity", offsetof(Struct,severity) ) ;

    type.insert( "units", offsetof(Struct,units), _unitsType() ) ;

    type.insert_native<Field>( "lower_disp_limit", offsetof(Struct,lower_disp_limit) ) ;
    type.insert_native<Field>( "upper_disp_limit", offsetof(Struct,upper_disp_limit) ) ;
    type.insert_native<Field>( "lower_alarm_limit", offsetof(Struct,lower_alarm_limit) ) ;
    type.insert_native<Field>( "upper_alarm_limit", offsetof(Struct,upper_alarm_limit) ) ;
    type.insert_native<Field>( "lower_warning_limit", offsetof(Struct,lower_warning_limit) ) ;
    type.insert_native<Field>( "upper_warning_limit", offsetof(Struct,upper_warning_limit) ) ;
    type.insert_native<Field>( "lower_ctrl_limit", offsetof(Struct,lower_ctrl_limit) ) ;
    type.insert_native<Field>( "upper_ctrl_limit", offsetof(Struct,upper_ctrl_limit) ) ;

  }

  template <typename Struct, typename Field>
  void
  defineStringCtrlFields( hdf5pp::CompoundType& type )
  {
    type.insert( "pvname", offsetof(Struct,sPvName), _pvnameType() ) ;

    type.insert_native<int16_t>( "status", offsetof(Struct,status) ) ;
    type.insert_native<int16_t>( "severity", offsetof(Struct,severity) ) ;

  }

  template <typename Struct, typename Field>
  void
  defineEnumCtrlFields( hdf5pp::CompoundType& type, size_t nStates )
  {
    type.insert( "pvname", offsetof(Struct,sPvName), _pvnameType() ) ;

    type.insert_native<int16_t>( "status", offsetof(Struct,status) ) ;
    type.insert_native<int16_t>( "severity", offsetof(Struct,severity) ) ;

    type.insert_native<int16_t>( "no_str", offsetof(Struct,no_str) ) ;
    hdf5pp::Type arrayType = hdf5pp::ArrayType::arrayType( _enumStrType(), nStates ) ;
    type.insert( "strs", offsetof(Struct,strs), arrayType ) ;
  }

  template <typename Struct>
  void
  defineTimeFields( hdf5pp::CompoundType& type )
  {
    // embedded EPICS timestamp type
    hdf5pp::CompoundType timeType = hdf5pp::CompoundType::compoundType<Pds::Epics::epicsTimeStamp>() ;
    timeType.insert_native<uint32_t>( "secPastEpoch", offsetof(Pds::Epics::epicsTimeStamp,secPastEpoch) ) ;
    timeType.insert_native<uint32_t>( "nsec", offsetof(Pds::Epics::epicsTimeStamp,nsec) ) ;

    // this info must be the same for all time types
    type.insert_native<int16_t>( "status", offsetof(Struct,status) ) ;
    type.insert_native<int16_t>( "severity", offsetof(Struct,severity) ) ;
    type.insert( "stamp", offsetof(Struct,stamp), timeType ) ;
  }

  template <typename Struct, typename Field>
  void
  defineValueField( hdf5pp::CompoundType& type, size_t size )
  {
    if ( size > 1 ) {
      hdf5pp::Type arrayType = hdf5pp::ArrayType::arrayType<Field>(size) ;
      type.insert( "value", offsetof(Struct,value), arrayType ) ;
    } else {
      type.insert_native<Field>( "value", offsetof(Struct,value) ) ;
    }
  }

  template <typename Struct>
  void
  defineStringValueField( hdf5pp::CompoundType& type, size_t size )
  {
    if ( size > 1 ) {
      hdf5pp::Type arrayType = hdf5pp::ArrayType::arrayType(_strValueType(),size) ;
      type.insert( "value", offsetof(Struct,value), arrayType ) ;
    } else {
      type.insert( "value", offsetof(Struct,value), _strValueType() ) ;
    }
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// main method
CvtDataContFactoryEpics::container_type*
CvtDataContFactoryEpics::container( hdf5pp::Group group, const Pds::EpicsPvHeader& pv ) const
{
  hdf5pp::Type type = native_type(pv) ;

  hsize_t chunk = m_chunkSize / type.size() ;
  MsgLog( "CvtDataContFactoryEpics", debug, "create container " << m_name << " with chunk size " << chunk ) ;
  return new container_type ( m_name, group, type, chunk, m_deflate, m_shuffle ) ;
}


// get the type for given PV
hdf5pp::Type
CvtDataContFactoryEpics::hdf_type( const Pds::EpicsPvHeader& pv, bool native )
{
  // rest of it is determined dynamically
  size_t size = 0 ;
  switch ( pv.iDbrType ) {
  case DBR_TIME_STRING:
  case DBR_CTRL_STRING:
    size = ::structSize<DBR_STRING>(pv) ;
    break ;
  case DBR_TIME_SHORT:
  case DBR_CTRL_SHORT:
    size = ::structSize<DBR_SHORT>(pv) ;
    break ;
  case DBR_TIME_FLOAT:
  case DBR_CTRL_FLOAT:
    size = ::structSize<DBR_FLOAT>(pv) ;
    break ;
  case DBR_TIME_ENUM:
  case DBR_CTRL_ENUM:
    size = ::structSize<DBR_ENUM>(pv) ;
    break ;
  case DBR_TIME_CHAR:
  case DBR_CTRL_CHAR:
    size = ::structSize<DBR_CHAR>(pv) ;
    break ;
  case DBR_TIME_LONG:
  case DBR_CTRL_LONG:
    size = ::structSize<DBR_LONG>(pv) ;
    break ;
  case DBR_TIME_DOUBLE:
  case DBR_CTRL_DOUBLE:
    size = ::structSize<DBR_DOUBLE>(pv) ;
    break ;

  default:
    MsgLog( "CvtDataContFactoryEpics", error, "CvtDataContFactoryEpics: unexpected PV type = " << pv.iDbrType ) ;
    break ;
  }

  // header is common to all EPICS types
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType(size) ;
  type.insert_native<int16_t>( "pvId", offsetof(Pds::EpicsPvHeader,iPvId) ) ;
  type.insert_native<int16_t>( "dbrType", offsetof(Pds::EpicsPvHeader,iDbrType) ) ;
  type.insert_native<int16_t>( "numElements", offsetof(Pds::EpicsPvHeader,iNumElements) ) ;


  // rest of it is determined dynamically
  switch ( pv.iDbrType ) {
  case DBR_TIME_STRING:
    ::defineTimeFields<Pds::EpicsPvTime<DBR_STRING> >( type ) ;
    ::defineStringValueField<Pds::EpicsPvTime<DBR_STRING> >( type, pv.iNumElements ) ;
    break ;
  case DBR_TIME_SHORT:
    ::defineTimeFields<Pds::EpicsPvTime<DBR_SHORT> >( type ) ;
    ::defineValueField<Pds::EpicsPvTime<DBR_SHORT>,int16_t>( type, pv.iNumElements ) ;
    break ;
  case DBR_TIME_FLOAT:
    ::defineTimeFields<Pds::EpicsPvTime<DBR_FLOAT> >( type ) ;
    ::defineValueField<Pds::EpicsPvTime<DBR_FLOAT>,float>( type, pv.iNumElements ) ;
    break ;
  case DBR_TIME_ENUM:
    ::defineTimeFields<Pds::EpicsPvTime<DBR_ENUM> >( type ) ;
    ::defineValueField<Pds::EpicsPvTime<DBR_ENUM>,uint16_t>( type, pv.iNumElements ) ;
    break ;
  case DBR_TIME_CHAR:
    ::defineTimeFields<Pds::EpicsPvTime<DBR_CHAR> >( type ) ;
    ::defineValueField<Pds::EpicsPvTime<DBR_CHAR>,uint8_t>( type, pv.iNumElements ) ;
    break ;
  case DBR_TIME_LONG:
    ::defineTimeFields<Pds::EpicsPvTime<DBR_LONG> >( type ) ;
    ::defineValueField<Pds::EpicsPvTime<DBR_LONG>,int32_t>( type, pv.iNumElements ) ;
    break ;
  case DBR_TIME_DOUBLE:
    ::defineTimeFields<Pds::EpicsPvTime<DBR_DOUBLE> >( type ) ;
    ::defineValueField<Pds::EpicsPvTime<DBR_DOUBLE>,double>( type, pv.iNumElements ) ;
    break ;

  case DBR_CTRL_STRING:
    ::defineStringCtrlFields<Pds::EpicsPvCtrl<DBR_STRING>,int16_t>( type ) ;
    ::defineStringValueField<Pds::EpicsPvCtrl<DBR_STRING> >( type, pv.iNumElements ) ;
    break ;
  case DBR_CTRL_SHORT:
    ::defineCtrlFields<Pds::EpicsPvCtrl<DBR_SHORT>,int16_t>( type ) ;
    ::defineValueField<Pds::EpicsPvCtrl<DBR_SHORT>,int16_t>( type, pv.iNumElements ) ;
    break ;
  case DBR_CTRL_FLOAT:
    type.insert_native<int16_t>( "precision", offsetof(Pds::EpicsPvCtrl<DBR_FLOAT>,precision) ) ;
    ::defineCtrlFields<Pds::EpicsPvCtrl<DBR_FLOAT>,float>( type ) ;
    ::defineValueField<Pds::EpicsPvCtrl<DBR_FLOAT>,float>( type, pv.iNumElements ) ;
    break ;
  case DBR_CTRL_ENUM:
    ::defineEnumCtrlFields<Pds::EpicsPvCtrl<DBR_ENUM>,int16_t>( type, static_cast<const Pds::EpicsPvCtrl<DBR_ENUM>&>(pv).no_str ) ;
    ::defineValueField<Pds::EpicsPvCtrl<DBR_ENUM>,uint16_t>( type, pv.iNumElements ) ;
    break ;
  case DBR_CTRL_CHAR:
    ::defineCtrlFields<Pds::EpicsPvCtrl<DBR_CHAR>,uint8_t>( type ) ;
    ::defineValueField<Pds::EpicsPvCtrl<DBR_CHAR>,uint8_t>( type, pv.iNumElements ) ;
    break ;
  case DBR_CTRL_LONG:
    ::defineCtrlFields<Pds::EpicsPvCtrl<DBR_LONG>,int32_t>( type ) ;
    ::defineValueField<Pds::EpicsPvCtrl<DBR_LONG>,int32_t>( type, pv.iNumElements ) ;
    break ;
  case DBR_CTRL_DOUBLE:
    type.insert_native<int16_t>( "precision", offsetof(Pds::EpicsPvCtrl<DBR_DOUBLE>,precision) ) ;
    ::defineCtrlFields<Pds::EpicsPvCtrl<DBR_DOUBLE>,double>( type ) ;
    ::defineValueField<Pds::EpicsPvCtrl<DBR_DOUBLE>,double>( type, pv.iNumElements ) ;
    break ;

  default:
    MsgLog( "CvtDataContFactoryEpics", error, "CvtDataContFactoryEpics: unexpected PV type = " << pv.iDbrType ) ;
    break ;
  }

  return type ;
}

} // namespace O2OTranslator
