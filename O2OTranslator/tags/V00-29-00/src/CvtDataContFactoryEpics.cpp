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

using namespace Pds::Epics;

namespace {

  hdf5pp::Type _timeType()
  {
    // embedded EPICS timestamp type
    static hdf5pp::CompoundType timeType = hdf5pp::CompoundType::compoundType<epicsTimeStamp>() ;
    static bool init = false ;
    if ( not init ) {
      timeType.insert_native<uint32_t>( "secPastEpoch", 0 ) ;
      timeType.insert_native<uint32_t>( "nsec", 4 ) ;
      init = true ;
    }
    return timeType ;
  }


  // define few types for strings in DBRs
  hdf5pp::Type _strType( size_t size )
  {
    hdf5pp::Type strType = hdf5pp::Type::Copy(H5T_C_S1);
    strType.set_size( size ) ;
    return strType ;
  }


  hdf5pp::Type _pvnameType()
  {
    static hdf5pp::Type pvnameType = _strType( sizeof( iMaxPvNameLength ) );
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





  template <typename ValueType>
  unsigned
  defineValueField(hdf5pp::CompoundType& type, size_t size, unsigned offset)
  {
    if ( size > 1 ) {
      hdf5pp::Type arrayType = hdf5pp::ArrayType::arrayType<ValueType>(size) ;
      type.insert( "value", offset, arrayType ) ;
      offset += sizeof(ValueType)*size;
    } else {
      type.insert_native<ValueType>( "value", offset ) ;
      offset += sizeof(ValueType);
    }
    return offset;
  }

  unsigned
  defineStringValueField(hdf5pp::CompoundType& type, size_t size, unsigned offset)
  {
    if ( size > 1 ) {
      hdf5pp::Type arrayType = hdf5pp::ArrayType::arrayType(_strValueType(), size) ;
      type.insert( "value", offset, arrayType ) ;
      offset += MAX_STRING_SIZE*size;
    } else {
      type.insert( "value", offset, _strValueType() ) ;
      offset += MAX_STRING_SIZE;
    }
    return offset;
  }

  // returns offset past the end of defined structure
  static unsigned defineFieldsTimeDbr(hdf5pp::CompoundType& type, unsigned offset)
  {
    // this info must be the same for all time types
    type.insert_native<int16_t>( "status", offset ) ;
    offset += sizeof(int16_t);
    type.insert_native<int16_t>( "severity", offset ) ;
    offset += sizeof(int16_t);
    type.insert( "stamp", offset, _timeType() ) ;
    offset += sizeof(epicsTimeStamp);
    return offset;
  }


  // type traits stuff
  template <int iDbrType> struct DbrId2Type {};

  template <> struct DbrId2Type<DBR_TIME_STRING> {
    typedef dbr_time_string DbrType;
    typedef EpicsPvTimeString EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      // determine DBR offset from the beginning of the pv object
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;
      offset = defineFieldsTimeDbr(type, offset);
      offset = defineStringValueField(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_TIME_SHORT> {

    typedef dbr_time_short DbrType;
    typedef EpicsPvTimeShort EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      // determine DBR offset from the beginning of the pv object
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;
      offset = defineFieldsTimeDbr(type, offset);
      offset += sizeof(int16_t); // RISC_pad
      offset = defineValueField<int16_t>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_TIME_FLOAT> {
    typedef dbr_time_float DbrType;
    typedef EpicsPvTimeFloat EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      // determine DBR offset from the beginning of the pv object
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;
      offset = defineFieldsTimeDbr(type, offset);
      offset = defineValueField<float>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_TIME_ENUM> {
    typedef dbr_time_enum DbrType;
    typedef EpicsPvTimeEnum EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      // determine DBR offset from the beginning of the pv object
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;
      offset = defineFieldsTimeDbr(type, offset);
      offset += sizeof(int16_t); // RISC_pad
      offset = defineValueField<uint16_t>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_TIME_CHAR> {
    typedef dbr_time_char DbrType;
    typedef EpicsPvTimeChar EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      // determine DBR offset from the beginning of the pv object
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;
      offset = defineFieldsTimeDbr(type, offset);
      offset += sizeof(int16_t); // RISC_pad
      offset += sizeof(int8_t); // RISC_pad
      offset = defineValueField<uint8_t>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_TIME_LONG> {
    typedef dbr_time_long DbrType;
    typedef EpicsPvTimeLong EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      // determine DBR offset from the beginning of the pv object
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;
      offset = defineFieldsTimeDbr(type, offset);
      offset = defineValueField<int32_t>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_TIME_DOUBLE> {
    typedef dbr_time_double DbrType;
    typedef EpicsPvTimeDouble EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      // determine DBR offset from the beginning of the pv object
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;
      offset = defineFieldsTimeDbr(type, offset);
      offset += sizeof(int32_t); // RISC_pad
      offset = defineValueField<int32_t>(type, pv.numElements(), offset);
    }
  };


  template <> struct DbrId2Type<DBR_CTRL_STRING> {

    typedef dbr_sts_string DbrType;
    typedef EpicsPvCtrlString EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;

      type.insert_native<int16_t>( "status", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "severity", offset ) ;
      offset += sizeof(int16_t);
      offset = defineStringValueField(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_CTRL_SHORT> {

    typedef dbr_ctrl_short DbrType;
    typedef EpicsPvCtrlShort EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;

      type.insert_native<int16_t>( "status", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "severity", offset ) ;
      offset += sizeof(int16_t);

      type.insert( "units", offset, _unitsType() ) ;
      offset += MAX_UNITS_SIZE;

      typedef int16_t value_type;
      type.insert_native<value_type>( "upper_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_ctrl_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_ctrl_limit", offset ) ;
      offset += sizeof(value_type);

      offset = defineValueField<value_type>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_CTRL_FLOAT> {
    typedef dbr_ctrl_float DbrType;
    typedef EpicsPvCtrlFloat EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;

      type.insert_native<int16_t>( "status", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "severity", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "precision", offset ) ;
      offset += sizeof(int16_t);
      offset += sizeof(int16_t);  // RISC_pad

      type.insert( "units", offset, _unitsType() ) ;
      offset += MAX_UNITS_SIZE;

      typedef float value_type;
      type.insert_native<value_type>( "lower_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_ctrl_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_ctrl_limit", offset ) ;
      offset += sizeof(value_type);

      offset = defineValueField<value_type>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_CTRL_ENUM> {
    typedef dbr_ctrl_enum DbrType;
    typedef EpicsPvCtrlEnum EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;

      type.insert_native<int16_t>( "status", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "severity", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "no_str", offset ) ;
      offset += sizeof(int16_t);

      hdf5pp::Type arrayType = hdf5pp::ArrayType::arrayType( _enumStrType(), pv.dbr().no_str() ) ;
      type.insert( "strs", offset, arrayType ) ;
      offset += MAX_ENUM_STATES*MAX_ENUM_STRING_SIZE;

      offset = defineValueField<uint16_t>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_CTRL_CHAR> {
    typedef dbr_ctrl_char DbrType;
    typedef EpicsPvCtrlChar EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;

      type.insert_native<int16_t>( "status", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "severity", offset ) ;
      offset += sizeof(int16_t);

      type.insert( "units", offset, _unitsType() ) ;
      offset += MAX_UNITS_SIZE;

      typedef uint8_t value_type;
      type.insert_native<value_type>( "upper_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_ctrl_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_ctrl_limit", offset ) ;
      offset += sizeof(value_type);

      offset += sizeof(uint8_t);    // RISC_pad

      offset = defineValueField<value_type>(type, pv.numElements(), offset);
    }
  };

  template <> struct DbrId2Type<DBR_CTRL_LONG> {
    typedef dbr_ctrl_long DbrType;
    typedef EpicsPvCtrlLong EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;

      type.insert_native<int16_t>( "status", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "severity", offset ) ;
      offset += sizeof(int16_t);

      type.insert( "units", offset, _unitsType() ) ;
      offset += MAX_UNITS_SIZE;

      typedef int32_t value_type;
      type.insert_native<value_type>( "upper_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_ctrl_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_ctrl_limit", offset ) ;
      offset += sizeof(value_type);

      offset = defineValueField<value_type>(type, pv.numElements(), offset);
    }
  };
  template <> struct DbrId2Type<DBR_CTRL_DOUBLE> {
    typedef dbr_ctrl_double DbrType;
    typedef EpicsPvCtrlDouble EpicsType;

    static void defineFields(hdf5pp::CompoundType& type, const EpicsType& pv)
    {
      unsigned offset = (const char*)&pv.dbr() - (const char*)&pv;

      type.insert_native<int16_t>( "status", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "severity", offset ) ;
      offset += sizeof(int16_t);
      type.insert_native<int16_t>( "precision", offset ) ;
      offset += sizeof(int16_t);
      offset += sizeof(int16_t);  // RISC_pad

      type.insert( "units", offset, _unitsType() ) ;
      offset += MAX_UNITS_SIZE;

      typedef double value_type;
      type.insert_native<value_type>( "lower_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_disp_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_alarm_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_warning_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "lower_ctrl_limit", offset ) ;
      offset += sizeof(value_type);
      type.insert_native<value_type>( "upper_ctrl_limit", offset ) ;
      offset += sizeof(value_type);

      offset = defineValueField<value_type>(type, pv.numElements(), offset);
    }
  };



}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// main method
CvtDataContFactoryEpics::container_type*
CvtDataContFactoryEpics::container( hdf5pp::Group group, const EpicsPvHeader& pv ) const
{
  hdf5pp::Type type = native_type(pv) ;

  MsgLog( "CvtDataContFactoryEpics", debug, "create container " << m_name << " with chunk size " << m_chunkSize ) ;
  return new container_type ( m_name, group, type, m_chunkSize, m_deflate, m_shuffle ) ;
}


// get the type for given PV
hdf5pp::Type
CvtDataContFactoryEpics::hdf_type( const EpicsPvHeader& pv, bool native )
{
  // switch based on actual PV type
  switch ( pv.dbrType() ) {
  case DBR_TIME_STRING:
    return hdf_type_time<DBR_TIME_STRING>(pv);
  case DBR_TIME_SHORT:
    return hdf_type_time<DBR_TIME_SHORT>(pv);
  case DBR_TIME_FLOAT:
    return hdf_type_time<DBR_TIME_FLOAT>(pv);
  case DBR_TIME_ENUM:
    return hdf_type_time<DBR_TIME_ENUM>(pv);
  case DBR_TIME_CHAR:
    return hdf_type_time<DBR_TIME_CHAR>(pv);
  case DBR_TIME_LONG:
    return hdf_type_time<DBR_TIME_LONG>(pv);
  case DBR_TIME_DOUBLE:
    return hdf_type_time<DBR_TIME_DOUBLE>(pv);
  case DBR_CTRL_STRING:
    return hdf_type_ctrl<DBR_CTRL_STRING>(pv);
  case DBR_CTRL_SHORT:
    return hdf_type_ctrl<DBR_CTRL_SHORT>(pv);
  case DBR_CTRL_FLOAT:
    return hdf_type_ctrl<DBR_CTRL_FLOAT>(pv);
  case DBR_CTRL_ENUM:
    return hdf_type_ctrl<DBR_CTRL_ENUM>(pv);
  case DBR_CTRL_CHAR:
    return hdf_type_ctrl<DBR_CTRL_CHAR>(pv);
  case DBR_CTRL_LONG:
    return hdf_type_ctrl<DBR_CTRL_LONG>(pv);
  case DBR_CTRL_DOUBLE:
    return hdf_type_ctrl<DBR_CTRL_DOUBLE>(pv);

  default:
    MsgLog( "CvtDataContFactoryEpics", error, "CvtDataContFactoryEpics: unexpected PV type = " << pv.dbrType() ) ;
  }
  return hdf5pp::Type();
}

template <int iDbrType>
hdf5pp::Type
CvtDataContFactoryEpics::hdf_type_ctrl( const Pds::Epics::EpicsPvHeader& hdr )
{
  typedef typename DbrId2Type<iDbrType>::EpicsType EpicsType;

  const EpicsType& pv = static_cast<const EpicsType&>(hdr);
  size_t size = pv._sizeof();

  // header is common to all EPICS types
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType(size) ;
  type.insert_native<int16_t>( "pvId", 0 ) ;
  type.insert_native<int16_t>( "dbrType", 2 ) ;
  type.insert_native<int16_t>( "numElements", 4 ) ;
  type.insert( "pvname", pv.pvName() - (const char*)&pv, _pvnameType() ) ;

  DbrId2Type<iDbrType>::defineFields(type, pv);

  return type ;
}

template <int iDbrType>
hdf5pp::Type
CvtDataContFactoryEpics::hdf_type_time( const Pds::Epics::EpicsPvHeader& hdr )
{
  typedef typename DbrId2Type<iDbrType>::EpicsType EpicsType;

  const EpicsType& pv = static_cast<const EpicsType&>(hdr);
  size_t size = pv._sizeof();

  // header is common to all EPICS types
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType(size) ;
  type.insert_native<int16_t>( "pvId", 0 ) ;
  type.insert_native<int16_t>( "dbrType", 2 ) ;
  type.insert_native<int16_t>( "numElements", 4 ) ;

  DbrId2Type<iDbrType>::defineFields(type, pv);

  return type ;
}


} // namespace O2OTranslator
