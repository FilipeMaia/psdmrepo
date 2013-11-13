//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Gsc16aiConfigV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/Gsc16aiConfigV1.h"

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

//----------------
// Constructors --
//----------------
Gsc16aiConfigV1::Gsc16aiConfigV1 (const XtcType& data)
  : _voltageRange(data.voltageRange())
  , _firstChan(data.firstChan())
  , _lastChan(data.lastChan())
  , _inputMode(data.inputMode())
  , _triggerMode(data.triggerMode())
  , _dataFormat(data.dataFormat())
  , _fps(data.fps())
  , _autocalibEnable(data.autocalibEnable())
  , _timeTagEnable(data.timeTagEnable())
{
}

hdf5pp::Type
Gsc16aiConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
Gsc16aiConfigV1::native_type()
{
  hdf5pp::EnumType<uint16_t> voltRangeEnumType = hdf5pp::EnumType<uint16_t>::enumType() ;
  voltRangeEnumType.insert ( "VoltageRange_10V", XtcType::VoltageRange_10V ) ;
  voltRangeEnumType.insert ( "VoltageRange_5V", XtcType::VoltageRange_5V ) ;
  voltRangeEnumType.insert ( "VoltageRange_2_5V", XtcType::VoltageRange_2_5V ) ;

  hdf5pp::EnumType<uint16_t> inputModeEnumType = hdf5pp::EnumType<uint16_t>::enumType() ;
  inputModeEnumType.insert ( "InputMode_Differential", XtcType::InputMode_Differential ) ;
  inputModeEnumType.insert ( "InputMode_Zero", XtcType::InputMode_Zero ) ;
  inputModeEnumType.insert ( "InputMode_Vref", XtcType::InputMode_Vref ) ;

  hdf5pp::EnumType<uint16_t> trigModeEnumType = hdf5pp::EnumType<uint16_t>::enumType() ;
  trigModeEnumType.insert ( "TriggerMode_ExtPos", XtcType::TriggerMode_ExtPos ) ;
  trigModeEnumType.insert ( "TriggerMode_ExtNeg", XtcType::TriggerMode_ExtNeg ) ;
  trigModeEnumType.insert ( "TriggerMode_IntClk", XtcType::TriggerMode_IntClk ) ;

  hdf5pp::EnumType<uint16_t> dataFormatEnumType = hdf5pp::EnumType<uint16_t>::enumType() ;
  dataFormatEnumType.insert ( "DataFormat_TwosComplement", XtcType::DataFormat_TwosComplement ) ;
  dataFormatEnumType.insert ( "DataFormat_OffsetBinary", XtcType::DataFormat_OffsetBinary ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<Gsc16aiConfigV1>() ;
  confType.insert("voltageRange", offsetof(Gsc16aiConfigV1, _voltageRange), voltRangeEnumType) ;
  confType.insert_native<uint16_t>("firstChan", offsetof(Gsc16aiConfigV1, _firstChan)) ;
  confType.insert_native<uint16_t>("lastChan", offsetof(Gsc16aiConfigV1, _lastChan)) ;
  confType.insert("inputMode", offsetof(Gsc16aiConfigV1, _inputMode), inputModeEnumType) ;
  confType.insert("triggerMode", offsetof(Gsc16aiConfigV1, _triggerMode), trigModeEnumType) ;
  confType.insert("dataFormat", offsetof(Gsc16aiConfigV1, _dataFormat), dataFormatEnumType) ;
  confType.insert_native<uint16_t>("fps", offsetof(Gsc16aiConfigV1, _fps)) ;
  confType.insert_native<uint8_t>("autocalibEnable", offsetof(Gsc16aiConfigV1, _autocalibEnable)) ;
  confType.insert_native<uint8_t>("timeTagEnable", offsetof(Gsc16aiConfigV1, _timeTagEnable)) ;

  return confType ;
}

void
Gsc16aiConfigV1::store(const XtcType& config, hdf5pp::Group location)
{
  // make scalar data set for main object
  Gsc16aiConfigV1 data(config);
  storeDataObject(data, "config", location);
}

} // namespace H5DataTypes
