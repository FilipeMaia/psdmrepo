//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisTdcConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/AcqirisTdcConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/TypeTraits.h"
#include "hdf5pp/DataSet.h"
#include "hdf5pp/DataSpace.h"

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

AcqirisTdcChannel::AcqirisTdcChannel ( const Pds::Acqiris::TdcChannel& v )
{
  m_data.channel = v.channel() ;
  m_data.mode = v.mode() ;
  m_data.slope = v.slope() ;
  m_data.level = v.level() ;
}

hdf5pp::Type
AcqirisTdcChannel::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AcqirisTdcChannel::native_type()
{
  hdf5pp::EnumType<int32_t> chanType = hdf5pp::EnumType<int32_t>::enumType();
  chanType.insert("Veto", Pds::Acqiris::TdcChannel::Veto);
  chanType.insert("Common", Pds::Acqiris::TdcChannel::Common);
  chanType.insert("Input1", Pds::Acqiris::TdcChannel::Input1);
  chanType.insert("Input2", Pds::Acqiris::TdcChannel::Input2);
  chanType.insert("Input3", Pds::Acqiris::TdcChannel::Input3);
  chanType.insert("Input4", Pds::Acqiris::TdcChannel::Input4);
  chanType.insert("Input5", Pds::Acqiris::TdcChannel::Input5);
  chanType.insert("Input6", Pds::Acqiris::TdcChannel::Input6);

  hdf5pp::EnumType<uint16_t> modeType = hdf5pp::EnumType<uint16_t>::enumType();
  modeType.insert("Active", Pds::Acqiris::TdcChannel::Active);
  modeType.insert("Inactive", Pds::Acqiris::TdcChannel::Inactive);

  hdf5pp::EnumType<uint16_t> slopeType = hdf5pp::EnumType<uint16_t>::enumType();
  slopeType.insert("Positive", Pds::Acqiris::TdcChannel::Positive);
  slopeType.insert("Negative", Pds::Acqiris::TdcChannel::Negative);

  hdf5pp::CompoundType tdcChanType = hdf5pp::CompoundType::compoundType<AcqirisTdcChannel_Data>() ;
  tdcChanType.insert( "channel", offsetof(AcqirisTdcChannel_Data,channel), chanType ) ;
  tdcChanType.insert( "mode", offsetof(AcqirisTdcChannel_Data,mode), modeType ) ;
  tdcChanType.insert( "slope", offsetof(AcqirisTdcChannel_Data,slope), slopeType ) ;
  tdcChanType.insert_native<double>( "level", offsetof(AcqirisTdcChannel_Data,level) ) ;

  return tdcChanType ;
}



AcqirisTdcAuxIO::AcqirisTdcAuxIO ( const Pds::Acqiris::TdcAuxIO& v )
{
  m_data.channel = v.channel() ;
  m_data.mode = v.mode() ;
  m_data.term = v.term() ;
}

hdf5pp::Type
AcqirisTdcAuxIO::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AcqirisTdcAuxIO::native_type()
{
  hdf5pp::EnumType<uint16_t> chanType = hdf5pp::EnumType<uint16_t>::enumType();
  chanType.insert("IOAux1", Pds::Acqiris::TdcAuxIO::IOAux1);
  chanType.insert("IOAux2", Pds::Acqiris::TdcAuxIO::IOAux2);

  hdf5pp::EnumType<uint16_t> modeType = hdf5pp::EnumType<uint16_t>::enumType();
  modeType.insert("BankSwitch", Pds::Acqiris::TdcAuxIO::BankSwitch);
  modeType.insert("Marker",     Pds::Acqiris::TdcAuxIO::Marker);
  modeType.insert("OutputLo",   Pds::Acqiris::TdcAuxIO::OutputLo);
  modeType.insert("OutputHi",   Pds::Acqiris::TdcAuxIO::OutputHi);

  hdf5pp::EnumType<uint16_t> termType = hdf5pp::EnumType<uint16_t>::enumType();
  termType.insert("ZHigh", Pds::Acqiris::TdcAuxIO::ZHigh);
  termType.insert("Z50", Pds::Acqiris::TdcAuxIO::Z50);

  hdf5pp::CompoundType tdcAuxIOType = hdf5pp::CompoundType::compoundType<AcqirisTdcAuxIO_Data>() ;
  tdcAuxIOType.insert( "channel", offsetof(AcqirisTdcAuxIO_Data,channel), chanType ) ;
  tdcAuxIOType.insert( "mode", offsetof(AcqirisTdcAuxIO_Data,mode), modeType ) ;
  tdcAuxIOType.insert( "term", offsetof(AcqirisTdcAuxIO_Data,term), termType ) ;

  return tdcAuxIOType ;
}


AcqirisTdcVetoIO::AcqirisTdcVetoIO ( const Pds::Acqiris::TdcVetoIO& v )
{
  m_data.channel = v.channel() ;
  m_data.mode = v.mode() ;
  m_data.term = v.term() ;
}

hdf5pp::Type
AcqirisTdcVetoIO::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AcqirisTdcVetoIO::native_type()
{
  hdf5pp::EnumType<uint16_t> chanType = hdf5pp::EnumType<uint16_t>::enumType();
  chanType.insert("ChVeto", Pds::Acqiris::TdcVetoIO::ChVeto);

  hdf5pp::EnumType<uint16_t> modeType = hdf5pp::EnumType<uint16_t>::enumType();
  modeType.insert("Veto", Pds::Acqiris::TdcVetoIO::Veto);
  modeType.insert("SwitchVeto", Pds::Acqiris::TdcVetoIO::SwitchVeto);
  modeType.insert("InvertedVeto", Pds::Acqiris::TdcVetoIO::InvertedVeto);
  modeType.insert("InvertedSwitchVeto", Pds::Acqiris::TdcVetoIO::InvertedSwitchVeto);

  hdf5pp::EnumType<uint16_t> termType = hdf5pp::EnumType<uint16_t>::enumType();
  termType.insert("ZHigh", Pds::Acqiris::TdcVetoIO::ZHigh);
  termType.insert("Z50", Pds::Acqiris::TdcVetoIO::Z50);

  hdf5pp::CompoundType tdcVetoIOType = hdf5pp::CompoundType::compoundType<AcqirisTdcVetoIO_Data>() ;
  tdcVetoIOType.insert( "channel", offsetof(AcqirisTdcVetoIO_Data,channel), chanType ) ;
  tdcVetoIOType.insert( "mode", offsetof(AcqirisTdcVetoIO_Data,mode), modeType ) ;
  tdcVetoIOType.insert( "term", offsetof(AcqirisTdcVetoIO_Data,term), termType ) ;

  return tdcVetoIOType ;
}



void
AcqirisTdcConfigV1::store ( const XtcType& config, hdf5pp::Group grp )
{
  // make array data set for subobject
  AcqirisTdcChannel tdcChan[Pds::Acqiris::TdcConfigV1::NChannels] ;
  for ( int i = 0 ; i < Pds::Acqiris::TdcConfigV1::NChannels ; ++ i ) {
    tdcChan[i] = AcqirisTdcChannel( config.channel(i) ) ;
  }
  storeDataObjects ( Pds::Acqiris::TdcConfigV1::NChannels, tdcChan, "channel", grp ) ;

  // make array data set for subobject
  AcqirisTdcAuxIO auxio[Pds::Acqiris::TdcConfigV1::NAuxIO] ;
  for ( int i = 0 ; i < Pds::Acqiris::TdcConfigV1::NAuxIO ; ++ i ) {
    auxio[i] = AcqirisTdcAuxIO( config.auxio(i) ) ;
  }
  storeDataObjects ( Pds::Acqiris::TdcConfigV1::NAuxIO, auxio, "auxio", grp ) ;

  // make scalar data set for subobject
  AcqirisTdcVetoIO veto ( config.veto() ) ;
  storeDataObject ( veto, "veto", grp ) ;

}

} // namespace H5DataTypes
