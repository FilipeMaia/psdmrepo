//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OHdf5Writer...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OHdf5Writer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <string>
#include <uuid/uuid.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/AcqirisConfigV1.h"
#include "H5DataTypes/AcqirisTdcConfigV1.h"
#include "H5DataTypes/BldDataEBeamV0.h"
#include "H5DataTypes/BldDataEBeamV1.h"
#include "H5DataTypes/BldDataEBeamV2.h"
#include "H5DataTypes/BldDataFEEGasDetEnergy.h"
#include "H5DataTypes/BldDataIpimbV0.h"
#include "H5DataTypes/BldDataIpimbV1.h"
#include "H5DataTypes/BldDataPhaseCavity.h"
#include "H5DataTypes/BldDataPimV1.h"
#include "H5DataTypes/CameraFrameFexConfigV1.h"
#include "H5DataTypes/CameraFrameV1.h"
#include "H5DataTypes/CameraTwoDGaussianV1.h"
#include "H5DataTypes/ControlDataConfigV1.h"
#include "H5DataTypes/CsPadConfigV1.h"
#include "H5DataTypes/CsPadConfigV2.h"
#include "H5DataTypes/CsPadConfigV3.h"
#include "H5DataTypes/EncoderConfigV1.h"
#include "H5DataTypes/EncoderConfigV2.h"
#include "H5DataTypes/EncoderDataV1.h"
#include "H5DataTypes/EncoderDataV2.h"
#include "H5DataTypes/EpicsPvHeader.h"
#include "H5DataTypes/EvrConfigV1.h"
#include "H5DataTypes/EvrConfigV2.h"
#include "H5DataTypes/EvrConfigV3.h"
#include "H5DataTypes/EvrConfigV4.h"
#include "H5DataTypes/EvrConfigV5.h"
#include "H5DataTypes/EvrIOConfigV1.h"
#include "H5DataTypes/FccdConfigV1.h"
#include "H5DataTypes/FccdConfigV2.h"
#include "H5DataTypes/Gsc16aiConfigV1.h"
#include "H5DataTypes/IpimbConfigV1.h"
#include "H5DataTypes/IpimbConfigV2.h"
#include "H5DataTypes/IpimbDataV1.h"
#include "H5DataTypes/IpimbDataV2.h"
#include "H5DataTypes/LusiDiodeFexConfigV1.h"
#include "H5DataTypes/LusiDiodeFexConfigV2.h"
#include "H5DataTypes/LusiDiodeFexV1.h"
#include "H5DataTypes/LusiIpmFexConfigV1.h"
#include "H5DataTypes/LusiIpmFexConfigV2.h"
#include "H5DataTypes/LusiIpmFexV1.h"
#include "H5DataTypes/LusiPimImageConfigV1.h"
#include "H5DataTypes/Opal1kConfigV1.h"
#include "H5DataTypes/PnCCDConfigV1.h"
#include "H5DataTypes/PnCCDConfigV2.h"
#include "H5DataTypes/PrincetonConfigV1.h"
#include "H5DataTypes/PrincetonConfigV2.h"
#include "H5DataTypes/PrincetonInfoV1.h"
#include "H5DataTypes/PulnixTM6740ConfigV1.h"
#include "H5DataTypes/PulnixTM6740ConfigV2.h"
#include "LusiTime/Time.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/AcqirisDataDescV1Cvt.h"
#include "O2OTranslator/AcqirisTdcDataV1Cvt.h"
#include "O2OTranslator/CameraFrameV1Cvt.h"
#include "O2OTranslator/ConfigDataTypeCvt.h"
#include "O2OTranslator/CsPadElementV1Cvt.h"
#include "O2OTranslator/CsPadElementV2Cvt.h"
#include "O2OTranslator/CsPadCalibV1Cvt.h"
#include "O2OTranslator/CsPadMiniCalibV1Cvt.h"
#include "O2OTranslator/CsPadMiniElementV1Cvt.h"
#include "O2OTranslator/EvrDataV3Cvt.h"
#include "O2OTranslator/EvtDataTypeCvtDef.h"
#include "O2OTranslator/EpicsDataTypeCvt.h"
#include "O2OTranslator/Gsc16aiDataV1Cvt.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OFileNameFactory.h"
#include "O2OTranslator/O2OMetaData.h"
#include "O2OTranslator/PnCCDFrameV1Cvt.h"
#include "O2OTranslator/PrincetonFrameV1Cvt.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/Level.hh"
#include "pdsdata/xtc/Src.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace O2OTranslator ;

namespace {

  const char* logger = "HDF5Writer" ;

  // printable state name
  const char* stateName ( O2OHdf5Writer::State state ) {
    switch ( state ) {
      case O2OHdf5Writer::Undefined :
        return "Undefined" ;
      case O2OHdf5Writer::Mapped :
        return "Mapped" ;
      case O2OHdf5Writer::Configured :
        return "Configured" ;
      case O2OHdf5Writer::Running :
        return "Running" ;
      case O2OHdf5Writer::CalibCycle :
        return "CalibCycle" ;
      case O2OHdf5Writer::NumberOfStates :
        break ;
    }
    return "*ERROR*" ;
  }

  // store time as attributes to the group
  void storeClock ( hdf5pp::Group group, const Pds::ClockTime& clock, const std::string& what )
  {
    hdf5pp::Attribute<uint32_t> attr1 = group.createAttr<uint32_t> ( what+".seconds" ) ;
    attr1.store ( clock.seconds() ) ;
    hdf5pp::Attribute<uint32_t> attr2 = group.createAttr<uint32_t> ( what+".nanoseconds" ) ;
    attr2.store ( clock.nanoseconds() ) ;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {


//----------------
// Constructors --
//----------------
O2OHdf5Writer::O2OHdf5Writer ( const O2OFileNameFactory& nameFactory,
                               bool overwrite,
                               SplitMode split,
                               hsize_t splitSize,
                               int compression,
                               bool extGroups,
                               const O2OMetaData& metadata )
  : O2OXtcScannerI()
  , m_nameFactory( nameFactory )
  , m_file()
  , m_state()
  , m_groups()
  , m_eventTime()
  , m_cvtMap()
  , m_compression(compression)
  , m_extGroups(extGroups)
  , m_metadata(metadata)
  , m_stateCounters()
  , m_transition(Pds::TransitionId::Unknown)
  , m_configStore()
  , m_calibStore()
  , m_transClock()
{
  std::fill_n(m_stateCounters, int(NumberOfStates), 0U);
  std::fill_n(m_transClock, int(Pds::TransitionId::NumberOf), LusiTime::Time(0,0));
    
  std::string fileTempl = m_nameFactory.makeH5Path ( split != NoSplit ) ;
  MsgLog( logger, debug, "O2OHdf5Writer - open output file " << fileTempl ) ;

  // Disable printing of error messages
  //stat = H5Eset_auto2( H5E_DEFAULT, 0, 0 ) ;

  // we want to create new file
  hdf5pp::PListFileAccess fapl ;
  if ( split == Family ) {
    // use FAMILY driver
    fapl.set_family_driver ( splitSize, hdf5pp::PListFileAccess() ) ;
  }

  // change the size of the B-Tree for chunked datasets
  hdf5pp::PListFileCreate fcpl;
  fcpl.set_istore_k(2); 
  fcpl.set_sym_k(2, 2); 
  
  hdf5pp::File::CreateMode mode = overwrite ? hdf5pp::File::Truncate : hdf5pp::File::Exclusive ;
  m_file = hdf5pp::File::create ( fileTempl, mode, fcpl, fapl ) ;

  // add UUID to the file attributes
  uuid_t uuid ;
  uuid_generate( uuid );
  char uuid_buf[64] ;
  uuid_unparse ( uuid, uuid_buf ) ;
  m_file.createAttr<const char*> ("UUID").store ( uuid_buf ) ;

  // add some metadata to the top group
  LusiTime::Time ctime = LusiTime::Time::now() ;
  m_file.createAttr<const char*> ("origin").store ( "translator" ) ;
  m_file.createAttr<const char*> ("created").store ( ctime.toString().c_str() ) ;

  m_file.createAttr<uint32_t> ("runNumber").store ( m_metadata.runNumber() ) ;
  m_file.createAttr<const char*> ("runType").store ( m_metadata.runType().c_str() ) ;
  m_file.createAttr<const char*> ("experiment").store ( m_metadata.experiment().c_str() ) ;

  // we are in bad state, this state should never be popped
  m_state.push(Undefined) ;

  // store top group
  m_groups.push ( m_file.openGroup("/") ) ;

  typedef O2OMetaData::const_iterator MDIter ;
  for ( MDIter it = m_metadata.extra_begin() ; it != m_metadata.extra_end() ; ++ it ) {
    try {
      m_file.createAttr<const char*> (it->first).store ( it->second.c_str() ) ;
    } catch ( std::exception& e ) {
      // this is not fatal, just print error message and continue
      MsgLog( logger, error, "failed to store metadata: " << e.what()
          << "\n\tkey='" << it->first << "', value='" << it->second << "'" ) ;
    }
  }

  // instantiate all factories
  DataTypeCvtPtr converter ;
  uint32_t typeId ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::AcqirisConfigV1> ( "Acqiris::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_AcqConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::AcqirisTdcConfigV1> ( "Acqiris::AcqirisTdcConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_AcqTdcConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::Opal1kConfigV1> ( "Opal1k::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_Opal1kConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::PulnixTM6740ConfigV1> ( "Pulnix::TM6740ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_TM6740Config,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::PulnixTM6740ConfigV2> ( "Pulnix::TM6740ConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_TM6740Config,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::CameraFrameFexConfigV1> ( "Camera::FrameFexConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_FrameFexConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EvrConfigV1> ( "EvrData::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EvrConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EvrConfigV2> ( "EvrData::ConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EvrConfig,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EvrConfigV3> ( "EvrData::ConfigV3" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EvrConfig,3).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EvrConfigV4> ( "EvrData::ConfigV4" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EvrConfig,4).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EvrConfigV5> ( "EvrData::ConfigV5" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EvrConfig,5).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EvrIOConfigV1> ( "EvrData::IOConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EvrIOConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::ControlDataConfigV1> ( "ControlData::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_ControlConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::PnCCDConfigV1> ( "PNCCD::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_pnCCDconfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::PnCCDConfigV2> ( "PNCCD::ConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_pnCCDconfig,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::PrincetonConfigV1> ( "Princeton::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_PrincetonConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::PrincetonConfigV2> ( "Princeton::ConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_PrincetonConfig,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::FccdConfigV1> ( "FCCD::FccdConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_FccdConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::FccdConfigV2> ( "FCCD::FccdConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_FccdConfig,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::IpimbConfigV1> ( "Ipimb::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_IpimbConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::IpimbConfigV2> ( "Ipimb::ConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_IpimbConfig,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EncoderConfigV1> ( "Encoder::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EncoderConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EncoderConfigV2> ( "Encoder::ConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EncoderConfig,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::LusiDiodeFexConfigV1> ( "Lusi::DiodeFexConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_DiodeFexConfig, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::LusiDiodeFexConfigV2> ( "Lusi::DiodeFexConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_DiodeFexConfig, 2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::LusiIpmFexConfigV1> ( "Lusi::IpmFexConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_IpmFexConfig, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::LusiIpmFexConfigV2> ( "Lusi::IpmFexConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_IpmFexConfig, 2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::LusiPimImageConfigV1> ( "Lusi::PimImageConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_PimImageConfig, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::CsPadConfigV1> ( "CsPad::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::CsPadConfigV2> ( "CsPad::ConfigV2" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::CsPadConfigV3> ( "CsPad::ConfigV3" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 3).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::Gsc16aiConfigV1> ( "Gsc16ai::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_Gsc16aiConfig, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // special converter object for CsPad calibration data
  converter.reset( new CsPadCalibV1Cvt ( "CsPad::CalibV1", m_metadata, m_calibStore ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 3).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // special converter object for CsPad calibration data
  converter.reset( new CsPadMiniCalibV1Cvt ( "CsPad::CalibV1", m_metadata, m_calibStore ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadConfig, 3).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  hsize_t chunk_size = 16*1024 ;

  // instantiate all factories for event converters
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::CameraTwoDGaussianV1> (
      "Camera::TwoDGaussianV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_TwoDGaussian,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 0
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::BldDataFEEGasDetEnergy> (
      "Bld::BldDataFEEGasDetEnergy", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_FEEGasDetEnergy,0).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 0
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::BldDataEBeamV0> (
      "Bld::BldDataEBeamV0", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EBeam,0).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::BldDataEBeamV1> (
      "Bld::BldDataEBeamV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EBeam,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 2
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::BldDataEBeamV2> (
      "Bld::BldDataEBeamV2", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EBeam,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 0
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::BldDataIpimbV0> (
      "Bld::BldDataIpimbV0", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_SharedIpimb,0).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::BldDataIpimbV1> (
      "Bld::BldDataIpimbV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_SharedIpimb,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 0
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::BldDataPhaseCavity> (
      "Bld::BldDataPhaseCavity", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_PhaseCavity,0).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::BldDataPimV1> (
      "Bld::BldDataPimV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_SharedPim,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::EncoderDataV1> (
      "Encoder::DataV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EncoderData,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 2
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::EncoderDataV2> (
      "Encoder::DataV2", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EncoderData,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::IpimbDataV1> (
      "Ipimb::DataV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_IpimbData,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::IpimbDataV2> (
      "Ipimb::DataV2", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_IpimbData,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // special converter for CameraFrame type
  converter.reset( new CameraFrameV1Cvt ( "Camera::FrameV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_Frame,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for Acqiris::DataDescV1, it needs two types of data
  converter.reset( new AcqirisDataDescV1Cvt ( "Acqiris::DataDescV1", m_configStore, chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_AcqWaveform,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for Acqiris::TdcDataV1
  converter.reset( new AcqirisTdcDataV1Cvt ( "Acqiris::TdcDataV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_AcqTdcData,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for EvrData::DataV3, it needs two types of data
  converter.reset( new EvrDataV3Cvt ( "EvrData::DataV3", m_configStore, chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EvrData,3).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for PNCCD::FrameV1, it needs two types of data
  converter.reset( new PnCCDFrameV1Cvt ( "PNCCD::FrameV1", m_configStore, chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_pnCCDframe,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for Princeton::FrameV1, it needs two types of data
  converter.reset( new PrincetonFrameV1Cvt ( "Princeton::FrameV1", m_configStore, chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_PrincetonFrame,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::PrincetonInfoV1> (
      "Princeton::InfoV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_PrincetonInfo, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // temporary/diagnostics  Epics converter (headers only)
//  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::EpicsPvHeader> (
//      "Epics::EpicsPvHeader", chunk_size, m_compression ) ) ;
//  typeId =  Pds::TypeId(Pds::TypeId::Id_Epics,1).value() ;
//  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // Epics converter, non-default chunk size
  converter.reset( new EpicsDataTypeCvt( "Epics::EpicsPv", 1024, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_Epics,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::LusiDiodeFexV1> (
      "Lusi::DiodeFexV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_DiodeFex, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // version for this type is 1
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::LusiIpmFexV1> (
      "Lusi::IpmFexV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_IpmFex, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for CsPad::ElementV1, it needs two types of data
  converter.reset( new CsPadElementV1Cvt ( "CsPad::ElementV1", m_configStore, 
                                           m_calibStore, chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadElement,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for CsPad::ElementV2, it needs two types of data
  converter.reset( new CsPadElementV2Cvt ( "CsPad::ElementV2", m_configStore, 
                                           m_calibStore, chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_CspadElement,2).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for CsPad::MiniElementV1, it needs calibrations
  converter.reset( new CsPadMiniElementV1Cvt (
      "CsPad::MiniElementV1", m_calibStore, chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_Cspad2x2Element, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for Gsc16ai::DataV1, it needs two types of data
  converter.reset( new Gsc16aiDataV1Cvt ( "Gsc16ai::DataV1", m_configStore, chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_Gsc16aiData, 1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

}

//--------------
// Destructor --
//--------------
O2OHdf5Writer::~O2OHdf5Writer ()
{
  MsgLog( logger, debug, "O2OHdf5Writer - close output file" ) ;

  m_cvtMap.clear() ;
  m_file.close() ;
}

// signal start/end of the event (datagram)
bool
O2OHdf5Writer::eventStart ( const Pds::Dgram& dgram )
{
  MsgLog( logger, debug, "O2OHdf5Writer::eventStart " << Pds::TransitionId::name(dgram.seq.service())
          << " dgram.seq.type=" << dgram.seq.type()
          << " dgram.seq.service=" << Pds::TransitionId::name(dgram.seq.service()) ) ;

  m_transition = dgram.seq.service();
  Pds::ClockTime clock = dgram.seq.clock();
  LusiTime::Time t(clock.seconds(), clock.nanoseconds());
  
  // store current event time
  m_eventTime = H5DataTypes::XtcClockTime(clock) ;
  

  bool skip = false;
  switch ( m_transition ) {

    case Pds::TransitionId::Map :

      if ( t != m_transClock[m_transition] ) {
        // close all states
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->closeGroup( dgram, Configured ) ;
        this->closeGroup( dgram, Mapped ) ;
        this->openGroup( dgram, Mapped ) ;
      }
      break ;

    case Pds::TransitionId::Unmap :

      if ( t != m_transClock[m_transition] ) {
        // close all states
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->closeGroup( dgram, Configured ) ;
        this->closeGroup( dgram, Mapped ) ;
      }
      break ;

    case Pds::TransitionId::Configure :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Mapped
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->closeGroup( dgram, Configured ) ;
        this->openGroup( dgram, Configured ) ;
      }
      break ;

    case Pds::TransitionId::Unconfigure :

      if ( t != m_transClock[m_transition] ) {
         // close all states up to Mapped
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->closeGroup( dgram, Configured ) ;
        this->closeGroup( dgram, Mapped ) ;
      }
      break ;

    case Pds::TransitionId::BeginRun :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Configured
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->openGroup( dgram, Running ) ;
      }
      break ;

    case Pds::TransitionId::EndRun :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Configured
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
      }
      break ;

    case Pds::TransitionId::BeginCalibCycle :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Running
        this->closeGroup( dgram, CalibCycle ) ;
        this->openGroup( dgram, CalibCycle ) ;
      }
      break ;

    case Pds::TransitionId::EndCalibCycle :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Running
        this->closeGroup( dgram, CalibCycle ) ;
      }

      break ;

    case Pds::TransitionId::L1Accept :

      // check the time, should not be sooner than begin of calib cycle
      if ( t < m_transClock[Pds::TransitionId::BeginCalibCycle] ) {
        MsgLog( logger, warning, "O2OHdf5Writer::eventStart: L1Accept time out of sync: "
              << Pds::TransitionId::name(dgram.seq.service())
              << " BeginCalibCycle time=" << m_transClock[Pds::TransitionId::BeginCalibCycle].toString("S%s%f")
              << " L1Accept time=" << t.toString("S%s%f")) ;
        skip = true;
      }

      break ;

    case Pds::TransitionId::Enable :
    case Pds::TransitionId::Disable :
    case Pds::TransitionId::Unknown :
    case Pds::TransitionId::Reset :
    case Pds::TransitionId::NumberOf :

      break ;
  }

  // store the time of the transition
  m_transClock[m_transition] = t;
  
  MsgLog( logger, debug, "O2OHdf5Writer -- now in the state " << ::stateName(m_state.top()) ) ;
  
  return not skip;
}

void
O2OHdf5Writer::eventEnd ( const Pds::Dgram& dgram )
{
}


void
O2OHdf5Writer::openGroup ( const Pds::Dgram& dgram, State state )
{
  // get the counter for this state
  unsigned counter = m_stateCounters[state] ;
  ++ m_stateCounters[state] ;

  // reset counter for sub-states, note there are no breaks
  switch( state ) {
  case Undefined:
    m_stateCounters[Mapped] = 0;
  case Mapped:
    m_stateCounters[Configured] = 0;
  case Configured:
    m_stateCounters[Running] = 0;
  case Running:
    m_stateCounters[CalibCycle] = 0;
  case CalibCycle:
  case NumberOfStates:
    break;
  }

  // create group
  const std::string& name = groupName ( state, counter ) ;
  MsgLog( logger, debug, "HDF5Writer -- creating group " << name ) ;
  hdf5pp::Group group;
  if (m_groups.top().hasChild(name)) {
    group = m_groups.top().openGroup(name) ;
  } else {
    group = m_groups.top().createGroup(name) ;
  }

  // store transition time as couple of attributes to this new group
  ::storeClock ( group, dgram.seq.clock(), "start" ) ;

  // switch to mapped state
  m_state.push(state) ;
  m_groups.push( group ) ;

  // notify all converters
  for ( CvtMap::iterator it = m_cvtMap.begin() ; it != m_cvtMap.end() ; ++ it ) {
    it->second->openGroup( group ) ;
  }
}

void
O2OHdf5Writer::closeGroup ( const Pds::Dgram& dgram, State state )
{
  if ( m_state.top() != state ) return ;

  // store transition time as couple of attributes to this new group
  ::storeClock ( m_groups.top(), dgram.seq.clock(), "end" ) ;

  // notify all converters
  for ( CvtMap::iterator it = m_cvtMap.begin() ; it != m_cvtMap.end() ; ++ it ) {
    it->second->closeGroup( m_groups.top() ) ;
  }

  // close the group
  m_groups.top().close() ;

  // switch back to previous state
  m_state.pop() ;
  m_groups.pop() ;
}

// signal start/end of the level
void
O2OHdf5Writer::levelStart ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::levelStart " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::levelEnd ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::levelEnd " << Pds::Level::name(src.level()) ) ;
}

// visit the data object
void
O2OHdf5Writer::dataObject ( const void* data, size_t size,
    const Pds::TypeId& typeId, const O2OXtcSrc& src )
{
  // for Configure and BeginCalibCycle transitions store config objects at Source level
  if ( ( m_transition == Pds::TransitionId::Configure
      or m_transition == Pds::TransitionId::BeginCalibCycle )  ) {
    MsgLog( logger, debug, "O2OHdf5Writer: store config object "
        << src.top()
        << " name=" <<  Pds::TypeId::name(typeId.id())
        << " version=" <<  typeId.version() ) ;
    m_configStore.store(typeId, src.top(), data, size);
  }
  
  // find this type in the converter map
  CvtMap::iterator it = m_cvtMap.find( typeId.value() ) ;
  if ( it != m_cvtMap.end() ) {

    do {

      DataTypeCvtPtr converter = it->second ;
      try {
        converter->convert( data, size, typeId, src, m_eventTime ) ;
      } catch (const O2OXTCSizeException& ex) {
        // on size mismatch print an error message but continue
        MsgLog(logger, error, ex.what());
      }

      ++ it ;

    } while ( it != m_cvtMap.end() and it->first == typeId.value() ) ;

  } else {

    MsgLogRoot( error, "O2OHdf5Writer::dataObject -- unexpected type or version: "
                << Pds::TypeId::name(typeId.id()) << "/" << typeId.version() ) ;

  }

}

// Construct a group name
std::string
O2OHdf5Writer::groupName( State state, unsigned counter ) const
{
  const char* prefix = "Undefined" ;
  switch ( state ) {
    case O2OHdf5Writer::Mapped :
      prefix = "Map" ;
      break ;
    case O2OHdf5Writer::Configured :
      prefix = "Configure" ;
      break ;
    case O2OHdf5Writer::Running :
      prefix = "Run" ;
      break ;
    case O2OHdf5Writer::CalibCycle :
      prefix = "CalibCycle" ;
      break ;
    case O2OHdf5Writer::Undefined :
    default :
      prefix = "Undefined" ;
      break ;
    case O2OHdf5Writer::NumberOfStates:
      break ;
  }

  if ( m_extGroups ) {
    // dump seconds as a hex string, it will be group name
    char buf[32] ;
    snprintf ( buf, sizeof buf, "%s:%04d", prefix, counter ) ;
    return buf;
  } else {
    return prefix;
  }
}

} // namespace O2OTranslator
