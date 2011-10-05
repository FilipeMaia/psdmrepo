//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDFrameV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/PnCCDFrameV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/pnCCD/ConfigV1.hh"
#include "pdsdata/pnCCD/ConfigV2.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "PnCCDFrameV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
PnCCDFrameV1Cvt::PnCCDFrameV1Cvt ( const std::string& typeGroupName,
                                   const ConfigObjectStore& configStore,
                                   hsize_t chunk_size,
                                   int deflate )
  : EvtDataTypeCvt<Pds::PNCCD::FrameV1>(typeGroupName)
  , m_configStore(configStore)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_frameCont(0)
  , m_frameDataCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
PnCCDFrameV1Cvt::~PnCCDFrameV1Cvt ()
{
  delete m_frameCont ;
  delete m_frameDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
PnCCDFrameV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTime& time )
{
  // find corresponding configuration object
  const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>(src.top());
  Pds::PNCCD::ConfigV1 config;
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_pnCCDconfig, 1);
  const Pds::PNCCD::ConfigV1* config1 = m_configStore.find<Pds::PNCCD::ConfigV1>(cfgTypeId, src.top());
  MsgLog( logger, debug, "PnCCDFrameV1Cvt: looking for config object "
      << Pds::DetInfo::name(info)
      << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
      << " version=" <<  cfgTypeId.version() ) ;
  if ( config1 ) {
    config = Pds::PNCCD::ConfigV1(*config1);
  } else {
    Pds::TypeId cfgTypeId(Pds::TypeId::Id_pnCCDconfig, 2);
    const Pds::PNCCD::ConfigV2* config2 = m_configStore.find<Pds::PNCCD::ConfigV2>(cfgTypeId, src.top());
    MsgLog( logger, debug, "PnCCDFrameV1Cvt: looking for config object "
        << Pds::DetInfo::name(info)
        << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
        << " version=" <<  cfgTypeId.version() ) ;
    if (not config2) {
      MsgLog ( logger, error, "PnCCDFrameV1Cvt - no configuration object was defined" );
      return ;
    }
    config = Pds::PNCCD::ConfigV1(config2->numLinks(), config2->payloadSizePerLink());
  }

  // create all containers if running first time
  if ( not m_frameCont ) {

    // create container for frames
    CvtDataContFactoryTyped<H5DataTypes::PnCCDFrameV1> frContFactory( "frame", m_chunk_size, m_deflate, true ) ;
    m_frameCont = new FrameCont ( frContFactory ) ;

    // create container for frame data
    CvtDataContFactoryTyped<uint16_t> dataContFactory( "data", m_chunk_size, m_deflate, true ) ;
    m_frameDataCont = new FrameDataCont ( dataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }


  // get few constants
  const uint32_t numLinks = config.numLinks() ;
  const unsigned sizeofData = data.sizeofData(config) ;

  // make data arrays
  H5DataTypes::PnCCDFrameV1 frame[numLinks] ;
  uint16_t frameData[numLinks][sizeofData] ;

  // move the data
  const XtcType* dd = &data ;
  for ( unsigned link = 0 ; link != numLinks ; ++ link ) {

    // copy frame info
    frame[link] = H5DataTypes::PnCCDFrameV1(*dd) ;

    // copy data
    const uint16_t* ddata = dd->data() ;
    std::copy( ddata, ddata+sizeofData, (uint16_t*)frameData[link] ) ;

    // move to next frame
    dd = dd->next(config) ;
  }

  // store the data
  hdf5pp::Type type = H5DataTypes::PnCCDFrameV1::stored_type ( config ) ;
  m_frameCont->container(group,type)->append ( frame[0], type ) ;
  type = H5DataTypes::PnCCDFrameV1::stored_data_type ( config ) ;
  m_frameDataCont->container(group,type)->append ( frameData[0][0], type ) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
PnCCDFrameV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_frameCont ) m_frameCont->closeGroup( group ) ;
  if ( m_frameDataCont ) m_frameDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
