//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FliFrameV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/FliFrameV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/fli/ConfigV1.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "FliFrameV1Cvt" ;
}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
FliFrameV1Cvt::FliFrameV1Cvt ( const std::string& typeGroupName,
                                           const ConfigObjectStore& configStore,
                                           hsize_t chunk_size,
                                           int deflate )
  : EvtDataTypeCvt<Pds::Fli::FrameV1>(typeGroupName)
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
FliFrameV1Cvt::~FliFrameV1Cvt ()
{
  delete m_frameCont ;
  delete m_frameDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
FliFrameV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTime& time )
{
  // find corresponding configuration object
  uint32_t height = 0;
  uint32_t width = 0;
  Pds::TypeId cfgTypeId1(Pds::TypeId::Id_FliConfig, 1);
  if (const Pds::Fli::ConfigV1* config = m_configStore.find<Pds::Fli::ConfigV1>(cfgTypeId1, src.top())) {
    uint32_t binX = config->binX();
    uint32_t binY = config->binY();
    height = (config->height() + binY - 1) / binY;
    width = (config->width() + binX - 1) / binX;
  } else {
    MsgLog ( logger, error, "FliFrameV1Cvt - no configuration object was defined" );
    return ;
  }

  // create all containers if running first time
  if ( not m_frameCont ) {

    // create container for frames
    CvtDataContFactoryDef<H5DataTypes::FliFrameV1> frContFactory( "frame", m_chunk_size, m_deflate, true ) ;
    m_frameCont = new FrameCont ( frContFactory ) ;

    // create container for frame data
    CvtDataContFactoryTyped<uint16_t> dataContFactory( "data", m_chunk_size, m_deflate, true ) ;
    m_frameDataCont = new FrameDataCont ( dataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  // store the data
  H5DataTypes::FliFrameV1 frame(data);
  m_frameCont->container(group)->append ( frame ) ;
  hdf5pp::Type type = H5DataTypes::FliFrameV1::stored_data_type(height, width) ;
  m_frameDataCont->container(group,type)->append ( *data.data(), type ) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
FliFrameV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_frameCont ) m_frameCont->closeGroup( group ) ;
  if ( m_frameDataCont ) m_frameDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
