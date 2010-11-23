//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonFrameV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/PrincetonFrameV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/princeton/ConfigV1.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "PrincetonFrameV1Cvt" ;
}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
PrincetonFrameV1Cvt::PrincetonFrameV1Cvt ( const std::string& typeGroupName,
                                           const ConfigObjectStore& configStore,
                                           hsize_t chunk_size,
                                           int deflate )
  : EvtDataTypeCvt<Pds::Princeton::FrameV1>(typeGroupName)
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
PrincetonFrameV1Cvt::~PrincetonFrameV1Cvt ()
{
  delete m_frameCont ;
  delete m_frameDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
PrincetonFrameV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTime& time )
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_PrincetonConfig, 1);
  const Pds::Princeton::ConfigV1* config = m_configStore.find<Pds::Princeton::ConfigV1>(cfgTypeId, src.top());
  if ( not config ) {
    MsgLog ( logger, error, "PrincetonFrameV1Cvt - no configuration object was defined" );
    return ;
  }

  // create all containers if running first time
  if ( not m_frameCont ) {

    // create container for frames
    CvtDataContFactoryDef<H5DataTypes::PrincetonFrameV1> frContFactory( "frame", m_chunk_size, m_deflate ) ;
    m_frameCont = new FrameCont ( frContFactory ) ;

    // create container for frame data
    CvtDataContFactoryTyped<uint16_t> dataContFactory( "data", m_chunk_size, m_deflate ) ;
    m_frameDataCont = new FrameDataCont ( dataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  // store the data
  H5DataTypes::PrincetonFrameV1 frame(data);
  m_frameCont->container(group)->append ( frame ) ;
  hdf5pp::Type type = H5DataTypes::PrincetonFrameV1::stored_data_type ( *config ) ;
  m_frameDataCont->container(group,type)->append ( *data.data(), type ) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
PrincetonFrameV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_frameCont ) m_frameCont->closeGroup( group ) ;
  if ( m_frameDataCont ) m_frameDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
