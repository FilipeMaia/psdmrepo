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
#include "pdsdata/princeton/ConfigV2.hh"
#include "pdsdata/princeton/ConfigV3.hh"
#include "pdsdata/princeton/ConfigV4.hh"

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
  : EvtDataTypeCvt<XtcType>(typeGroupName, chunk_size, deflate)
  , m_configStore(configStore)
  , m_frameCont(0)
  , m_frameDataCont(0)
{
}

//--------------
// Destructor --
//--------------
PrincetonFrameV1Cvt::~PrincetonFrameV1Cvt ()
{
  delete m_frameCont ;
  delete m_frameDataCont ;
}

// method called to create all necessary data containers
void
PrincetonFrameV1Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  FrameCont::factory_type frContFactory( "frame", chunk_size, deflate, true ) ;
  m_frameCont = new FrameCont ( frContFactory ) ;

  // create container for frame data
  FrameDataCont::factory_type dataContFactory( "data", chunk_size, deflate, true ) ;
  m_frameDataCont = new FrameDataCont ( dataContFactory ) ;
}

// typed conversion method
void
PrincetonFrameV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  uint32_t height = 0;
  uint32_t width = 0;
  Pds::TypeId cfgTypeId1(Pds::TypeId::Id_PrincetonConfig, 1);
  Pds::TypeId cfgTypeId2(Pds::TypeId::Id_PrincetonConfig, 2);
  Pds::TypeId cfgTypeId3(Pds::TypeId::Id_PrincetonConfig, 3);
  Pds::TypeId cfgTypeId4(Pds::TypeId::Id_PrincetonConfig, 4);
  if (const Pds::Princeton::ConfigV1* config = m_configStore.find<Pds::Princeton::ConfigV1>(cfgTypeId1, src.top())) {
    uint32_t binX = config->binX();
    uint32_t binY = config->binY();
    height = (config->height() + binY - 1) / binY;
    width = (config->width() + binX - 1) / binX;
  } else if (const Pds::Princeton::ConfigV2* config = m_configStore.find<Pds::Princeton::ConfigV2>(cfgTypeId2, src.top())) {
    uint32_t binX = config->binX();
    uint32_t binY = config->binY();
    height = (config->height() + binY - 1) / binY;
    width = (config->width() + binX - 1) / binX;
  } else if (const Pds::Princeton::ConfigV3* config = m_configStore.find<Pds::Princeton::ConfigV3>(cfgTypeId3, src.top())) {
    uint32_t binX = config->binX();
    uint32_t binY = config->binY();
    height = (config->height() + binY - 1) / binY;
    width = (config->width() + binX - 1) / binX;
  } else if (const Pds::Princeton::ConfigV4* config = m_configStore.find<Pds::Princeton::ConfigV4>(cfgTypeId4, src.top())) {
    uint32_t binX = config->binX();
    uint32_t binY = config->binY();
    height = (config->height() + binY - 1) / binY;
    width = (config->width() + binX - 1) / binX;
  } else {
    MsgLog ( logger, error, "PrincetonFrameV1Cvt - no configuration object was defined" );
    return ;
  }

  // store the data
  H5Type frame(data);
  m_frameCont->container(group)->append ( frame ) ;
  hdf5pp::Type type = H5Type::stored_data_type(height, width) ;
  m_frameDataCont->container(group,type)->append ( *data.data(), type ) ;
}

/// method called when the driver closes a group in the file
void
PrincetonFrameV1Cvt::closeContainers( hdf5pp::Group group )
{
  if ( m_frameCont ) m_frameCont->closeGroup( group ) ;
  if ( m_frameDataCont ) m_frameDataCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
