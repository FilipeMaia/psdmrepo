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
#include "H5DataTypes/FliFrameV1.h"
#include "H5DataTypes/AndorFrameV1.h"
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
template <typename FrameType>
FliFrameV1Cvt<FrameType>::FliFrameV1Cvt ( const std::string& typeGroupName,
                                           const ConfigObjectStore& configStore,
                                           Pds::TypeId cfgTypeId,
                                           hsize_t chunk_size,
                                           int deflate )
  : EvtDataTypeCvt<XtcType>(typeGroupName, chunk_size, deflate)
  , m_configStore(configStore)
  , m_cfgTypeId(cfgTypeId)
  , m_frameCont(0)
  , m_frameDataCont(0)
{
}

//--------------
// Destructor --
//--------------
template <typename FrameType>
FliFrameV1Cvt<FrameType>::~FliFrameV1Cvt ()
{
  delete m_frameCont ;
  delete m_frameDataCont ;
}

// method called to create all necessary data containers
template <typename FrameType>
void
FliFrameV1Cvt<FrameType>::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  typename FrameCont::factory_type frContFactory( "frame", chunk_size, deflate, true ) ;
  m_frameCont = new FrameCont ( frContFactory ) ;

  // create container for frame data
  FrameDataCont::factory_type dataContFactory( "data", chunk_size, deflate, true ) ;
  m_frameDataCont = new FrameDataCont ( dataContFactory ) ;
}

// typed conversion method
template <typename FrameType>
void
FliFrameV1Cvt<FrameType>::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  uint32_t height = 0;
  uint32_t width = 0;
  if (const Pds::Fli::ConfigV1* config = m_configStore.find<Pds::Fli::ConfigV1>(m_cfgTypeId, src.top())) {
    uint32_t binX = config->binX();
    uint32_t binY = config->binY();
    height = (config->height() + binY - 1) / binY;
    width = (config->width() + binX - 1) / binX;
  } else {
    MsgLog ( logger, error, "FliFrameV1Cvt - no configuration object was defined" );
    return ;
  }

  // store the data
  H5Type frame(data);
  m_frameCont->container(group)->append ( frame ) ;
  hdf5pp::Type type = H5Type::stored_data_type(height, width) ;
  m_frameDataCont->container(group,type)->append ( *data.data(), type ) ;
}

/// method called when the driver closes a group in the file
template <typename FrameType>
void
FliFrameV1Cvt<FrameType>::closeContainers( hdf5pp::Group group )
{
  if ( m_frameCont ) m_frameCont->closeGroup( group ) ;
  if ( m_frameDataCont ) m_frameDataCont->closeGroup( group ) ;
}

// explicitly instantiate for know types
template class FliFrameV1Cvt<H5DataTypes::AndorFrameV1>;
template class FliFrameV1Cvt<H5DataTypes::FliFrameV1>;

} // namespace O2OTranslator
