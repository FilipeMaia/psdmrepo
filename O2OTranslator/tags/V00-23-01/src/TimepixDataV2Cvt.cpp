//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixDataV2Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/TimepixDataV2Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "TimepixDataV2Cvt" ;
}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
TimepixDataV2Cvt::TimepixDataV2Cvt ( const std::string& typeGroupName,
                                     hsize_t chunk_size,
                                     int deflate )
  : EvtDataTypeCvt<XtcType>(typeGroupName, chunk_size, deflate)
  , m_dataCont(0)
  , m_imageCont(0)
{
}

//--------------
// Destructor --
//--------------
TimepixDataV2Cvt::~TimepixDataV2Cvt ()
{
  delete m_dataCont ;
  delete m_imageCont ;
}

// method called to create all necessary data containers
void
TimepixDataV2Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  DataCont::factory_type dataContFactory( "data", chunk_size, deflate, true ) ;
  m_dataCont = new DataCont ( dataContFactory ) ;

  // create container for frame data
  ImageCont::factory_type imageContFactory( "image", chunk_size, deflate, true ) ;
  m_imageCont = new ImageCont ( imageContFactory ) ;
}

// typed conversion method
void
TimepixDataV2Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  uint32_t height = data.height();
  uint32_t width = data.width();

  // store the data
  H5Type tpdata(data);
  m_dataCont->container(group)->append(tpdata);
  hdf5pp::Type type = H5Type::stored_data_type(height, width) ;
  m_imageCont->container(group,type)->append(*(uint16_t*)data.data(), type);
}

/// method called when the driver closes a group in the file
void
TimepixDataV2Cvt::closeContainers( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_imageCont ) m_imageCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
