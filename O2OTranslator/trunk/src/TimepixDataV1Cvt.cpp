//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixDataV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/TimepixDataV1Cvt.h"

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
  const char logger[] = "TimepixDataV1Cvt" ;
}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
TimepixDataV1Cvt::TimepixDataV1Cvt ( const std::string& typeGroupName,
                                     hsize_t chunk_size,
                                     int deflate )
  : EvtDataTypeCvt<Pds::Timepix::DataV1>(typeGroupName, chunk_size, deflate)
  , m_dataCont(0)
  , m_imageCont(0)
{
}

//--------------
// Destructor --
//--------------
TimepixDataV1Cvt::~TimepixDataV1Cvt ()
{
  delete m_dataCont ;
  delete m_imageCont ;
}

// method called to create all necessary data containers
void
TimepixDataV1Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  CvtDataContFactoryDef<H5DataTypes::TimepixDataV2> dataContFactory( "data", chunk_size, deflate, true ) ;
  m_dataCont = new DataCont ( dataContFactory ) ;

  // create container for frame data
  CvtDataContFactoryTyped<uint16_t> imageContFactory( "image", chunk_size, deflate, true ) ;
  m_imageCont = new ImageCont ( imageContFactory ) ;
}

// typed conversion method
void
TimepixDataV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // DataV1 images come with some strange shuffling. To un-shuffle it we transform it
  // into DataV2 which knows how to do it.
  unsigned objSize = sizeof(Pds::Timepix::DataV2) + data.data_size();
  char* buf = new char[objSize];
  Pds::Timepix::DataV2* data2 = new (buf) Pds::Timepix::DataV2(data);

  uint32_t height = data2->height();
  uint32_t width = data2->width();

  // store the data
  H5DataTypes::TimepixDataV2 tpdata(*data2);
  m_dataCont->container(group)->append(tpdata);
  hdf5pp::Type type = H5DataTypes::TimepixDataV2::stored_data_type(height, width) ;
  m_imageCont->container(group,type)->append(*(uint16_t*)data2->data(), type);

  delete [] buf;
}

/// method called when the driver closes a group in the file
void
TimepixDataV1Cvt::closeContainers( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_imageCont ) m_imageCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
