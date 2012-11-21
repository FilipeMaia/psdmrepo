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
  : EvtDataTypeCvt<Pds::Timepix::DataV2>(typeGroupName)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_dataCont(0)
  , m_imageCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
TimepixDataV2Cvt::~TimepixDataV2Cvt ()
{
  delete m_dataCont ;
  delete m_imageCont ;
  delete m_timeCont ;
}

// typed conversion method
void
TimepixDataV2Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTimeStamp& time )
{
  // create all containers if running first time
  if ( not m_dataCont ) {

    // create container for frames
    CvtDataContFactoryDef<H5DataTypes::TimepixDataV2> dataContFactory( "data", m_chunk_size, m_deflate, true ) ;
    m_dataCont = new DataCont ( dataContFactory ) ;

    // create container for frame data
    CvtDataContFactoryTyped<uint16_t> imageContFactory( "image", m_chunk_size, m_deflate, true ) ;
    m_imageCont = new ImageCont ( imageContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTimeStamp> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  uint32_t height = data.height();
  uint32_t width = data.width();

  // store the data
  H5DataTypes::TimepixDataV2 tpdata(data);
  m_dataCont->container(group)->append(tpdata);
  hdf5pp::Type type = H5DataTypes::TimepixDataV2::stored_data_type(height, width) ;
  m_imageCont->container(group,type)->append(*(uint16_t*)data.data(), type);
  m_timeCont->container(group)->append(time);
}

/// method called when the driver closes a group in the file
void
TimepixDataV2Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_imageCont ) m_imageCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
