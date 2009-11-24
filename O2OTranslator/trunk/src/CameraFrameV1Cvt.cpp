//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraFrameV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CameraFrameV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "CameraFrameV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
CameraFrameV1Cvt::CameraFrameV1Cvt (const std::string& typeGroupName,
                                    hsize_t chunk_size,
                                    int deflate )
  : EvtDataTypeCvt<Pds::Camera::FrameV1>( typeGroupName )
  , m_chunk_size( chunk_size )
  , m_deflate( deflate )
  , m_imgType()
  , m_dataCont(0)
  , m_imageCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
CameraFrameV1Cvt::~CameraFrameV1Cvt ()
{
  delete m_dataCont ;
  delete m_imageCont ;
  delete m_timeCont ;
}

// typed conversion method
void
CameraFrameV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTime& time )
{
  if ( sizeof data + data.data_size() != size ) {
    throw O2OXTCSizeException ( "Camera::FrameV1", sizeof data + data.data_size(), size ) ;
  }

  if ( not m_dataCont ) {

    // make container for data objects
    CvtDataContFactoryDef<H5DataTypes::CameraFrameV1> dataContFactory ( "data", m_chunk_size, m_deflate ) ;
    m_dataCont = new DataCont ( dataContFactory ) ;

    // get the type for the image
    CvtDataContFactoryTyped<const unsigned char> imgContFactory( "image", m_chunk_size, m_deflate ) ;
    m_imageCont = new ImageCont ( imgContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  // store the data in the containers
  m_dataCont->container(group)->append ( H5DataTypes::CameraFrameV1(data) ) ;
  hdf5pp::Type imgType = H5DataTypes::CameraFrameV1::imageType ( data ) ;
  m_imageCont->container(group,imgType)->append ( *data.data(), imgType ) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
CameraFrameV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_imageCont ) m_imageCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}


} // namespace O2OTranslator
