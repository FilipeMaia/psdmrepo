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
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

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
CameraFrameV1Cvt::CameraFrameV1Cvt (hdf5pp::Group group,
                                    hsize_t chunk_size,
                                    int deflate )
  : DataTypeCvt<H5DataTypes::CameraFrameV1::XtcType>()
  , m_group(group)
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
CameraFrameV1Cvt::~CameraFrameV1Cvt ()
{
  delete m_dataCont ;
  delete m_imageCont ;
  delete m_timeCont ;
}

// typed conversion method
void CameraFrameV1Cvt::typedConvert ( const Pds::Camera::FrameV1& data,
                                      const H5DataTypes::XtcClockTime& time )
{

  if ( not m_dataCont ) {

    // make container for data objects
    hsize_t chunk = std::max ( m_chunk_size / sizeof(H5DataTypes::CameraFrameV1), hsize_t(1) ) ;
    MsgLog( logger, debug, "chunk size for data: " << chunk ) ;
    m_dataCont = new DataCont ( "data", m_group, chunk, m_deflate ) ;

    // get the type for the image
    m_imgType = H5DataTypes::CameraFrameV1::imageType ( data ) ;
    chunk = std::max ( m_chunk_size / m_imgType.size(), hsize_t(1) ) ;
    MsgLog( logger, debug, "chunk size for image: " << chunk ) ;
    m_imageCont = new ImageCont ( "image", m_group, m_imgType, chunk, m_deflate ) ;
    m_imageCont->dataset().createAttr<const char*> ( "CLASS" ).store("IMAGE") ;

    // make container for time
    chunk = std::max ( m_chunk_size / sizeof(H5DataTypes::XtcClockTime), hsize_t(1) ) ;
    MsgLog( logger, debug, "chunk size for time: " << chunk ) ;
    m_timeCont = new XtcClockTimeCont ( "time", m_group, chunk, m_deflate ) ;

  }

  // store the data in the containers
  m_dataCont->append ( H5DataTypes::CameraFrameV1(data) ) ;
  m_imageCont->append ( *data.data(), m_imgType ) ;
  m_timeCont->append ( time ) ;
}


} // namespace O2OTranslator
