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
#include "hdf5pp/TypeTraits.h"

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
  : EvtDataTypeCvt<Pds::Camera::FrameV1>( typeGroupName, chunk_size, deflate )
  , m_imgType()
  , m_dataCont(0)
  , m_imageCont(0)
{
}

//--------------
// Destructor --
//--------------
CameraFrameV1Cvt::~CameraFrameV1Cvt ()
{
  delete m_dataCont ;
  delete m_imageCont ;
  delete m_dimFixFlagCont;
}

// method called to create all necessary data containers
void
CameraFrameV1Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // make container for data objects
  DataCont::factory_type dataContFactory ( "data", chunk_size, deflate, true ) ;
  m_dataCont = new DataCont ( dataContFactory ) ;

  // get the type for the image
  ImageCont::factory_type imgContFactory( "image", chunk_size, deflate, true ) ;
  m_imageCont = new ImageCont ( imgContFactory ) ;

  // separate dataset which indicates that image dimensions are correct
  DimFixFlagCont::factory_type dimFixFlagFactory ( "_dim_fix_flag_201103", 1, deflate ) ;
  m_dimFixFlagCont = new DimFixFlagCont(dimFixFlagFactory);
}

// typed conversion method
void
CameraFrameV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  if ( H5Type::xtcSize(data) != size ) {
    throw O2OXTCSizeException ( ERR_LOC, "Camera::FrameV1", H5Type::xtcSize(data), size ) ;
  }

  // store the data in the containers
  m_dataCont->container(group)->append ( H5Type(data) ) ;
  hdf5pp::Type imgType = H5Type::imageType ( data ) ;
  m_imageCont->container(group,imgType)->append ( *data.data(), imgType ) ;
  
  // do not store anything in the flag container, just create it
  hdf5pp::Type flagType = hdf5pp::TypeTraits<uint16_t>::stored_type();
  m_dimFixFlagCont->container(group,flagType);
}

/// method called when the driver closes a group in the file
void
CameraFrameV1Cvt::closeContainers( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_imageCont ) m_imageCont->closeGroup( group ) ;
  if ( m_dimFixFlagCont ) m_dimFixFlagCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
