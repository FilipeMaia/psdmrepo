//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPimV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/BldDataPimV1Cvt.h"

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
  const char logger[] = "BldDataPimV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
BldDataPimV1Cvt::BldDataPimV1Cvt (const std::string& typeGroupName,
                                    hsize_t chunk_size,
                                    int deflate )
  : EvtDataTypeCvt<XtcType>( typeGroupName, chunk_size, deflate )
  , m_imgType()
  , m_dataCont(0)
  , m_imageCont(0)
{
}

//--------------
// Destructor --
//--------------
BldDataPimV1Cvt::~BldDataPimV1Cvt ()
{
  delete m_dataCont ;
  delete m_imageCont ;
}

// method called to create all necessary data containers
void
BldDataPimV1Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // make container for data objects
  DataCont::factory_type dataContFactory ( "data", chunk_size, deflate, true ) ;
  m_dataCont = new DataCont ( dataContFactory ) ;

  // get the type for the image
  ImageCont::factory_type imgContFactory( "image", chunk_size, deflate, true ) ;
  m_imageCont = new ImageCont ( imgContFactory ) ;
}

// typed conversion method
void
BldDataPimV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  if ( H5Type::xtcSize(data) != size ) {
    throw O2OXTCSizeException ( ERR_LOC, "BldDataPimV1", H5Type::xtcSize(data), size ) ;
  }

  // store the data in the containers
  m_dataCont->container(group)->append ( H5Type(data) ) ;
  hdf5pp::Type imgType = H5Type::imageType ( data ) ;
  m_imageCont->container(group,imgType)->append ( *data.frame.data(), imgType ) ;
}

/// method called when the driver closes a group in the file
void
BldDataPimV1Cvt::closeContainers( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_imageCont ) m_imageCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
