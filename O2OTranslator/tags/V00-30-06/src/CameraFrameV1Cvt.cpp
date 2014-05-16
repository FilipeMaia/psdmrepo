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
CameraFrameV1Cvt::CameraFrameV1Cvt (const hdf5pp::Group& group, const std::string& typeGroupName,
    const Pds::Src& src, const CvtOptions& cvtOptions, int schemaVersion )
  : EvtDataTypeCvt<Pds::Camera::FrameV1>( group, typeGroupName, src, cvtOptions, schemaVersion )
  , m_imgType()
  , m_dataCont()
  , m_imageCont()
  , m_dimFixFlagCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
CameraFrameV1Cvt::~CameraFrameV1Cvt ()
{
}

// method called to create all necessary data containers
void
CameraFrameV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // make container for data objects
  m_dataCont = makeCont<DataCont>("data", group, true);

  // separate dataset which indicates that image dimensions are correct
  m_dimFixFlagCont = makeCont<DimFixFlagCont>("_dim_fix_flag_201103", group, false);
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
  m_dataCont->append(H5Type(data));
  hdf5pp::Type imgType = H5Type::imageType(data);
  if (not m_imageCont) {
    m_imageCont = makeCont<ImageCont>("image", group, true, imgType);
    if (n_miss) m_imageCont->resize(n_miss);
  }
  m_imageCont->append(*data._int_pixel_data().data(), imgType);
}

// fill containers for missing data
void
CameraFrameV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  m_dataCont->resize(m_dataCont->size() + 1);
  if (m_imageCont) {
    m_imageCont->resize(m_imageCont->size() + 1);
  } else {
    ++ n_miss;
  }
}

} // namespace O2OTranslator
