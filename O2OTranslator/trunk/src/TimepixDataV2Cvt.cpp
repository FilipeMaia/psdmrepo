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
TimepixDataV2Cvt::TimepixDataV2Cvt ( const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const CvtOptions& cvtOptions )
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions)
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
TimepixDataV2Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  m_dataCont = makeCont<DataCont>("data", group, true);
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
  m_dataCont->append(tpdata);

  hdf5pp::Type type = H5Type::stored_data_type(height, width) ;
  if (not m_imageCont) m_imageCont = makeCont<ImageCont>("image", group, true, type);
  m_imageCont->append(*(uint16_t*)data.data(), type);
}

} // namespace O2OTranslator
