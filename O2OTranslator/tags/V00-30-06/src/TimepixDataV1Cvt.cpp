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
TimepixDataV1Cvt::TimepixDataV1Cvt ( const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const CvtOptions& cvtOptions,
    int schemaVersion )
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_dataCont()
  , m_imageCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
TimepixDataV1Cvt::~TimepixDataV1Cvt ()
{
}

// method called to create all necessary data containers
void
TimepixDataV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  m_dataCont = makeCont<DataCont>("data", group, true);
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
  Pds::Timepix::DataV2 tmp2(data.width(), data.height(), data.timestamp(), data.frameCounter(), data.lostRows());
  unsigned objSize = tmp2._sizeof();
  char* buf = new char[objSize];
  Pds::Timepix::DataV2* data2 = new (buf) Pds::Timepix::DataV2(data);

  uint32_t height = data2->height();
  uint32_t width = data2->width();

  // store the data
  H5Type tpdata(*data2);
  m_dataCont->append(tpdata);

  hdf5pp::Type type = H5Type::stored_data_type(height, width) ;
  if (not m_imageCont) {
    m_imageCont = makeCont<ImageCont>("image", group, true, type);
    if (n_miss) m_imageCont->resize(n_miss);
  }
  m_imageCont->append(*(uint16_t*)data2->data().data(), type);

  delete [] buf;
}

// fill containers for missing data
void
TimepixDataV1Cvt::fillMissing(hdf5pp::Group group,
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
