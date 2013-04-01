//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FliFrameV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/FliFrameV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "H5DataTypes/FliFrameV1.h"
#include "H5DataTypes/AndorFrameV1.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/fli/ConfigV1.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "FliFrameV1Cvt" ;
}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
template <typename FrameType>
FliFrameV1Cvt<FrameType>::FliFrameV1Cvt ( const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    Pds::TypeId cfgTypeId,
    const CvtOptions& cvtOptions,
    int schemaVersion )
  : Super(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configStore(configStore)
  , m_cfgTypeId(cfgTypeId)
  , m_frameCont()
  , m_frameDataCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
template <typename FrameType>
FliFrameV1Cvt<FrameType>::~FliFrameV1Cvt ()
{
}

// method called to create all necessary data containers
template <typename FrameType>
void
FliFrameV1Cvt<FrameType>::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  m_frameCont = Super::template makeCont<FrameCont>("frame", group, true);
}

// typed conversion method
template <typename FrameType>
void
FliFrameV1Cvt<FrameType>::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  uint32_t height = 0;
  uint32_t width = 0;
  if (const Pds::Fli::ConfigV1* config = m_configStore.find<Pds::Fli::ConfigV1>(m_cfgTypeId, src.top())) {
    uint32_t binX = config->binX();
    uint32_t binY = config->binY();
    height = (config->height() + binY - 1) / binY;
    width = (config->width() + binX - 1) / binX;
  } else {
    MsgLog ( logger, error, "FliFrameV1Cvt - no configuration object was defined" );
    return ;
  }

  // store the data
  H5Type frame(data);
  m_frameCont->append ( frame ) ;
  hdf5pp::Type type = H5Type::stored_data_type(height, width) ;
  if (not m_frameDataCont) {
    m_frameDataCont = Super::template makeCont<FrameDataCont>("data", group, true, type);
    if (n_miss) m_frameDataCont->resize(n_miss);
  }
  m_frameDataCont->append(*data.data(), type);
}

// fill containers for missing data
template <typename FrameType>
void
FliFrameV1Cvt<FrameType>::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  m_frameCont->resize(m_frameCont->size() + 1);
  if (m_frameDataCont) {
    m_frameDataCont->resize(m_frameDataCont->size() + 1);
  } else {
    ++ n_miss;
  }
}


// explicitly instantiate for know types
template class FliFrameV1Cvt<H5DataTypes::AndorFrameV1>;
template class FliFrameV1Cvt<H5DataTypes::FliFrameV1>;

} // namespace O2OTranslator
