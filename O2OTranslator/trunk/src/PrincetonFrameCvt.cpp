//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonFrameCvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/PrincetonFrameCvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "H5DataTypes/PrincetonFrameV1.h"
#include "H5DataTypes/PrincetonFrameV2.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/psddl/princeton.ddl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "PrincetonFrameCvt" ;
}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
template <typename H5DataType>
PrincetonFrameCvt<H5DataType>::PrincetonFrameCvt ( const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CvtOptions& cvtOptions,
    int schemaVersion )
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configStore(configStore)
  , m_frameCont()
  , m_frameDataCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
template <typename H5DataType>
PrincetonFrameCvt<H5DataType>::~PrincetonFrameCvt ()
{
}

// method called to create all necessary data containers
template <typename H5DataType>
void
PrincetonFrameCvt<H5DataType>::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  m_frameCont = this->template makeCont<FrameCont>("frame", group, true);
}

// typed conversion method
template <typename H5DataType>
void
PrincetonFrameCvt<H5DataType>::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::TypeId::Type tid = Pds::TypeId::Id_PrincetonConfig;
  if (const Pds::Princeton::ConfigV1* config = m_configStore.find<Pds::Princeton::ConfigV1>(Pds::TypeId(tid, 1), src.top())) {
    fillContainers(group, data, size, typeId, src, *config);
  } else if (const Pds::Princeton::ConfigV2* config = m_configStore.find<Pds::Princeton::ConfigV2>(Pds::TypeId(tid, 2), src.top())) {
    fillContainers(group, data, size, typeId, src, *config);
  } else if (const Pds::Princeton::ConfigV3* config = m_configStore.find<Pds::Princeton::ConfigV3>(Pds::TypeId(tid, 3), src.top())) {
    fillContainers(group, data, size, typeId, src, *config);
  } else if (const Pds::Princeton::ConfigV4* config = m_configStore.find<Pds::Princeton::ConfigV4>(Pds::TypeId(tid, 4), src.top())) {
    fillContainers(group, data, size, typeId, src, *config);
  } else if (const Pds::Princeton::ConfigV5* config = m_configStore.find<Pds::Princeton::ConfigV5>(Pds::TypeId(tid, 5), src.top())) {
    fillContainers(group, data, size, typeId, src, *config);
  } else {
    MsgLog ( logger, error, "PrincetonFrameCvt - no configuration object was defined" );
  }
}

// typed conversion method templated on Configuration type
template <typename H5DataType>
template <typename Config>
void
PrincetonFrameCvt<H5DataType>::fillContainers(hdf5pp::Group group,
                                  const XtcType& data,
                                  size_t size,
                                  const Pds::TypeId& typeId,
                                  const O2OXtcSrc& src,
                                  const Config& cfg)
{
  uint32_t binX = cfg.binX();
  uint32_t binY = cfg.binY();
  uint32_t height = (cfg.height() + binY - 1) / binY;
  uint32_t width = (cfg.width() + binX - 1) / binX;

  // store the data
  H5Type frame(data);
  m_frameCont->append(frame);

  hdf5pp::Type type = H5Type::stored_data_type(height, width) ;
  if (not m_frameDataCont) {
    m_frameDataCont = this->template makeCont<FrameDataCont>("data", group, true, type) ;
    if (n_miss) m_frameDataCont->resize(n_miss);
  }
  m_frameDataCont->append(*data.data(cfg).data(), type);
}

// fill containers for missing data
template <typename H5DataType>
void
PrincetonFrameCvt<H5DataType>::fillMissing(hdf5pp::Group group,
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

template class PrincetonFrameCvt<H5DataTypes::PrincetonFrameV1>;
template class PrincetonFrameCvt<H5DataTypes::PrincetonFrameV2>;

} // namespace O2OTranslator
