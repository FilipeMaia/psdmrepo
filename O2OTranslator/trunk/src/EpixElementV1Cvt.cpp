//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixElementV1Cvt...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/EpixElementV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "EpixElementV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
EpixElementV1Cvt::EpixElementV1Cvt (const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CvtOptions& cvtOptions,
    int schemaVersion)
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configStore(configStore)
  , m_dataCont()
  , m_frameCont()
  , m_excludedRowsCont()
  , m_temperatureCont()
{
}

//--------------
// Destructor --
//--------------
EpixElementV1Cvt::~EpixElementV1Cvt ()
{
}

// method called to create all necessary data containers
void
EpixElementV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  std::vector<int> shape = this->shape(src);
  if (not shape.empty()) {

    m_dataCont = makeCont<DataCont>("data", group, true) ;

    hdf5pp::Type type = H5Type::frame_data_type(shape[1], shape[2]);
    m_frameCont = makeCont<FrameCont>("frame", group, true, type);

    if (shape[3]) {
      // HDF5 does not like zero-size dimensions
      type = H5Type::excludedRows_data_type(shape[3], shape[2]);
      m_excludedRowsCont = makeCont<ExcludedRowsCont>("excludedRows", group, true, type);
    }

    type = H5Type::temperature_data_type(shape[0]);
    m_temperatureCont = makeCont<TemperatureCont>("temperatures", group, true, type);
  }
}

// typed conversion method
void
EpixElementV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_EpixConfig, 1);
  const Pds::Epix::ConfigV1* config = m_configStore.find<Pds::Epix::ConfigV1>(cfgTypeId, src.top());
  MsgLog( logger, debug, "EpixElementV1Cvt: looking for config object "
      << src.top()
      << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
      << " version=" <<  cfgTypeId.version() ) ;
  if (not config) {
    MsgLog ( logger, error, "EpixElementV1Cvt - no configuration object was defined" );
    return;
  }

  // make data objects
  H5Type sData(data, *config);
  m_dataCont->append(sData) ;

  ndarray<const uint16_t, 2> frame = data.frame(*config);
  hdf5pp::Type type = H5Type::frame_data_type(config->numberOfRows(), config->numberOfColumns());
  m_frameCont->append(*frame.data(), type);

  if (m_excludedRowsCont) {
    ndarray<const uint16_t, 2> excludedRows = data.excludedRows(*config);
    type = H5Type::excludedRows_data_type(config->lastRowExclusions(), config->numberOfColumns());
    m_excludedRowsCont->append(*excludedRows.data(), type);
  }

  ndarray<const uint16_t, 1> temperatures = data.temperatures(*config);
  type = H5Type::temperature_data_type(config->numberOfAsics());
  m_temperatureCont->append(*temperatures.data(), type);
}

// fill containers for missing data
void
EpixElementV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  if (m_dataCont) {
    m_dataCont->resize(m_dataCont->size() + 1);
    m_frameCont->resize(m_frameCont->size() + 1);
    if (m_excludedRowsCont) m_excludedRowsCont->resize(m_frameCont->size() + 1);
    m_temperatureCont->resize(m_temperatureCont->size() + 1);
  }
}

// finds config object and gets various dimensions from it
std::vector<int>
EpixElementV1Cvt::shape(const O2OXtcSrc& src)
{
  std::vector<int> res;

  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_EpixConfig, 1);
  const Pds::Epix::ConfigV1* config = m_configStore.find<Pds::Epix::ConfigV1>(cfgTypeId, src.top());
  MsgLog( logger, debug, "EpixElementV1Cvt: looking for config object "
      << src.top()
      << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
      << " version=" <<  cfgTypeId.version() ) ;
  if (not config) {
    MsgLog ( logger, error, "EpixElementV1Cvt - no configuration object was defined" );
    return res;
  }
  
  res.push_back(config->numberOfAsics());
  res.push_back(config->numberOfRows());
  res.push_back(config->numberOfColumns());
  res.push_back(config->lastRowExclusions());

  return res;
}

} // namespace O2OTranslator
