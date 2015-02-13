//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixSamplerElementV1Cvt...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/EpixSamplerElementV1Cvt.h"

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
  const char logger[] = "EpixSamplerElementV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
EpixSamplerElementV1Cvt::EpixSamplerElementV1Cvt (const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CvtOptions& cvtOptions,
    int schemaVersion)
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configStore(configStore)
  , m_dataCont()
  , m_frameCont()
  , m_temperatureCont()
{
}

//--------------
// Destructor --
//--------------
EpixSamplerElementV1Cvt::~EpixSamplerElementV1Cvt ()
{
}

// method called to create all necessary data containers
void
EpixSamplerElementV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  std::vector<int> shape = this->shape(src);
  if (not shape.empty()) {

    m_dataCont = makeCont<DataCont>("data", group, true) ;

    hdf5pp::Type type = H5Type::frame_data_type(shape[0], shape[1]);
    m_frameCont = makeCont<FrameCont>("frame", group, true, type);

    type = H5Type::temperature_data_type(shape[0]);
    m_temperatureCont = makeCont<TemperatureCont>("temperatures", group, true, type);
  }
}

// typed conversion method
void
EpixSamplerElementV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_EpixSamplerConfig, 1);
  const Pds::EpixSampler::ConfigV1* config = m_configStore.find<Pds::EpixSampler::ConfigV1>(cfgTypeId, src.top());
  MsgLog( logger, debug, "EpixSamplerElementV1Cvt: looking for config object "
      << src.top()
      << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
      << " version=" <<  cfgTypeId.version() ) ;
  if (not config) {
    MsgLog ( logger, error, "EpixSamplerElementV1Cvt - no configuration object was defined" );
    return;
  }

  ndarray<const uint16_t, 2> frame = data.frame(*config);
  ndarray<const uint16_t, 1> temperatures = data.temperatures(*config);

  // make data objects
  H5Type sData(data);

  // store the data
  m_dataCont->append(sData) ;

  hdf5pp::Type type = H5Type::frame_data_type(config->numberOfChannels(), config->samplesPerChannel());
  m_frameCont->append(*frame.data(), type);

  type = H5Type::temperature_data_type(config->numberOfChannels());
  m_temperatureCont->append(*temperatures.data(), type);
}

// fill containers for missing data
void
EpixSamplerElementV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  if (m_dataCont) {
    m_dataCont->resize(m_dataCont->size() + 1);
    m_frameCont->resize(m_frameCont->size() + 1);
    m_temperatureCont->resize(m_temperatureCont->size() + 1);
  }
}

// finds config object and gets number of samples from it, returns -1 if config object is not there
std::vector<int>
EpixSamplerElementV1Cvt::shape(const O2OXtcSrc& src)
{
  std::vector<int> res;

  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_EpixSamplerConfig, 1);
  const Pds::EpixSampler::ConfigV1* config = m_configStore.find<Pds::EpixSampler::ConfigV1>(cfgTypeId, src.top());
  MsgLog( logger, debug, "EpixSamplerElementV1Cvt: looking for config object "
      << src.top()
      << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
      << " version=" <<  cfgTypeId.version() ) ;
  if (not config) {
    MsgLog ( logger, error, "EpixSamplerElementV1Cvt - no configuration object was defined" );
    return res;
  }
  
  res.push_back(config->numberOfChannels());
  res.push_back(config->samplesPerChannel());

  return res;
}

} // namespace O2OTranslator
