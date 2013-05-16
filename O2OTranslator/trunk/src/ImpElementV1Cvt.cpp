//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImpElementV1Cvt...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/ImpElementV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/gsc16ai/ConfigV1.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "ImpElementV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
ImpElementV1Cvt::ImpElementV1Cvt (const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CvtOptions& cvtOptions,
    int schemaVersion)
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configStore(configStore)
  , m_dataCont()
  , m_samplesCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
ImpElementV1Cvt::~ImpElementV1Cvt ()
{
}

// method called to create all necessary data containers
void
ImpElementV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  int nsmpl = nSamples(src);
  if (nsmpl >= 0) {
    hdf5pp::Type type = H5Type::stored_data_type(nsmpl);
    m_samplesCont = makeCont<SamplesCont>("samples", group, true, type);
    m_dataCont = makeCont<DataCont>("data", group, true);
  }
}

// typed conversion method
void
ImpElementV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  const int nsmpl = nSamples(src);
  if (nsmpl < 0) return;

  // make data objects
  H5Type sData(data);

  // have to copy it
  H5SampleType samples[nsmpl];
  for (int i = 0; i != nsmpl; ++ i) {
    samples[i] = H5SampleType(const_cast<XtcType&>(data).getSample(i));
  }
  
  // store the data
  m_dataCont->append(sData) ;
  hdf5pp::Type type = H5Type::stored_data_type(nsmpl);
  m_samplesCont->append(samples[0], type);
}

// fill containers for missing data
void
ImpElementV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  m_dataCont->resize(m_dataCont->size() + 1);
  m_samplesCont->resize(m_samplesCont->size() + 1);
}

// finds config object and gets number of samples from it, returns -1 if config object is not there
int 
ImpElementV1Cvt::nSamples(const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_ImpConfig, 1);
  const Pds::Imp::ConfigV1* config = m_configStore.find<Pds::Imp::ConfigV1>(cfgTypeId, src.top());
  MsgLog( logger, debug, "ImpElementV1Cvt: looking for config object "
      << src.top()
      << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
      << " version=" <<  cfgTypeId.version() ) ;
  if (not config) {
    MsgLog ( logger, error, "ImpElementV1Cvt - no configuration object was defined" );
    return -1;
  }
  
  return config->get(Pds::Imp::ConfigV1::NumberOfSamples);
}

} // namespace O2OTranslator
