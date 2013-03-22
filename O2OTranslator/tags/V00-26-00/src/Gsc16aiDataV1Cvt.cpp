//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Gsc16aiDataV1Cvt...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/Gsc16aiDataV1Cvt.h"

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
  const char logger[] = "Gsc16aiDataV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
Gsc16aiDataV1Cvt::Gsc16aiDataV1Cvt (const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CvtOptions& cvtOptions,
    int schemaVersion)
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configStore(configStore)
  , m_dataCont()
  , m_valueCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
Gsc16aiDataV1Cvt::~Gsc16aiDataV1Cvt ()
{
}

// method called to create all necessary data containers
void
Gsc16aiDataV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  m_dataCont = makeCont<DataCont>("timestamps", group, true);
}

// typed conversion method
void
Gsc16aiDataV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_Gsc16aiConfig, 1);
  const Pds::Gsc16ai::ConfigV1* config = m_configStore.find<Pds::Gsc16ai::ConfigV1>(cfgTypeId, src.top());
  MsgLog( logger, debug, "Gsc16aiDataV1Cvt: looking for config object "
      << src.top()
      << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
      << " version=" <<  cfgTypeId.version() ) ;
  if (not config) {
    MsgLog ( logger, error, "Gsc16aiDataV1Cvt - no configuration object was defined" );
    return ;
  }

  // make data objects
  H5Type timestampsData(data);

  // store the data
  m_dataCont->append(timestampsData) ;
  hdf5pp::Type type = H5Type::stored_data_type(*config);
  if (not m_valueCont) {
    m_valueCont = makeCont<ValueCont>("channelValue", group, true, type);
    if (n_miss) m_valueCont->resize(n_miss);
  }
  m_valueCont->append(data._channelValue[0], type);
}

// fill containers for missing data
void
Gsc16aiDataV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  m_dataCont->resize(m_dataCont->size() + 1);
  if (m_valueCont) {
    m_valueCont->resize(m_valueCont->size() + 1);
  } else {
    ++ n_miss;
  }
}

} // namespace O2OTranslator
