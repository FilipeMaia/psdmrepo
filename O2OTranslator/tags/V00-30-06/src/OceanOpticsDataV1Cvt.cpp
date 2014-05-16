//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsDataV1Cvt...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/OceanOpticsDataV1Cvt.h"

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
  const char logger[] = "OceanOpticsDataV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
OceanOpticsDataV1Cvt::OceanOpticsDataV1Cvt(const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CvtOptions& cvtOptions,
    int schemaVersion)
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configStore(configStore)
  , m_objCont()
  , m_dataCont()
  , m_corrDataCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
OceanOpticsDataV1Cvt::~OceanOpticsDataV1Cvt ()
{
}

// method called to create all necessary data containers
void
OceanOpticsDataV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for objects
  m_objCont = makeCont<ObjectCont>("data", group, true) ;
}

// typed conversion method
void
OceanOpticsDataV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // make corrected data
  float corrData[Pds::OceanOptics::DataV1::iNumPixels];

  // find corresponding configuration object
  Pds::TypeId cfgTypeId1(Pds::TypeId::Id_OceanOpticsConfig, 1);
  Pds::TypeId cfgTypeId2(Pds::TypeId::Id_OceanOpticsConfig, 2);
  if (const Pds::OceanOptics::ConfigV1* config = m_configStore.find<Pds::OceanOptics::ConfigV1>(cfgTypeId1, src.top())) {
    for (int i = 0; i != Pds::OceanOptics::DataV1::iNumPixels; ++ i) {
      corrData[i] = data.nonlinerCorrected(*config, i);
    }
  } else if (const Pds::OceanOptics::ConfigV2* config = m_configStore.find<Pds::OceanOptics::ConfigV2>(cfgTypeId2, src.top())) {
    for (int i = 0; i != Pds::OceanOptics::DataV1::iNumPixels; ++ i) {
      corrData[i] = data.nonlinerCorrected(*config, i);
    }
  } else {
    MsgLog ( logger, error, "OceanOpticsDataV1Cvt - no configuration object was defined" );
    return ;
  }

  // store the data
  H5Type obj(data);
  m_objCont->append(obj) ;

  hdf5pp::Type type = H5Type::stored_data_type() ;
  if (not m_dataCont) {
    m_dataCont = makeCont<DataCont>("spectra", group, true, type) ;
    if (n_miss) m_dataCont->resize(n_miss);
  }
  m_dataCont->append(*data.data().data(), type);

  type = H5Type::stored_corrected_data_type() ;
  if (not m_corrDataCont) {
    m_corrDataCont = makeCont<CorrectedDataCont>("corrSpectra", group, true, type);
    if (n_miss) m_corrDataCont->resize(n_miss);
  }
  m_corrDataCont->append(*corrData, type);
}

// fill containers for missing data
void
OceanOpticsDataV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  m_objCont->resize(m_objCont->size() + 1);
  if (m_dataCont) {
    m_dataCont->resize(m_dataCont->size() + 1);
    m_corrDataCont->resize(m_corrDataCont->size() + 1);
  } else {
    ++ n_miss;
  }
}

} // namespace O2OTranslator
