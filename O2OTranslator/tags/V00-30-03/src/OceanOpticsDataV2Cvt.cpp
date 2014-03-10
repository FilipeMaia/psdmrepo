//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsDataV2Cvt...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/OceanOpticsDataV2Cvt.h"

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
  const char logger[] = "OceanOpticsDataV2Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
OceanOpticsDataV2Cvt::OceanOpticsDataV2Cvt(const hdf5pp::Group& group,
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
OceanOpticsDataV2Cvt::~OceanOpticsDataV2Cvt ()
{
}

// method called to create all necessary data containers
void
OceanOpticsDataV2Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for objects
  m_objCont = makeCont<ObjectCont>("data", group, true) ;
}

// typed conversion method
void
OceanOpticsDataV2Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId2(Pds::TypeId::Id_OceanOpticsConfig, 2);
  const Pds::OceanOptics::ConfigV2* config = m_configStore.find<Pds::OceanOptics::ConfigV2>(cfgTypeId2, src.top());
  if (not config) {
    MsgLog ( logger, error, "OceanOpticsDataV2Cvt - no configuration object was defined" );
    return ;
  }

  // make corrected data
  float corrData[Pds::OceanOptics::DataV2::iNumPixels];
  for (int i = 0; i != Pds::OceanOptics::DataV2::iNumPixels; ++ i) {
    corrData[i] = data.nonlinerCorrected(*config, i);
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
OceanOpticsDataV2Cvt::fillMissing(hdf5pp::Group group,
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
