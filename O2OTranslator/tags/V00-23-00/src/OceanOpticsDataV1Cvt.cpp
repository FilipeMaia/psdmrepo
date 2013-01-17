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
OceanOpticsDataV1Cvt::OceanOpticsDataV1Cvt(const std::string& typeGroupName,
                                           const ConfigObjectStore& configStore,
                                           hsize_t chunk_size,
                                           int deflate)
  : EvtDataTypeCvt<XtcType>(typeGroupName, chunk_size, deflate)
  , m_configStore(configStore)
  , m_objCont(0)
  , m_dataCont(0)
  , m_corrDataCont(0)
{
}

//--------------
// Destructor --
//--------------
OceanOpticsDataV1Cvt::~OceanOpticsDataV1Cvt ()
{
  delete m_objCont ;
  delete m_dataCont ;
  delete m_corrDataCont ;
}

// method called to create all necessary data containers
void
OceanOpticsDataV1Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for objects
  ObjectCont::factory_type objContFactory( "data", chunk_size, deflate, true ) ;
  m_objCont = new ObjectCont ( objContFactory ) ;

  // create container for data
  DataCont::factory_type dataContFactory( "spectra", chunk_size, deflate, true ) ;
  m_dataCont = new DataCont ( dataContFactory ) ;

  // create container for corrected data
  CorrectedDataCont::factory_type corrDataContFactory( "corrSpectra", chunk_size, deflate, true ) ;
  m_corrDataCont = new CorrectedDataCont ( corrDataContFactory ) ;
}

// typed conversion method
void
OceanOpticsDataV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId1(Pds::TypeId::Id_OceanOpticsConfig, 1);
  const Pds::OceanOptics::ConfigV1* config = m_configStore.find<Pds::OceanOptics::ConfigV1>(cfgTypeId1, src.top());
  if (not config) {
    MsgLog ( logger, error, "OceanOpticsDataV1Cvt - no configuration object was defined" );
    return ;
  }

  // make corrected data
  float corrData[Pds::OceanOptics::DataV1::iNumPixels];
  for (int i = 0; i != Pds::OceanOptics::DataV1::iNumPixels; ++ i) {
    corrData[i] = data.nonlinerCorrected(*config, i);
  }

  // store the data
  H5Type obj(data);
  m_objCont->container(group)->append(obj) ;
  hdf5pp::Type type = H5Type::stored_data_type() ;
  m_dataCont->container(group,type)->append(*data.data(), type);
  type = H5Type::stored_corrected_data_type() ;
  m_corrDataCont->container(group,type)->append(*corrData, type);
}

/// method called when the driver closes a group in the file
void
OceanOpticsDataV1Cvt::closeContainers( hdf5pp::Group group )
{
  if ( m_objCont ) m_objCont->closeGroup( group ) ;
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_corrDataCont ) m_corrDataCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
