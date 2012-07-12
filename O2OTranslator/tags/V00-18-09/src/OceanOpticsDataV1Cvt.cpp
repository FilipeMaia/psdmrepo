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
  : EvtDataTypeCvt<XtcType>(typeGroupName)
  , m_configStore(configStore)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_objCont(0)
  , m_dataCont(0)
  , m_corrDataCont(0)
  , m_timeCont(0)
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
  delete m_timeCont ;
}

// typed conversion method
void
OceanOpticsDataV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTime& time )
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId1(Pds::TypeId::Id_OceanOpticsConfig, 1);
  const Pds::OceanOptics::ConfigV1* config = m_configStore.find<Pds::OceanOptics::ConfigV1>(cfgTypeId1, src.top());
  if (not config) {
    MsgLog ( logger, error, "OceanOpticsDataV1Cvt - no configuration object was defined" );
    return ;
  }

  // create all containers if running first time
  if ( not m_objCont ) {

    // create container for objects
    CvtDataContFactoryDef<H5DataTypes::OceanOpticsDataV1> objContFactory( "data", m_chunk_size, m_deflate, true ) ;
    m_objCont = new ObjectCont ( objContFactory ) ;

    // create container for data
    CvtDataContFactoryTyped<uint16_t> dataContFactory( "spectra", m_chunk_size, m_deflate, true ) ;
    m_dataCont = new DataCont ( dataContFactory ) ;

    // create container for corrected data
    CvtDataContFactoryTyped<float> corrDataContFactory( "corrSpectra", m_chunk_size, m_deflate, true ) ;
    m_corrDataCont = new CorrectedDataCont ( corrDataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  // make corrected data
  float corrData[Pds::OceanOptics::DataV1::iNumPixels];
  for (int i = 0; i != Pds::OceanOptics::DataV1::iNumPixels; ++ i) {
    corrData[i] = data.nonlinerCorrected(*config, i);
  }

  // store the data
  H5DataTypes::OceanOpticsDataV1 obj(data);
  m_objCont->container(group)->append(obj) ;
  hdf5pp::Type type = H5DataTypes::OceanOpticsDataV1::stored_data_type() ;
  m_dataCont->container(group,type)->append(*data.data(), type);
  type = H5DataTypes::OceanOpticsDataV1::stored_corrected_data_type() ;
  m_corrDataCont->container(group,type)->append(*corrData, type);
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
OceanOpticsDataV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_objCont ) m_objCont->closeGroup( group ) ;
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_corrDataCont ) m_corrDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
