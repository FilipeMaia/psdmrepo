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
Gsc16aiDataV1Cvt::Gsc16aiDataV1Cvt (const std::string& typeGroupName,
    const ConfigObjectStore& configStore,
    hsize_t chunk_size,
    int deflate)
  : EvtDataTypeCvt<Pds::Gsc16ai::DataV1>(typeGroupName, chunk_size, deflate)
  , m_configStore(configStore)
  , m_dataCont(0)
  , m_valueCont(0)
{
}

//--------------
// Destructor --
//--------------
Gsc16aiDataV1Cvt::~Gsc16aiDataV1Cvt ()
{
  delete m_dataCont ;
  delete m_valueCont ;
}

// method called to create all necessary data containers
void
Gsc16aiDataV1Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // create container for frames
  DataCont::factory_type dataContFactory( "timestamps", chunk_size, deflate, true ) ;
  m_dataCont = new DataCont ( dataContFactory ) ;

  // create container for frame data
  ValueCont::factory_type valueContFactory( "channelValue", chunk_size, deflate, true ) ;
  m_valueCont = new ValueCont ( valueContFactory ) ;
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
  H5DataTypes::Gsc16aiDataV1 timestampsData(data);

  // store the data
  m_dataCont->container(group)->append(timestampsData) ;
  hdf5pp::Type type = H5DataTypes::Gsc16aiDataV1::stored_data_type(*config);
  m_valueCont->container(group,type)->append(data._channelValue[0], type);
}

/// method called when the driver closes a group in the file
void
Gsc16aiDataV1Cvt::closeContainers( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_valueCont ) m_valueCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
