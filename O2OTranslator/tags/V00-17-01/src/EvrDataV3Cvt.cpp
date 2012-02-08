//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrDataV3Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/EvrDataV3Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/evr/ConfigV3.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "EvrDataV3Cvt" ;
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
EvrDataV3Cvt::EvrDataV3Cvt ( const std::string& typeGroupName,
                             const ConfigObjectStore& configStore,
                             hsize_t chunk_size,
                             int deflate )
  : EvtDataTypeCvt<Pds::EvrData::DataV3>(typeGroupName)
  , m_configStore(configStore)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_evrDataCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
EvrDataV3Cvt::~EvrDataV3Cvt ()
{
  delete m_evrDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
EvrDataV3Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTime& time )
{
  // create all containers if running first time
  if ( not m_evrDataCont ) {

    // create container for frames
    CvtDataContFactoryDef<H5DataTypes::EvrDataV3> evrDataContFactory( "evrData", m_chunk_size, m_deflate, true ) ;
    m_evrDataCont = new EvrDataCont ( evrDataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  // store the data
  H5DataTypes::EvrDataV3 evrData(data);
  m_evrDataCont->container(group)->append ( evrData ) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
EvrDataV3Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_evrDataCont ) m_evrDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
