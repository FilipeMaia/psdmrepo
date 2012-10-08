//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisTdcDataV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/AcqirisTdcDataV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "AcqirisTdcDataV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
AcqirisTdcDataV1Cvt::AcqirisTdcDataV1Cvt (const std::string& typeGroupName,
                                    hsize_t chunk_size,
                                    int deflate )
  : EvtDataTypeCvt<XtcType>( typeGroupName )
  , m_chunk_size( chunk_size )
  , m_deflate( deflate )
  , m_dataCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
AcqirisTdcDataV1Cvt::~AcqirisTdcDataV1Cvt ()
{
  delete m_dataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
AcqirisTdcDataV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                            const XtcType& data,
                                            size_t size,
                                            const Pds::TypeId& typeId,
                                            const O2OXtcSrc& src,
                                            const H5DataTypes::XtcClockTime& time )
{
  if ( size % H5Type::xtcSize(data) != 0 ) {
    throw O2OXTCSizeException ( ERR_LOC, "Acqiris::TdcDataV1", H5Type::xtcSize(data), size ) ;
  }

  size_t count = size / H5Type::xtcSize(data);
  
  if ( not m_dataCont ) {

    // make container for data objects
    CvtDataContFactoryDef<H5Type> dataContFactory ( "data", m_chunk_size, m_deflate, true ) ;
    m_dataCont = new DataCont ( dataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  // store the data in the containers
  H5Type h5data(count, &data);
  m_dataCont->container(group)->append (h5data) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
AcqirisTdcDataV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}


} // namespace O2OTranslator
