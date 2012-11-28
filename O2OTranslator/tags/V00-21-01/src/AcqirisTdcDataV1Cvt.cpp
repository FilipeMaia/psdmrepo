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
  : EvtDataTypeCvt<XtcType>( typeGroupName, chunk_size, deflate )
  , m_dataCont(0)
{
}

//--------------
// Destructor --
//--------------
AcqirisTdcDataV1Cvt::~AcqirisTdcDataV1Cvt ()
{
  delete m_dataCont ;
}

/// method called to create all necessary data containers
void
AcqirisTdcDataV1Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // make container for data objects
  DataCont::factory_type dataContFactory ( "data", chunk_size, deflate, true ) ;
  m_dataCont = new DataCont ( dataContFactory ) ;
}

// typed conversion method
void
AcqirisTdcDataV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  if ( size % H5Type::xtcSize(data) != 0 ) {
    throw O2OXTCSizeException ( ERR_LOC, "Acqiris::TdcDataV1", H5Type::xtcSize(data), size ) ;
  }

  size_t count = size / H5Type::xtcSize(data);
  
  // store the data in the containers
  H5Type h5data(count, &data);
  m_dataCont->container(group)->append (h5data) ;
}

/// method called when the driver closes a group in the file
void
AcqirisTdcDataV1Cvt::closeContainers( hdf5pp::Group group )
{
  if ( m_dataCont ) m_dataCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
