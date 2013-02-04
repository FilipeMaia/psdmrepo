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
AcqirisTdcDataV1Cvt::AcqirisTdcDataV1Cvt (const hdf5pp::Group& group, const std::string& typeGroupName,
    const Pds::Src& src, const CvtOptions& cvtOptions )
  : EvtDataTypeCvt<XtcType>( group, typeGroupName, src, cvtOptions )
  , m_dataCont()
{
}

//--------------
// Destructor --
//--------------
AcqirisTdcDataV1Cvt::~AcqirisTdcDataV1Cvt ()
{
}

/// method called to create all necessary data containers
void
AcqirisTdcDataV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // make container for data objects
  m_dataCont = makeCont<DataCont>("data", group, true) ;
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
  m_dataCont->append (h5data) ;
}

// fill containers for missing data
void
AcqirisTdcDataV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  m_dataCont->resize(m_dataCont->size() + 1);
}

} // namespace O2OTranslator
