//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisConfigV1Cvt...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/AcqirisConfigV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "AcqirisConfigV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
AcqirisConfigV1Cvt::AcqirisConfigV1Cvt (const hdf5pp::Group& group, const std::string& typeGroupName,
    const Pds::Src& src, const CvtOptions& cvtOptions, int schemaVersion)
  : EvtDataTypeCvt<Pds::Acqiris::ConfigV1>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configCont()
  , m_horizCont()
  , m_trigCont()
  , m_vertCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
AcqirisConfigV1Cvt::~AcqirisConfigV1Cvt ()
{
}

/// method called to create all necessary data containers
void
AcqirisConfigV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // make containers for data objects
  m_configCont = makeCont<ConfigCont>("config", group, true);
  m_horizCont = makeCont<HorizCont>("horiz", group, true);
  m_trigCont = makeCont<TrigCont>("trig", group, true);
  // m_vertCont needs actual data to know its type
}

// typed conversion method
void
AcqirisConfigV1Cvt::fillContainers(hdf5pp::Group group,
                            const XtcType& data,
                            size_t size,
                            const Pds::TypeId& typeId,
                            const O2OXtcSrc& src)
{
  // make scalar data set for main object
  H5DataTypes::AcqirisConfigV1 cdata ( data ) ;
  H5DataTypes::AcqirisHorizV1 hdata ( data.horiz() ) ;
  H5DataTypes::AcqirisTrigV1 tdata ( data.trig() ) ;

  // make array data set for subobject
  const uint32_t nbrChannels = data.nbrChannels() ;
  H5DataTypes::AcqirisVertV1 vdata[nbrChannels] ;
  for ( uint32_t i = 0 ; i < nbrChannels ; ++ i ) {
    vdata[i] = H5DataTypes::AcqirisVertV1( data.vert(i) ) ;
  }

  hdf5pp::Type vType = hdf5pp::TypeTraits<H5DataTypes::AcqirisVertV1>::native_type(nbrChannels);
  if (not m_vertCont) {
    m_vertCont = makeCont<VertCont>("vert", group, true, vType);
    if (n_miss) m_vertCont->resize(n_miss);
  }

  m_configCont->append(cdata);
  m_horizCont->append(hdata);
  m_trigCont->append(tdata);
  m_vertCont->append(vdata[0], vType);
}

// fill containers for missing data
void
AcqirisConfigV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  m_configCont->resize(m_configCont->size() + 1);
  m_horizCont->resize(m_horizCont->size() + 1);
  m_trigCont->resize(m_trigCont->size() + 1);
  if (m_vertCont) {
    m_vertCont->resize(m_vertCont->size() + 1);
  } else {
    ++ n_miss;
  }

}

} // namespace O2OTranslator
