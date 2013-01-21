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
AcqirisConfigV1Cvt::AcqirisConfigV1Cvt (const std::string& typeGroupName, hsize_t chunk_size, int deflate, SrcFilter srcFilter)
  : EvtDataTypeCvt<Pds::Acqiris::ConfigV1>(typeGroupName, chunk_size, deflate, srcFilter)
  , m_configCont(0)
  , m_horizCont(0)
  , m_trigCont(0)
  , m_vertCont(0)
{
}

//--------------
// Destructor --
//--------------
AcqirisConfigV1Cvt::~AcqirisConfigV1Cvt ()
{
  delete m_configCont ;
  delete m_horizCont ;
  delete m_trigCont ;
  delete m_vertCont ;
}

/// method called to create all necessary data containers
void
AcqirisConfigV1Cvt::makeContainers(hsize_t chunk_size, int deflate,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // make containers for data objects
  m_configCont = new ConfigCont(ConfigCont::factory_type("config", chunk_size, deflate, true));
  m_horizCont = new HorizCont(HorizCont::factory_type("horiz", chunk_size, deflate, true));
  m_trigCont = new TrigCont(TrigCont::factory_type("trig", chunk_size, deflate, true));
  m_vertCont = new VertCont(VertCont::factory_type("vert", chunk_size, deflate, true));
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

  m_configCont->container(group)->append(cdata);
  m_horizCont->container(group)->append(hdata);
  m_trigCont->container(group)->append(tdata);
  hdf5pp::Type vType = hdf5pp::TypeTraits<H5DataTypes::AcqirisVertV1>::native_type(nbrChannels);
  m_vertCont->container(group, vType)->append(vdata[0], vType);
}

/// method called when the driver closes a group in the file
void
AcqirisConfigV1Cvt::closeContainers(hdf5pp::Group group)
{
  if (m_configCont) m_configCont->closeGroup(group);
  if (m_horizCont) m_horizCont->closeGroup(group);
  if (m_trigCont) m_trigCont->closeGroup(group);
  if (m_vertCont) m_vertCont->closeGroup(group);
}

} // namespace O2OTranslator
