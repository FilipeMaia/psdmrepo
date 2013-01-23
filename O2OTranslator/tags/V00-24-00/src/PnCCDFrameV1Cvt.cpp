//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDFrameV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/PnCCDFrameV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/pnCCD/ConfigV1.hh"
#include "pdsdata/pnCCD/ConfigV2.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "PnCCDFrameV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
PnCCDFrameV1Cvt::PnCCDFrameV1Cvt ( const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CvtOptions& cvtOptions )
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions)
  , m_configStore(configStore)
  , m_frameCont(0)
  , m_frameDataCont(0)
{
}

//--------------
// Destructor --
//--------------
PnCCDFrameV1Cvt::~PnCCDFrameV1Cvt ()
{
  delete m_frameCont ;
  delete m_frameDataCont ;
}

// method called to create all necessary data containers
void
PnCCDFrameV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
}

// typed conversion method
void
PnCCDFrameV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::PNCCD::ConfigV1 config;
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_pnCCDconfig, 1);
  const Pds::PNCCD::ConfigV1* config1 = m_configStore.find<Pds::PNCCD::ConfigV1>(cfgTypeId, src.top());
  MsgLog( logger, debug, "PnCCDFrameV1Cvt: looking for config object "
      << src.top()
      << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
      << " version=" <<  cfgTypeId.version() ) ;
  if ( config1 ) {
    config = Pds::PNCCD::ConfigV1(*config1);
  } else {
    Pds::TypeId cfgTypeId(Pds::TypeId::Id_pnCCDconfig, 2);
    const Pds::PNCCD::ConfigV2* config2 = m_configStore.find<Pds::PNCCD::ConfigV2>(cfgTypeId, src.top());
    MsgLog( logger, debug, "PnCCDFrameV1Cvt: looking for config object "
        << src.top()
        << " name=" <<  Pds::TypeId::name(cfgTypeId.id())
        << " version=" <<  cfgTypeId.version() ) ;
    if (not config2) {
      MsgLog ( logger, error, "PnCCDFrameV1Cvt - no configuration object was defined" );
      return ;
    }
    config = Pds::PNCCD::ConfigV1(config2->numLinks(), config2->payloadSizePerLink());
  }

  // get few constants
  const uint32_t numLinks = config.numLinks() ;
  const unsigned sizeofData = data.sizeofData(config) ;

  // make data arrays
  H5Type frame[numLinks] ;
  uint16_t frameData[numLinks][sizeofData] ;

  // move the data
  const XtcType* dd = &data ;
  for ( unsigned link = 0 ; link != numLinks ; ++ link ) {

    // copy frame info
    frame[link] = H5Type(*dd) ;

    // copy data
    const uint16_t* ddata = dd->data() ;
    std::copy( ddata, ddata+sizeofData, (uint16_t*)frameData[link] ) ;

    // move to next frame
    dd = dd->next(config) ;
  }

  // store the data
  hdf5pp::Type type = H5Type::stored_type ( config ) ;
  if (not m_frameCont) m_frameCont = makeCont<FrameCont>("frame", group, true, type);
  m_frameCont->append(frame[0], type);

  type = H5Type::stored_data_type ( config ) ;
  if (not m_frameDataCont) m_frameDataCont = makeCont<FrameDataCont>("data", group, true, type);
  m_frameDataCont->append(frameData[0][0], type);
}

} // namespace O2OTranslator
