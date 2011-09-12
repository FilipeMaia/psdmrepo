//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadMiniElementV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CsPadMiniElementV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPadFilterV1.h"
#include "pdscalibdata/CsPadMiniPedestalsV1.h"
#include "pdscalibdata/CsPadMiniPixelStatusV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "CsPadMiniElementV1Cvt" ;
  
  // slow bit count
  unsigned bitCount(uint32_t mask, unsigned maxBits) {
    unsigned res = 0;
    for (  ; maxBits ; -- maxBits ) {
      if ( mask & 1 ) ++ res ;
      mask >>= 1 ;
    }
    return res ;
  }
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
CsPadMiniElementV1Cvt::CsPadMiniElementV1Cvt ( const std::string& typeGroupName,
                                               const CalibObjectStore& calibStore,
                                               hsize_t chunk_size,
                                               int deflate )
  : EvtDataTypeCvt<Pds::CsPad::MiniElementV1>(typeGroupName)
  , m_calibStore(calibStore)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_elementCont(0)
  , m_pixelDataCont(0)
  , m_cmodeDataCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
CsPadMiniElementV1Cvt::~CsPadMiniElementV1Cvt ()
{
  delete m_elementCont ;
  delete m_pixelDataCont ;
  delete m_cmodeDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
CsPadMiniElementV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                              const XtcType& data,
                                              size_t size,
                                              const Pds::TypeId& typeId,
                                              const O2OXtcSrc& src,
                                              const H5DataTypes::XtcClockTime& time )
{
  // get calibrarion data
  const Pds::DetInfo& address = static_cast<const Pds::DetInfo&>(src.top());
  boost::shared_ptr<pdscalibdata::CsPadMiniPedestalsV1> pedestals =
    m_calibStore.get<pdscalibdata::CsPadMiniPedestalsV1>(address);
  boost::shared_ptr<pdscalibdata::CsPadMiniPixelStatusV1> pixStatusCalib =
    m_calibStore.get<pdscalibdata::CsPadMiniPixelStatusV1>(address);
  boost::shared_ptr<pdscalibdata::CsPadCommonModeSubV1> cModeCalib =
    m_calibStore.get<pdscalibdata::CsPadCommonModeSubV1>(address);
  boost::shared_ptr<pdscalibdata::CsPadFilterV1> filterCalib =
    m_calibStore.get<pdscalibdata::CsPadFilterV1>(address);

  // create all containers if running first time
  if ( not m_elementCont ) {

    // create container for frames
    CvtDataContFactoryTyped<H5DataTypes::CsPadMiniElementV1> elContFactory( "element", m_chunk_size, m_deflate, true ) ;
    m_elementCont = new ElementCont ( elContFactory ) ;

    // create container for frame data
    CvtDataContFactoryTyped<int16_t> dataContFactory( "data", m_chunk_size, m_deflate, true ) ;
    m_pixelDataCont = new PixelDataCont ( dataContFactory ) ;

    if (cModeCalib.get()) {
      // create container for common mode data
      CvtDataContFactoryTyped<float> cmodeContFactory( "common_mode", m_chunk_size, m_deflate, true ) ;
      m_cmodeDataCont = new CommonModeDataCont ( cmodeContFactory ) ;
    }

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  // get few constants
  const unsigned nSect = 2;
  const unsigned ssize = Pds::CsPad::ColumnsPerASIC*Pds::CsPad::MaxRowsPerASIC*2;

  // make data arrays
  int16_t pixelData[Pds::CsPad::ColumnsPerASIC][Pds::CsPad::MaxRowsPerASIC*2][nSect];
  float commonMode[nSect];

  // move the data
  H5DataTypes::CsPadMiniElementV1 elem(data);

  // loop over sections
  for ( unsigned sect = 0; sect < nSect ; ++ sect ) {

    // start of pixel data
    const uint16_t* sdata = (const uint16_t*)(&data.pair[0][0]) + sect;

    // output pixel data
    int16_t* output = &pixelData[0][0][sect];

    // default common mode value for this section,
    // large negative number means unknown
    commonMode[sect] = pdscalibdata::CsPadCommonModeSubV1::UnknownCM;

    // status codes for pixels
    const uint16_t* pixStatus = 0;
    if (pixStatusCalib.get()) {
      pixStatus = &pixStatusCalib->status()[0][0][sect];
    }

    // this sector's pedestal data
    const float* peddata = 0;
    if (pedestals.get()) {
      peddata = &pedestals->pedestals()[0][0][sect];
    }

    // calculate common mode if requested
    float cmode = 0;
    if (cModeCalib.get()) {
      MsgLog(logger, debug, "calculating common mode for s=" << sect);
      cmode = cModeCalib->findCommonMode(sdata, peddata, pixStatus, ssize, nSect);
      if (cmode == pdscalibdata::CsPadCommonModeSubV1::UnknownCM) {
        // reset subtracted value
        cmode = 0;
      } else {
        // remember it
        commonMode[sect] = cmode;
      }
    }

    // subtract pedestals and common mode, plus round to nearest int
    if (peddata) {
      for (unsigned i = 0; i != ssize; i += nSect) {
        double val = sdata[i] - peddata[i] - cmode;
        output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
      }
    } else {
      for (unsigned i = 0; i != ssize; i += nSect) {
        double val = sdata[i] - - cmode;
        output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
      }
    }

  }

  // may not need it
  bool filter = true;
  if (filterCalib.get()) filter = filterCalib->filter(&pixelData[0][0][0], ssize*nSect);
  if (not filter) {
    MsgLog(logger, debug, "skipping CsPad data");
    return;
  }

  // store the data
  hdf5pp::Type type = H5DataTypes::CsPadMiniElementV1::stored_type();
  m_elementCont->container(group,type)->append ( elem, type ) ;
  type = H5DataTypes::CsPadMiniElementV1::stored_data_type() ;
  m_pixelDataCont->container(group,type)->append ( pixelData[0][0][0], type ) ;
  if (m_cmodeDataCont) {
    type = H5DataTypes::CsPadMiniElementV1::cmode_data_type() ;
    m_cmodeDataCont->container(group,type)->append ( commonMode[0], type ) ;
  }
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
CsPadMiniElementV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_elementCont ) m_elementCont->closeGroup( group ) ;
  if ( m_pixelDataCont ) m_pixelDataCont->closeGroup( group ) ;
  if ( m_cmodeDataCont ) m_cmodeDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
