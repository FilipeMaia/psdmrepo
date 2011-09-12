//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CsPadElementV1Cvt.h"

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
#include "pdscalibdata/CsPadPedestalsV1.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"
#include "pdsdata/cspad/ConfigV1.hh"
#include "pdsdata/cspad/ConfigV2.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "CsPadElementV1Cvt" ;
  
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
CsPadElementV1Cvt::CsPadElementV1Cvt ( const std::string& typeGroupName,
                                   const ConfigObjectStore& configStore,
                                   const CalibObjectStore& calibStore,
                                   hsize_t chunk_size,
                                   int deflate )
  : EvtDataTypeCvt<Pds::CsPad::ElementV1>(typeGroupName)
  , m_configStore(configStore)
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
CsPadElementV1Cvt::~CsPadElementV1Cvt ()
{
  delete m_elementCont ;
  delete m_pixelDataCont ;
  delete m_cmodeDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
CsPadElementV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                          const XtcType& data,
                                          size_t size,
                                          const Pds::TypeId& typeId,
                                          const O2OXtcSrc& src,
                                          const H5DataTypes::XtcClockTime& time )
{
  // based on cspad/ElementIterator but we cannot use that class directly
  uint32_t qMask;
  uint32_t sMask;
  
  // find corresponding configuration object, it could be ConfigV1 or ConfigV2 
  Pds::TypeId cfgTypeId1(Pds::TypeId::Id_CspadConfig, 1);
  Pds::TypeId cfgTypeId2(Pds::TypeId::Id_CspadConfig, 2);
  if ( const Pds::CsPad::ConfigV1* config = m_configStore.find<Pds::CsPad::ConfigV1>(cfgTypeId1, src.top()) ) {
    qMask = config->quadMask();
    sMask = config->asicMask()==1 ? 0x3 : 0xff;
  } else if ( const Pds::CsPad::ConfigV2* config = m_configStore.find<Pds::CsPad::ConfigV2>(cfgTypeId2, src.top()) ) {
    qMask = config->quadMask();
    sMask = config->asicMask()==1 ? 0x3 : 0xff;
  } else {
    MsgLog ( logger, error, "CsPadElementV1Cvt - no configuration object was defined" );
    return ;
  }

  // get calibrarion data
  const Pds::DetInfo& address = static_cast<const Pds::DetInfo&>(src.top());
  boost::shared_ptr<pdscalibdata::CsPadPedestalsV1> pedestals =
    m_calibStore.get<pdscalibdata::CsPadPedestalsV1>(address);
  boost::shared_ptr<pdscalibdata::CsPadPixelStatusV1> pixStatusCalib =
    m_calibStore.get<pdscalibdata::CsPadPixelStatusV1>(address);
  boost::shared_ptr<pdscalibdata::CsPadCommonModeSubV1> cModeCalib =
    m_calibStore.get<pdscalibdata::CsPadCommonModeSubV1>(address);

  // create all containers if running first time
  if ( not m_elementCont ) {

    // create container for frames
    CvtDataContFactoryTyped<H5DataTypes::CsPadElementV1> elContFactory( "element", m_chunk_size, m_deflate, true ) ;
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
  const unsigned nQuad = ::bitCount(qMask, Pds::CsPad::MaxQuadsPerSensor);
  const unsigned nSect = ::bitCount(sMask, Pds::CsPad::ASICsPerQuad/2);
  const unsigned ssize = Pds::CsPad::ColumnsPerASIC*Pds::CsPad::MaxRowsPerASIC*2;
  const unsigned qsize = nSect*ssize;

  // make data arrays
  H5DataTypes::CsPadElementV1 elems[nQuad] ;
  int16_t pixelData[nQuad][nSect][Pds::CsPad::ColumnsPerASIC][Pds::CsPad::MaxRowsPerASIC*2];
  float commonMode[nQuad][nSect];

  // move the data
  const XtcType* pdselem = &data ;
  for ( unsigned iq = 0 ; iq != nQuad ; ++ iq ) {

    // copy frame info
    elems[iq] = H5DataTypes::CsPadElementV1(*pdselem) ;

    const uint16_t* qdata = pdselem->data();

    int sect = 0;
    for ( int is = 0 ; is < Pds::CsPad::ASICsPerQuad/2 ; ++ is ) {
    
      if ( not (sMask & (1<<is)) ) continue;
      
      // start of the section data
      const uint16_t* sdata = qdata + sect*ssize;

      // status codes for pixels
      const uint16_t* pixStatus = 0;
      if (pixStatusCalib.get()) {
        pixStatus = &pixStatusCalib->status()[iq][is][0][0];
      }

      // this sector's pedestal data
      const float* peddata = 0;
      if (pedestals.get()) {
        peddata = &pedestals->pedestals()[iq][is][0][0];
      }
      
      // calculate common mode if requested
      commonMode[iq][sect] = pdscalibdata::CsPadCommonModeSubV1::UnknownCM;      
      float cmode = 0;
      if (cModeCalib.get()) {
        MsgLog(logger, debug, "calculating common mode for q=" << iq << " s=" << is);
        cmode = cModeCalib->findCommonMode(sdata, peddata, pixStatus, ssize);
        if (cmode == pdscalibdata::CsPadCommonModeSubV1::UnknownCM) {
          // reset subtracted value
          cmode = 0;
        } else {
          // remember it
          commonMode[iq][sect] = cmode;
        }
      }

      // start of output data
      int16_t* output = &pixelData[iq][sect][0][0];

      // subtract pedestals and common mode, plus round to nearest int
      if (peddata) {
        for (unsigned i = 0; i != ssize; ++ i) {
          double val = sdata[i] - peddata[i] - cmode;
          output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
        }
      } else {
        for (unsigned i = 0; i != ssize; ++ i) {
          double val = sdata[i] - cmode;
          output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
        }
      }
      
      ++ sect;
    
    }
    
    // move to next frame
    pdselem = (const XtcType*)(qdata+qsize+2) ;
  }

  // store the data
  hdf5pp::Type type = H5DataTypes::CsPadElementV1::stored_type(nQuad);
  m_elementCont->container(group,type)->append ( elems[0], type ) ;
  type = H5DataTypes::CsPadElementV1::stored_data_type(nQuad, nSect) ;
  m_pixelDataCont->container(group,type)->append ( pixelData[0][0][0][0], type ) ;
  if (m_cmodeDataCont) {
    type = H5DataTypes::CsPadElementV1::cmode_data_type(nQuad, nSect) ;
    m_cmodeDataCont->container(group,type)->append ( commonMode[0][0], type ) ;
  }
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
CsPadElementV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_elementCont ) m_elementCont->closeGroup( group ) ;
  if ( m_pixelDataCont ) m_pixelDataCont->closeGroup( group ) ;
  if ( m_cmodeDataCont ) m_cmodeDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
