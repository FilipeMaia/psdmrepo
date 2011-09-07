//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV2Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CsPadElementV2Cvt.h"

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
#include "pdscalibdata/CsPadPedestalsV1.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"
#include "pdsdata/cspad/ConfigV2.hh"
#include "pdsdata/cspad/ConfigV3.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "CsPadElementV2Cvt" ;
  
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
CsPadElementV2Cvt::CsPadElementV2Cvt ( const std::string& typeGroupName,
                                   const ConfigObjectStore& configStore,
                                   const CalibObjectStore& calibStore,
                                   hsize_t chunk_size,
                                   int deflate )
  : EvtDataTypeCvt<Pds::CsPad::ElementV2>(typeGroupName)
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
CsPadElementV2Cvt::~CsPadElementV2Cvt ()
{
  delete m_elementCont ;
  delete m_pixelDataCont ;
  delete m_cmodeDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
CsPadElementV2Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                          const XtcType& data,
                                          size_t size,
                                          const Pds::TypeId& typeId,
                                          const O2OXtcSrc& src,
                                          const H5DataTypes::XtcClockTime& time )
{
  // based on cspad/ElementIterator but we cannot use that class directly
  uint32_t qMask = 0;
  uint32_t sMask[Pds::CsPad::MaxQuadsPerSensor];
  unsigned sections = 0;
  
  // find corresponding configuration object, it could be ConfigV2 or ConfigV3
  Pds::TypeId cfgTypeId2(Pds::TypeId::Id_CspadConfig, 2);
  Pds::TypeId cfgTypeId3(Pds::TypeId::Id_CspadConfig, 3);
  if ( const Pds::CsPad::ConfigV2* config = m_configStore.find<Pds::CsPad::ConfigV2>(cfgTypeId2, src.top()) ) {
    qMask = config->quadMask();
    for (int q = 0 ; q != Pds::CsPad::MaxQuadsPerSensor; ++ q) {
      sMask[q] = config->roiMask(q);
      sections += ::bitCount(sMask[q], Pds::CsPad::ASICsPerQuad/2);
    }
  } else if ( const Pds::CsPad::ConfigV3* config = m_configStore.find<Pds::CsPad::ConfigV3>(cfgTypeId3, src.top()) ) {
    qMask = config->quadMask();
    for (int q = 0 ; q != Pds::CsPad::MaxQuadsPerSensor; ++ q) {
      sMask[q] = config->roiMask(q);
      sections += ::bitCount(sMask[q], Pds::CsPad::ASICsPerQuad/2);
    }
  } else {
    MsgLog ( logger, error, "CsPadElementV2Cvt - no configuration object was defined" );
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
  boost::shared_ptr<pdscalibdata::CsPadFilterV1> filterCalib =
    m_calibStore.get<pdscalibdata::CsPadFilterV1>(address);

  // create all containers if running first time
  if ( not m_elementCont ) {

    // create container for frames
    CvtDataContFactoryTyped<H5DataTypes::CsPadElementV2> elContFactory( "element", m_chunk_size, m_deflate, true ) ;
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
  const unsigned nSect = sections;
  const unsigned ssize = Pds::CsPad::ColumnsPerASIC*Pds::CsPad::MaxRowsPerASIC*2;

  // make data arrays
  H5DataTypes::CsPadElementV2 elems[nQuad] ;
  int16_t pixelData[nSect][Pds::CsPad::ColumnsPerASIC][Pds::CsPad::MaxRowsPerASIC*2];
  float commonMode[nSect];

  // quadrants may come unordered in XTC container, while clients
  // prefer them to be ordered, do ordering here
  const XtcType* pdselem = &data ;
  typedef std::map<unsigned, const XtcType*> ElementMap;
  ElementMap elements;
  for ( unsigned iq = 0 ; iq != nQuad ; ++ iq ) {

    unsigned q = pdselem->quad();
    unsigned nSectPerQuad = ::bitCount(sMask[q], Pds::CsPad::ASICsPerQuad/2);
    
    // add to map ordered on quad number
    elements.insert(ElementMap::value_type(q, pdselem));

    // move to next frame
    const uint16_t* sdata = (const uint16_t*)(pdselem+1);
    sdata += nSectPerQuad*ssize;
    pdselem = (const XtcType*)(sdata+2) ;
  }
  
  
  // move the data
  unsigned quad = 0;
  unsigned sect = 0;
  for ( ElementMap::const_iterator ie = elements.begin() ; ie != elements.end() ; ++ ie, ++ quad ) {
        
    unsigned q = ie->first;
    pdselem = ie->second;

    // copy frame info
    elems[quad] = H5DataTypes::CsPadElementV2(*pdselem) ;

    // start of pixel data
    const uint16_t* sdata = (const uint16_t*)(pdselem+1);

    for ( unsigned is = 0 ; is != Pds::CsPad::ASICsPerQuad/2 ; ++ is ) {
    
      if ( not (sMask[q] & (1 << is))) continue; 
  
      // output pixel data
      int16_t* output = &pixelData[sect][0][0];

      // default common mode value for this section,
      // large negative number means unknown
      commonMode[sect] = pdscalibdata::CsPadCommonModeSubV1::UnknownCM;

      // status codes for pixels
      const uint16_t* pixStatus = 0;
      if (pixStatusCalib.get()) {
        pixStatus = &pixStatusCalib->status()[q][is][0][0];
      }

      // this sector's pedestal data
      const float* peddata = 0;
      if (pedestals.get()) {
        peddata = &pedestals->pedestals()[q][is][0][0];
      }
      
      // calculate common mode if requested
      float cmode = 0;
      if (cModeCalib.get()) {
        MsgLog(logger, debug, "calculating common mode for q=" << q << " s=" << is);
        cmode = cModeCalib->findCommonMode(sdata, peddata, pixStatus, ssize);
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
        for (unsigned i = 0; i != ssize; ++ i) {
          output[i] = int(std::floor(sdata[i] - peddata[i] - cmode + 0.5));
        }
      } else {
        for (unsigned i = 0; i != ssize; ++ i) {
          output[i] = int(std::floor(sdata[i] - cmode + 0.5));
        }
      }
      
      // advance to next section
      sdata += ssize;
      ++ sect;
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
  hdf5pp::Type type = H5DataTypes::CsPadElementV2::stored_type(nQuad);
  m_elementCont->container(group,type)->append ( elems[0], type ) ;
  type = H5DataTypes::CsPadElementV2::stored_data_type(nSect) ;
  m_pixelDataCont->container(group,type)->append ( pixelData[0][0][0], type ) ;
  if (m_cmodeDataCont) {
    type = H5DataTypes::CsPadElementV2::cmode_data_type(nSect) ;
    m_cmodeDataCont->container(group,type)->append ( commonMode[0], type ) ;
  }
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
CsPadElementV2Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_elementCont ) m_elementCont->closeGroup( group ) ;
  if ( m_pixelDataCont ) m_pixelDataCont->closeGroup( group ) ;
  if ( m_cmodeDataCont ) m_cmodeDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
