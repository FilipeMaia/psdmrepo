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
#include "pdsdata/cspad/ConfigV4.hh"
#include "pdsdata/cspad/ConfigV5.hh"

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
CsPadElementV2Cvt::CsPadElementV2Cvt ( const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CalibObjectStore& calibStore,
    const CvtOptions& cvtOptions,
    int schemaVersion )
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
  , m_configStore(configStore)
  , m_calibStore(calibStore)
  , m_elementCont()
  , m_pixelDataCont()
  , m_cmodeDataCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
CsPadElementV2Cvt::~CsPadElementV2Cvt ()
{
}

// method called to create all necessary data containers
void
CsPadElementV2Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // nothing to do here, we need real data for this
}

// typed conversion method
void
CsPadElementV2Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // based on cspad/ElementIterator but we cannot use that class directly
  uint32_t qMask = 0;
  uint32_t sMask[Pds::CsPad::MaxQuadsPerSensor];
  unsigned sections = 0;
  
  // find corresponding configuration object, it could be ConfigV2 or ConfigV3
  Pds::TypeId cfgTypeId2(Pds::TypeId::Id_CspadConfig, 2);
  Pds::TypeId cfgTypeId3(Pds::TypeId::Id_CspadConfig, 3);
  Pds::TypeId cfgTypeId4(Pds::TypeId::Id_CspadConfig, 4);
  Pds::TypeId cfgTypeId5(Pds::TypeId::Id_CspadConfig, 5);
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
  } else if ( const Pds::CsPad::ConfigV4* config = m_configStore.find<Pds::CsPad::ConfigV4>(cfgTypeId4, src.top()) ) {
    qMask = config->quadMask();
    for (int q = 0 ; q != Pds::CsPad::MaxQuadsPerSensor; ++ q) {
      sMask[q] = config->roiMask(q);
      sections += ::bitCount(sMask[q], Pds::CsPad::ASICsPerQuad/2);
    }
  } else if ( const Pds::CsPad::ConfigV5* config = m_configStore.find<Pds::CsPad::ConfigV5>(cfgTypeId5, src.top()) ) {
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

  // get few constants
  const unsigned nQuad = ::bitCount(qMask, Pds::CsPad::MaxQuadsPerSensor);
  const unsigned nSect = sections;
  const unsigned ssize = Pds::CsPad::ColumnsPerASIC*Pds::CsPad::MaxRowsPerASIC*2;

  // make data arrays
  H5Type elems[nQuad] ;
  int16_t pixelData[nSect][Pds::CsPad::ColumnsPerASIC][Pds::CsPad::MaxRowsPerASIC*2];
  float commonMode[nSect];
  const uint16_t* pixStatus[nSect];

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
    elems[quad] = H5Type(*pdselem) ;

    // start of pixel data
    const int16_t* sdata = (const int16_t*)(pdselem+1);

    for ( unsigned is = 0 ; is != Pds::CsPad::ASICsPerQuad/2 ; ++ is ) {
    
      if ( not (sMask[q] & (1 << is))) continue; 
  
      // output pixel data
      int16_t* output = &pixelData[sect][0][0];

      // default common mode value for this section,
      // large negative number means unknown
      commonMode[sect] = pdscalibdata::CsPadCommonModeSubV1::UnknownCM;

      // status codes for pixels
      pixStatus[sect] = 0;
      if (pixStatusCalib.get()) {
        pixStatus[sect] = &pixStatusCalib->status()[q][is][0][0];
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
        cmode = cModeCalib->findCommonMode(sdata, peddata, pixStatus[sect], ssize);
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
          double val = sdata[i] - peddata[i] - cmode;
          output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
        }
      } else {
        for (unsigned i = 0; i != ssize; ++ i) {
          double val = sdata[i] - cmode;
          output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
        }
      }
      
      // advance to next section
      sdata += ssize;
      ++ sect;
    }

  }

  // may not need it
  bool filter = true;
  if (filterCalib.get()) {

    // wrap data into ndarray
    ndarray<int16_t, 3> pixArr = make_ndarray(&pixelData[0][0][0], nSect, Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC*2);

    if (pixStatusCalib) {
      // copy pixel status to an array
      uint16_t pixelStatusArray[nSect][Pds::CsPad::ColumnsPerASIC][Pds::CsPad::MaxRowsPerASIC*2];
      for (unsigned sect = 0; sect != nSect; ++ sect) {
        std::copy(pixStatus[sect], pixStatus[sect]+ssize, &pixelStatusArray[sect][0][0]);
      }
      ndarray<uint16_t, 3> statArr = make_ndarray(&pixelStatusArray[0][0][0], nSect, Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC*2);
      filter = filterCalib->filter(pixArr, statArr);
    } else {
      filter = filterCalib->filter(pixArr);
    }
  }
  if (not filter) {
    MsgLog(logger, debug, "skipping CsPad data");
    return;
  }
  
  // store the data
  hdf5pp::Type type = H5Type::stored_type(nQuad);
  if (not m_elementCont) {
    m_elementCont = makeCont<ElementCont>("element", group, true, type);
    if (n_miss) m_elementCont->resize(n_miss);
  }
  m_elementCont->append ( elems[0], type ) ;

  type = H5Type::stored_data_type(nSect) ;
  if (not m_pixelDataCont) {
    m_pixelDataCont = makeCont<PixelDataCont>("data", group, true, type);
    if (n_miss) m_pixelDataCont->resize(n_miss);
  }
  m_pixelDataCont->append ( pixelData[0][0][0], type ) ;

  if (cModeCalib) {
    type = H5Type::cmode_data_type(nSect) ;
    if (not m_cmodeDataCont) {
      m_cmodeDataCont = makeCont<CommonModeDataCont>("common_mode", group, true, type);
      if (n_miss) m_cmodeDataCont->resize(n_miss);
    }
    m_cmodeDataCont->append ( commonMode[0], type ) ;
  }
}

// fill containers for missing data
void
CsPadElementV2Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  if (m_elementCont) {
    m_elementCont->resize(m_elementCont->size() + 1);
    m_pixelDataCont->resize(m_pixelDataCont->size() + 1);
    m_cmodeDataCont->resize(m_cmodeDataCont->size() + 1);
  } else {
    ++ n_miss;
  }
}

} // namespace O2OTranslator
