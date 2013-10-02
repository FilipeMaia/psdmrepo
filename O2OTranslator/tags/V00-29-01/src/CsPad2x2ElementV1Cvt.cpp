//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ElementV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CsPad2x2ElementV1Cvt.h"

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
#include "pdscalibdata/CsPad2x2PedestalsV1.h"
#include "pdscalibdata/CsPad2x2PixelStatusV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "CsPad2x2ElementV1Cvt" ;
  
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
CsPad2x2ElementV1Cvt::CsPad2x2ElementV1Cvt ( const hdf5pp::Group& group, const std::string& typeGroupName,
    const Pds::Src& src, const CalibObjectStore& calibStore, const CvtOptions& cvtOptions, int schemaVersion )
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
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
CsPad2x2ElementV1Cvt::~CsPad2x2ElementV1Cvt ()
{
}

// method called to create all necessary data containers
void
CsPad2x2ElementV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // nothing to do here, we need real data for this
}

// typed conversion method
void
CsPad2x2ElementV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // get calibrarion data
  const Pds::DetInfo& address = static_cast<const Pds::DetInfo&>(src.top());
  boost::shared_ptr<pdscalibdata::CsPad2x2PedestalsV1> pedestals =
    m_calibStore.get<pdscalibdata::CsPad2x2PedestalsV1>(address);
  boost::shared_ptr<pdscalibdata::CsPad2x2PixelStatusV1> pixStatusCalib =
    m_calibStore.get<pdscalibdata::CsPad2x2PixelStatusV1>(address);
  boost::shared_ptr<pdscalibdata::CsPadCommonModeSubV1> cModeCalib =
    m_calibStore.get<pdscalibdata::CsPadCommonModeSubV1>(address);
  boost::shared_ptr<pdscalibdata::CsPadFilterV1> filterCalib =
    m_calibStore.get<pdscalibdata::CsPadFilterV1>(address);

  // get few constants
  const unsigned nSect = 2;
  const unsigned ssize = Pds::CsPad2x2::ColumnsPerASIC*Pds::CsPad2x2::MaxRowsPerASIC*2;

  // make data arrays
  int16_t pixelData[Pds::CsPad2x2::ColumnsPerASIC][Pds::CsPad2x2::MaxRowsPerASIC*2][nSect];
  float commonMode[nSect];

  // move the data
  H5Type elem(data);

  const ndarray<const int16_t, 3>& ndata = data.data();

  // loop over sections
  for ( unsigned sect = 0; sect < nSect ; ++ sect ) {

    // start of pixel data
    const int16_t* sdata = &ndata[0][0][sect];

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
      unsigned last = ssize*nSect;
      for (unsigned i = 0; i != last; i += nSect) {
        double val = sdata[i] - peddata[i] - cmode;
        output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
      }
    } else {
      unsigned last = ssize*nSect;
      for (unsigned i = 0; i != last; i += nSect) {
        double val = sdata[i] - cmode;
        output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
      }
    }

  }

  // may not need it
  bool filter = true;
  if (filterCalib.get()) {
    ndarray<int16_t, 3> pixArr = make_ndarray(&pixelData[0][0][0], Pds::CsPad2x2::ColumnsPerASIC, Pds::CsPad2x2::MaxRowsPerASIC*2, nSect);
    if (pixStatusCalib) {
      filter = filterCalib->filter(pixArr, pixStatusCalib->status());
    } else {
      filter = filterCalib->filter(pixArr);
    }
  }
  if (not filter) {
    MsgLog(logger, debug, "skipping CsPad data");
    return;
  }

  // store the data
  hdf5pp::Type type = H5Type::stored_type();
  if (not m_elementCont) {
    m_elementCont = makeCont<ElementCont>("element", group, true, type);
    if (n_miss) m_elementCont->resize(n_miss);
  }
  m_elementCont->append ( elem, type ) ;

  type = H5Type::stored_data_type() ;
  if (not m_pixelDataCont) {
    m_pixelDataCont = makeCont<PixelDataCont>("data", group, true, type);
    if (n_miss) m_pixelDataCont->resize(n_miss);
  }
  m_pixelDataCont->append ( pixelData[0][0][0], type ) ;

  if (cModeCalib) {
    type = H5Type::cmode_data_type() ;
    if (not m_cmodeDataCont) {
      m_cmodeDataCont = makeCont<CommonModeDataCont>("common_mode", group, true, type);
      if (n_miss) m_cmodeDataCont->resize(n_miss);
    }
    m_cmodeDataCont->append(commonMode[0], type);
  }
}

// fill containers for missing data
void
CsPad2x2ElementV1Cvt::fillMissing(hdf5pp::Group group,
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
