//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataProxy2x2...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/DataProxy2x2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "cspad_mod/CsPad2x2ElementV1.h"
#include "MsgLogger/MsgLogger.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPad2x2PedestalsV1.h"
#include "pdscalibdata/CsPad2x2PixelGainV1.h"
#include "pdscalibdata/CsPad2x2PixelStatusV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "DataProxy2x2";

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
DataProxy2x2::DataProxy2x2 (const PSEvt::EventKey& key, PSEnv::EnvObjectStore& calibStore)
  : PSEvt::Proxy<Psana::CsPad2x2::ElementV1>()
  , m_key(key)
  , m_calibStore(calibStore)
  , m_data()
{
}

//--------------
// Destructor --
//--------------
DataProxy2x2::~DataProxy2x2 ()
{
}

boost::shared_ptr<Psana::CsPad2x2::ElementV1>
DataProxy2x2::getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key)
{
  if (m_data.get()) return m_data;

  // get the original object
  boost::shared_ptr<void> vptr = dict->get(m_key.typeinfo(), PSEvt::Source(m_key.src()), m_key.key(), 0);
  if (not vptr.get()) return m_data;
  boost::shared_ptr<Psana::CsPad2x2::ElementV1> obj = boost::static_pointer_cast<Psana::CsPad2x2::ElementV1>(vptr);

  // get calibration data
  boost::shared_ptr<pdscalibdata::CsPad2x2PedestalsV1> pedestals = m_calibStore.get(m_key.src());
  boost::shared_ptr<pdscalibdata::CsPad2x2PixelGainV1> pixelGain = m_calibStore.get(m_key.src());
  boost::shared_ptr<pdscalibdata::CsPad2x2PixelStatusV1> pixStatusCalib = m_calibStore.get(m_key.src());
  boost::shared_ptr<pdscalibdata::CsPadCommonModeSubV1> cModeCalib = m_calibStore.get(m_key.src());

  // get few constants
  const unsigned nSect = 2;
  const unsigned ssize = Psana::CsPad2x2::ColumnsPerASIC*Psana::CsPad2x2::MaxRowsPerASIC*2;

  // make data arrays
  int16_t* pixelData = new int16_t[ssize*nSect];
  float commonMode[nSect];

  // loop over sections
  for ( unsigned sect = 0; sect < nSect ; ++ sect ) {

    const ndarray<const int16_t, 3>& data = obj->data();

    // start of pixel data
    const int16_t* sdata = data.data() + sect;

    // output pixel data
    int16_t* output = pixelData + sect;

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

    // this sector's pixel gain data
    const float* gaindata = 0;
    if (pixelGain) {
      gaindata = &pixelGain->pixelGains()[0][0][sect];
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

    // subtract pedestals and common mode, apply pixel gain, and round to
    // nearest int; pixel gain is _inverse_ gain so we multiply it
    if (peddata and gaindata) {
      unsigned last = ssize*nSect;
      for (unsigned i = 0; i != last; i += nSect) {
        double val = (sdata[i] - peddata[i] - cmode) * gaindata[i];
        output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
      }
    } else if (peddata) {
      unsigned last = ssize*nSect;
      for (unsigned i = 0; i != last; i += nSect) {
        double val = sdata[i] - peddata[i] - cmode;
        output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
      }
    } else if (gaindata) {
      unsigned last = ssize*nSect;
      for (unsigned i = 0; i != last; i += nSect) {
        double val = (sdata[i] - cmode) * gaindata[i];
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

  // make new object
  m_data.reset(new CsPad2x2ElementV1(*obj, pixelData, commonMode));

  return m_data;
}

} // namespace cspad_mod
