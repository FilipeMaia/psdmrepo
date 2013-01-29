//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataProxyT...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/DataProxyT.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPadPedestalsV1.h"
#include "pdscalibdata/CsPadPixelGainV1.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "DataProxyT";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
template <typename DataType, typename ElemType>
DataProxyT<DataType, ElemType>::DataProxyT (const PSEvt::EventKey& key, PSEnv::EnvObjectStore& calibStore)
  : PSEvt::Proxy<DataIfaceType>()
  , m_key(key)
  , m_calibStore(calibStore)
  , m_data()
{
}

//--------------
// Destructor --
//--------------
template <typename DataType, typename ElemType>
DataProxyT<DataType, ElemType>::~DataProxyT ()
{
}

template <typename DataType, typename ElemType>
boost::shared_ptr<typename DataType::IfaceType>
DataProxyT<DataType, ElemType>::getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key)
{
  if (m_data.get()) return m_data;

  // get the original object
  boost::shared_ptr<void> vptr = dict->get(m_key.typeinfo(), PSEvt::Source(m_key.src()), m_key.key(), 0);
  if (not vptr.get()) return m_data;
  boost::shared_ptr<DataIfaceType> obj = boost::static_pointer_cast<DataIfaceType>(vptr);

  // get calibration data
  boost::shared_ptr<pdscalibdata::CsPadPedestalsV1> pedestals = m_calibStore.get(m_key.src());
  boost::shared_ptr<pdscalibdata::CsPadPixelGainV1> pixelGain = m_calibStore.get(m_key.src());
  boost::shared_ptr<pdscalibdata::CsPadPixelStatusV1> pixStatusCalib = m_calibStore.get(m_key.src());
  boost::shared_ptr<pdscalibdata::CsPadCommonModeSubV1> cModeCalib = m_calibStore.get(m_key.src());

  // make new instance
  m_data.reset(new DataType());

  // number of elements
  const int nElem = obj->quads_shape()[0];

  for (int iElem = 0; iElem != nElem; ++ iElem) {

    const ElemIfaceType& elem = obj->quads(iElem);

    // quadrant number
    int iq = elem.quad();

    // data array
    const ndarray<int16_t, 3>& data = elem.data();

    uint32_t sMask = elem.sectionMask();

    // get few constants
    const unsigned nSect = data.shape()[0];
    const unsigned ssize = Pds::CsPad::ColumnsPerASIC*Pds::CsPad::MaxRowsPerASIC*2;

    // make data arrays
    int16_t* pixelData = new int16_t[nSect*ssize];
    float commonMode[nSect];

    // loop over sections
    for ( unsigned is = 0, sect = 0 ; is != Pds::CsPad::ASICsPerQuad/2 ; ++ is ) {

      if ( not (sMask & (1 << is))) continue;

      // start of pixel data
      const int16_t* sdata = &data[sect][0][0];

      // output pixel data
      int16_t* output = pixelData + sect*ssize;

      // default common mode value for this section,
      // large negative number means unknown
      commonMode[sect] = pdscalibdata::CsPadCommonModeSubV1::UnknownCM;

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

      // this sector's pixel gain data
      const float* gaindata = 0;
      if (pixelGain) {
        gaindata = &pixelGain->pixelGains()[iq][is][0][0];
      }

      // calculate common mode if requested
      float cmode = 0;
      if (cModeCalib.get()) {
        MsgLog(logger, debug, "calculating common mode for s=" << sect);
        cmode = cModeCalib->findCommonMode(sdata, peddata, pixStatus, ssize);
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
        for (unsigned i = 0; i != ssize; ++ i) {
          double val = (sdata[i] - peddata[i] - cmode) * gaindata[i];
          output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
        }
      } else if (peddata) {
        for (unsigned i = 0; i != ssize; ++ i) {
          double val = sdata[i] - peddata[i] - cmode;
          output[i] = val < 0 ? int(val - 0.5) : int(val + 0.5);
        }
      } else if (gaindata) {
        for (unsigned i = 0; i != ssize; ++ i) {
          double val = (sdata[i] - cmode) * gaindata[i];
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

    m_data->append(new ElemType(elem, pixelData, commonMode));
  }

  return m_data;
}

// explicit instatiation
template class DataProxyT<cspad_mod::DataV1, cspad_mod::ElementV1>;
template class DataProxyT<cspad_mod::DataV2, cspad_mod::ElementV2>;

} // namespace cspad_mod
