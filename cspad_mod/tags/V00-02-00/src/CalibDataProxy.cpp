//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibDataProxy...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/CalibDataProxy.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPadFilterV1.h"
#include "pdscalibdata/CsPadMiniPedestalsV1.h"
#include "pdscalibdata/CsPadMiniPixelStatusV1.h"
#include "pdscalibdata/CsPadPedestalsV1.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "CalibDataProxy";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
template <typename T>
CalibDataProxy<T>::CalibDataProxy (const std::string& calibDir,
    const std::string& calibClass, const std::string& calibType, int run)
  : PSEvt::Proxy<T>()
  , m_finder(calibDir, calibClass)
  , m_calibType(calibType)
  , m_run(run)
  , m_data()
{
}

//--------------
// Destructor --
//--------------
template <typename T>
CalibDataProxy<T>::~CalibDataProxy ()
{
}

template <typename T>
boost::shared_ptr<T>
CalibDataProxy<T>::getTypedImpl(PSEvt::ProxyDictI* dict,
                                const Pds::Src& source,
                                const std::string& key)
{
  _Src src(source);

  // check cache first
  typename Src2Data::iterator it = m_data.find(src);
  if (it != m_data.end()) return it->second;

  // make a new one
  boost::shared_ptr<T> ptr;

  std::string calibFileName = m_finder.findCalibFile(source, m_calibType, m_run);
  if (not calibFileName.empty()) {
    MsgLog(logger, trace, "CalibDataProxy: found calibration file " << calibFileName);
    // make new instance that will read data from file
    ptr.reset(new T(calibFileName));
  }

  // store it in cache
  m_data.insert(typename Src2Data::value_type(src, ptr));

  return ptr;
}

// explicit instantiation
template class CalibDataProxy<pdscalibdata::CsPadCommonModeSubV1>;
template class CalibDataProxy<pdscalibdata::CsPadFilterV1>;
template class CalibDataProxy<pdscalibdata::CsPadMiniPedestalsV1>;
template class CalibDataProxy<pdscalibdata::CsPadMiniPixelStatusV1>;
template class CalibDataProxy<pdscalibdata::CsPadPedestalsV1>;
template class CalibDataProxy<pdscalibdata::CsPadPixelStatusV1>;

} // namespace cspad_mod
