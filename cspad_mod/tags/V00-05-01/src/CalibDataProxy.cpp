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
#include "pdscalibdata/CsPad2x2PedestalsV1.h"
#include "pdscalibdata/CsPad2x2PixelStatusV1.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPadFilterV1.h"
#include "pdscalibdata/CsPadPedestalsV1.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "PSCalib/CalibFileFinder.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "CalibDataProxy";

  /// Return calibration class (like CsPad::CalibV1) for given source
  std::string source2class(const Pds::Src& src)
  {
    if (src.level() == Pds::Level::Source) {
      const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>(src);
      if (info.device() == Pds::DetInfo::Cspad2x2) return "CsPad2x2::CalibV1";
      if (info.device() == Pds::DetInfo::Cspad) return "CsPad::CalibV1";
    }
    return std::string();
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
template <typename T>
CalibDataProxy<T>::CalibDataProxy (const std::string& calibDir, const std::string& calibType, int run)
  : PSEvt::Proxy<T>()
  , m_calibDir(calibDir)
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

  const std::string& cclass = ::source2class(source);
  if (not cclass.empty()) {

    PSCalib::CalibFileFinder finder(m_calibDir, cclass);
    std::string calibFileName = finder.findCalibFile(source, m_calibType, m_run);
    if (not calibFileName.empty()) {
      MsgLog(logger, trace, "CalibDataProxy: found calibration file " << calibFileName);
      // make new instance that will read data from file
      ptr.reset(new T(calibFileName));
    }

  }

  // store it in cache
  m_data.insert(typename Src2Data::value_type(src, ptr));

  return ptr;
}


// explicit instantiation
template class CalibDataProxy<pdscalibdata::CsPad2x2PedestalsV1>;
template class CalibDataProxy<pdscalibdata::CsPad2x2PixelStatusV1>;
template class CalibDataProxy<pdscalibdata::CsPadCommonModeSubV1>;
template class CalibDataProxy<pdscalibdata::CsPadFilterV1>;
template class CalibDataProxy<pdscalibdata::CsPadPedestalsV1>;
template class CalibDataProxy<pdscalibdata::CsPadPixelStatusV1>;

} // namespace cspad_mod
