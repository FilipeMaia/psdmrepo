#ifndef CSPAD_MOD_CALIBDATAPROXY_H
#define CSPAD_MOD_CALIBDATAPROXY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibDataProxy.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>

//----------------------
// Base Class Headers --
//----------------------
#include "PSEvt/Proxy.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace cspad_mod {

/// @addtogroup cspad_mod

/**
 *  @ingroup cspad_mod
 *
 *  @brief Proxy class for calibration data
 *
 *
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename T>
class CalibDataProxy : public PSEvt::Proxy<T> {
public:

  /**
   *   Make new proxy instance
   *
   *   @param[in] calibDir   Experiment calibration directory
   *   @param[in] calibType  Type of object such as "pedestals" or "common_mode"
   *   @param[in] run        Run number
   */
  CalibDataProxy (const std::string& calibDir, const std::string& calibType, int run);

  virtual ~CalibDataProxy ();

  /**
   *  @brief Get the correctly-typed object from the proxy.
   *
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of the correct type.
   */
  virtual boost::shared_ptr<T> getTypedImpl(PSEvt::ProxyDictI* dict,
                                            const Pds::Src& source,
                                            const std::string& key);

protected:

private:

  // helper class to make Pds::Src usable as a key
  struct _Src {
    _Src(const Pds::Src& asrc) : src(asrc) {}

    bool operator<(const _Src rhs) const {
      // ignore PID in comparison
      if (src.level() == rhs.src.level()) {
        return src.phy() < rhs.src.phy();
      }
      return src.level() < rhs.src.level();
    }

    Pds::Src src;
  };

  typedef std::map<_Src, boost::shared_ptr<T> > Src2Data;

  // Data members
  const std::string m_calibDir;
  const std::string m_calibType;
  int m_run;
  Src2Data m_data;

};

} // namespace cspad_mod

#endif // CSPAD_MOD_CALIBDATAPROXY_H
