#ifndef CONFIGSVC_CONFIGSVCPYHELPER_H
#define CONFIGSVC_CONFIGSVCPYHELPER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigSvcPyHelper.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigSvc/ConfigSvc.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ConfigSvc {

/// @addtogroup ConfigSvc

/**
 *  @ingroup ConfigSvc
 *
 *  @brief Python-friendly wrapper for ConfigSvc class.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ConfigSvcPyHelper {
public:

  // Default constructor
  ConfigSvcPyHelper () {}

  // These methods throw exception if parameter is not defined
  int getBool(const std::string& section, const std::string& param) const {
    return bool(m_config.get(section, param));
  }
  long getInt(const std::string& section, const std::string& param) const {
    return long(m_config.get(section, param));
  }
  double getDouble(const std::string& section, const std::string& param) const {
    return double(m_config.get(section, param));
  }
  std::string getStr(const std::string& section, const std::string& param) const {
    return m_config.getStr(section, param);
  }

  // These methods return default value if parameter is not defined
  int getBool(const std::string& section, const std::string& param, int def) const {
    return bool(m_config.get(section, param, bool(def)));
  }
  long getInt(const std::string& section, const std::string& param, long def) const {
    return long(m_config.get(section, param, def));
  }
  double getDouble(const std::string& section, const std::string& param, double def) const {
    return double(m_config.get(section, param, def));
  }
  std::string getStr(const std::string& section, const std::string& param, const std::string& def) const {
    return m_config.getStr(section, param, def);
  }

  // These methods throw exception if parameter is not defined
  std::vector<int> getBoolList(const std::string& section, const std::string& param) const {
    return m_config.getList(section, param);
  }
  std::vector<long> getIntList(const std::string& section, const std::string& param) const {
    return m_config.getList(section, param);
  }
  std::vector<double> getDoubleList(const std::string& section, const std::string& param) const {
    return m_config.getList(section, param);
  }
  std::vector<std::string> getStrList(const std::string& section, const std::string& param) const {
    return m_config.getList(section, param);
  }

  // These methods return default value if parameter is not defined
  std::vector<int> getBoolList(const std::string& section, const std::string& param, const std::vector<int>& def) const {
    return m_config.getList(section, param, def);
  }
  std::vector<long> getIntList(const std::string& section, const std::string& param, const std::vector<long>& def) const {
    return m_config.getList(section, param, def);
  }
  std::vector<double> getDoubleList(const std::string& section, const std::string& param, const std::vector<double>& def) const {
    return m_config.getList(section, param, def);
  }
  std::vector<std::string> getStrList(const std::string& section, const std::string& param, const std::vector<std::string>& def) const {
    return m_config.getList(section, param, def);
  }

  void put(const std::string& section, const std::string& param, const std::string& value) {
    m_config.put(section, param, value);
  }

protected:

private:

  ConfigSvc::ConfigSvc m_config;
  
};

} // namespace ConfigSvc

#endif // CONFIGSVC_CONFIGSVCPYHELPER_H
