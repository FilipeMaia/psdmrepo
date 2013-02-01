//--------------------------------------------------------------------------
// File and Version Information:
//      $Id: GetterMap.cpp 5266 2013-01-31 20:14:36Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//      Class GetterMap...
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_python/GetterMap.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <cstring>
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "GetterMap";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------


namespace psddl_python {

GetterMap&
GetterMap::instance()
{
  static GetterMap singleton;
  return singleton;
}


void GetterMap::printTables()
{
  WithMsgLog(logger, info, str) {
    str << "*** getterMap:\n";
    for (GetterNameMap::const_iterator it = m_getterNameMap.begin(); it != m_getterNameMap.end(); ++ it) {
      str << it->first << " -> " << it->second << '\n';
    }

    str << "*** templates:\n";
    for (TemplateMap::const_iterator it = m_templateMap.begin(); it != m_templateMap.end(); ++ it) {
      str << it->first << " ->";
      const NameList& names = it->second;
      for (NameList::const_iterator nit = names.begin(); nit != names.end(); ++ nit) {
        str << ' ' << *nit;
      }
      str << '\n';
    }
  }
}

void GetterMap::addGetter(const boost::shared_ptr<Getter>& getter)
{
  // Add mapping trom typeinfo to getter
  m_getterTypeMap.insert(std::make_pair(std::tr1::cref(getter->typeinfo()), getter));

  // Add the mapping for the original name (with version).
  // E.g. "BldDataEBeamV0" or "TdcDataV1_Item"
  m_getterNameMap.insert(std::make_pair(getter->getTypeName(), getter));

  const int version = getter->getVersion();
  if (version == -1) {
    return;
  }

  // This getter has a version. See if we find it embedded in the type name.
  std::string typeName(getter->getTypeName());
  std::string vpart = "V";
  vpart += boost::lexical_cast<std::string>(version);
  const size_t vpos = typeName.rfind(vpart);
  if (vpos == std::string::npos) {
    if (version != 0) {
      MsgLog(logger, error, "GetterMap::addGetter(" << typeName  << "): version is "
          << version << " but no V" << version << " found in class name");
    }
    return;
  }

  // strip version from type name, store original name for this name
  typeName.erase(vpos, vpart.size());
  m_templateMap[typeName].push_back(getter->getTypeName());
}

// Return template for versionless type name, if one exists.
const GetterMap::NameList&
GetterMap::getTemplate(const std::string& typeName) const
{
  TemplateMap::const_iterator it = m_templateMap.find(typeName);
  if (it == m_templateMap.end()) return m_emptyNameList;
  return it->second;
}

boost::shared_ptr<Getter>
GetterMap::getGetter(const std::type_info& type) const
{
  boost::shared_ptr<Getter> res;
  GetterTypeMap::const_iterator it = m_getterTypeMap.find(std::tr1::cref(type));
  if (it != m_getterTypeMap.end()) {
    res = it->second;
  }
  return res;
}

boost::shared_ptr<Getter>
GetterMap::getGetter(const std::string& typeName) const
{
  boost::shared_ptr<Getter> res;
  GetterNameMap::const_iterator it = m_getterNameMap.find(typeName);
  if (it != m_getterNameMap.end()) {
    res = it->second;
  }
  return res;
}

} // namespace psddl_python
