//--------------------------------------------------------------------------
// File and Version Information:
//      $Id: ConverterMap.cpp 5266 2013-01-31 20:14:36Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//      Class ConverterMap...
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_python/ConverterMap.h"

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
  const char logger[] = "ConverterMap";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------


namespace psddl_python {

ConverterMap&
ConverterMap::instance()
{
  static ConverterMap singleton;
  return singleton;
}

void ConverterMap::addConverter(const boost::shared_ptr<Converter>& cvt)
{
  // Add mapping trom typeinfo to Converter
  m_cvtTypeMap.insert(std::make_pair(cvt->typeinfo(), cvt));

  if (cvt->pdsTypeId() >= 0) {
    m_pdsTypeMap[cvt->pdsTypeId()].push_back(cvt->typeinfo());
  }
}

boost::shared_ptr<Converter>
ConverterMap::getConverter(const std::type_info* type) const
{
  boost::shared_ptr<Converter> res;
  ConverterTypeMap::const_iterator it = m_cvtTypeMap.find(type);
  if (it != m_cvtTypeMap.end()) {
    res = it->second;
  }
  return res;
}

// Return list of type names matching Pds::TypeId::Type value
const ConverterMap::TypeInfoList&
ConverterMap::pdsTypeInfos(int pdsTypeId) const
{
  PdsTypeMap::const_iterator it = m_pdsTypeMap.find(pdsTypeId);
  if (it == m_pdsTypeMap.end()) return m_emptyTypeList;
  return it->second;
}

} // namespace psddl_python
