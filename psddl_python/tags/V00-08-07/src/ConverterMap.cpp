//--------------------------------------------------------------------------
// File and Version Information:
//      $Id: ConverterMap.cpp 5266 2013-01-31 20:14:36Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//      Class ConverterMap...
//
//------------------------------------------------------------------------

// Python header first to suppress warnings
#include "python/Python.h"

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
#include <boost/foreach.hpp>

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
  BOOST_FOREACH(const std::type_info* type, cvt->from_cpp_types()) {
    m_from_cpp_types[type].push_back(cvt);
  }
  BOOST_FOREACH(PyTypeObject* type, cvt->from_py_types()) {
    m_from_py_types[type].push_back(cvt);
  }
  BOOST_FOREACH(PyTypeObject* type, cvt->to_py_types()) {
    m_to_py_types[type].push_back(cvt);
  }
  BOOST_FOREACH(int type, cvt->from_pds_types()) {
    m_from_pds_types[type].push_back(cvt);
  }
}

const ConverterMap::CvtList&
ConverterMap::getFromCppConverters(const std::type_info* type) const
{
  CppTypeMap::const_iterator it = m_from_cpp_types.find(type);
  if (it == m_from_cpp_types.end()) return m_emptyCvtList;
  return it->second;
}

const ConverterMap::CvtList&
ConverterMap::getFromPyConverters(const PyTypeObject* type) const
{
  PyTypeMap::const_iterator it = m_from_py_types.find(type);
  if (it == m_from_py_types.end()) return m_emptyCvtList;
  return it->second;
}

const ConverterMap::CvtList&
ConverterMap::getToPyConverters(const PyTypeObject* type) const
{
  PyTypeMap::const_iterator it = m_to_py_types.find(type);
  if (it == m_to_py_types.end()) return m_emptyCvtList;
  return it->second;
}

const ConverterMap::CvtList&
ConverterMap::getFromPdsConverters(int pdsTypeId) const
{
  PdsTypeIdMap::const_iterator it = m_from_pds_types.find(pdsTypeId);
  if (it == m_from_pds_types.end()) return m_emptyCvtList;
  return it->second;
}

} // namespace psddl_python
