//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AliasMap...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSEvt/AliasMap.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/EventKey.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEvt {

//----------------
// Constructors --
//----------------
AliasMap::AliasMap ()
  : m_alias2src()
  , m_src2alias()
{
}

//--------------
// Destructor --
//--------------
AliasMap::~AliasMap ()
{
}

// Add one more alias to the map
void
AliasMap::add(const std::string& alias, const Pds::Src& src)
{
  m_alias2src.insert(std::make_pair(alias, src));
  m_src2alias.insert(std::make_pair(src, alias));
}

// remove all aliases
void
AliasMap::clear()
{
  m_alias2src.clear();
  m_src2alias.clear();
}

// Find matching Src for given alias name.
Pds::Src
AliasMap::src(const std::string& alias) const
{
  Pds::Src res;
  std::map<std::string, Pds::Src>::const_iterator itr = m_alias2src.find(alias);
  if (itr != m_alias2src.end()) res = itr->second;
  return res;
}

// Find matching alias name for given Src.
std::string
AliasMap::alias(const Pds::Src& src) const
{
  std::string res;
  std::map<Pds::Src, std::string, SrcCmp>::const_iterator itr = m_src2alias.find(src);
  if (itr != m_src2alias.end()) res = itr->second;
  return res;
}

bool
AliasMap::SrcCmp::operator()(const Pds::Src& lhs, const Pds::Src& rhs) const
{
  return PSEvt::cmpPdsSrc(lhs, rhs) < 0;
}

} // namespace PSEvt
