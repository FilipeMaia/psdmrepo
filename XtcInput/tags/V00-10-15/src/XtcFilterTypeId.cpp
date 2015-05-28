//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilterTypeId...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcFilterTypeId.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  bool contains(const XtcInput::XtcFilterTypeId::IdList& idList, Pds::TypeId::Type id) {
    return std::binary_search(idList.begin(), idList.end(), id);
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcFilterTypeId::XtcFilterTypeId(const IdList& keep, const IdList& discard)
  : m_keep(keep)
  , m_discard(discard)
{
  // sort them for faster access
  std::sort(m_keep.begin(), m_keep.end());
  std::sort(m_discard.begin(), m_discard.end());
}


bool
XtcFilterTypeId::operator()(const Pds::Xtc* input) const
{
  Pds::TypeId::Type id = input->contains.id();

  if (not m_keep.empty() and not ::contains(m_keep, id)) {
    // not in the keep list
    return false;
  }
  if (not m_discard.empty() and ::contains(m_discard, id)) {
    // in discard list
    return false;
  }
  return true;
}


} // namespace XtcInput
