//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class acqiris...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_pds2psana/acqiris.ddl.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_pds2psana {
namespace Acqiris {

TdcDataV1::TdcDataV1(const boost::shared_ptr<const XtcType>& xtcPtr, size_t xtcSize)
  : Psana::Acqiris::TdcDataV1()
  , m_xtcObj(xtcPtr)
{
  // special constructor for TdcDataV1
  // the size of the data array is unknown and needs to be determined from XTC size
  
  size_t nItems = xtcSize / Psana::Acqiris::TdcDataV1_Item::_sizeof();
  _data.reserve(nItems);
  for (size_t i0=0; i0 != nItems; ++i0) {
    _data.push_back(pds_to_psana(xtcPtr->data(i0)));
  }
}

} // namespace Acqiris
} // namespace psddl_pds2psana
