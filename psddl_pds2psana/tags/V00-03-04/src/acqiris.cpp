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
  size_t nItems = xtcSize / PsddlPds::Acqiris::TdcDataV1_Item::_sizeof();

  typedef ndarray<PsddlPds::Acqiris::TdcDataV1_Item, 1> XtcNDArray;
  const XtcNDArray& xtc_ndarr = xtcPtr->data();
  _data_ndarray_storage_.reserve(nItems);
  for (unsigned i = 0; i != nItems; ++ i) {
    _data_ndarray_storage_.push_back(pds_to_psana(xtc_ndarr[i]));
  }
  _data_ndarray_shape_[0] = nItems;
}

} // namespace Acqiris
} // namespace psddl_pds2psana
