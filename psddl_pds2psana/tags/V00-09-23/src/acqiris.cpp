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
  size_t nItems = xtcSize / Pds::Acqiris::TdcDataV1_Item::_sizeof();

  typedef ndarray<Psana::Acqiris::TdcDataV1_Item, 1> NDArray;
  typedef ndarray<const Pds::Acqiris::TdcDataV1_Item, 1> XtcNDArray;
  const XtcNDArray& xtc_ndarr = xtcPtr->data(nItems);
  _data_ndarray_storage_ = NDArray(xtc_ndarr.shape());
  NDArray::iterator out = _data_ndarray_storage_.begin();
  for (unsigned i = 0; i != nItems; ++ i, ++ out) {
    *out = pds_to_psana(xtc_ndarr[i]);
  }
}

} // namespace Acqiris
} // namespace psddl_pds2psana
