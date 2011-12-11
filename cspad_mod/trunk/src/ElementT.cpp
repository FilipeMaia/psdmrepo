//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ElementT...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/ElementT.h"

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

namespace cspad_mod {

//----------------
// Constructors --
//----------------
template <typename ElemType>
ElementT<ElemType>::ElementT (const ElemType& elem,
    const int16_t* data,
    const float* common_mode)
  : ElemType()
  , m_virtual_channel(elem.virtual_channel())
  , m_lane(elem.lane())
  , m_tid(elem.tid())
  , m_acq_count(elem.acq_count())
  , m_op_code(elem.op_code())
  , m_quad(elem.quad())
  , m_seq_count(elem.seq_count())
  , m_ticks(elem.ticks())
  , m_fiducials(elem.fiducials())
  , m_sb_temp()
  , m_frame_type(elem.frame_type())
  , m_data(data)
  , m_sectionMask(elem.sectionMask())
  , m_common_mode()
{
  // copy sb_temp array
  const ndarray<uint16_t, 1>& sb_temp = elem.sb_temp();
  std::copy(sb_temp.begin(), sb_temp.end(), m_sb_temp);

  // copy data shape
  const ndarray<int16_t, 3>& edata = elem.data();
  std::copy(edata.shape(), edata.shape()+3, m_data_shape);

  // copy common_mode array
  int nsect = m_data_shape[0];
  std::copy(common_mode, common_mode+nsect, m_common_mode);
}

//--------------
// Destructor --
//--------------
template <typename ElemType>
ElementT<ElemType>::~ElementT ()
{
  delete [] m_data;
}


// Explicit instantiation
template class ElementT<Psana::CsPad::ElementV1>;
template class ElementT<Psana::CsPad::ElementV2>;

} // namespace cspad_mod
