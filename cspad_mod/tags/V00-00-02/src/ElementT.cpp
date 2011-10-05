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
  , m_sb_temp_shape(elem.sb_temp_shape())
  , m_data_shape(elem.data_shape())
  , m_extra_shape(elem._extra_shape())
{
  std::copy(elem.sb_temp(), elem.sb_temp()+Nsbtemp, m_sb_temp);
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
