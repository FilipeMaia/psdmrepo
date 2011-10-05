//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MiniElementV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/MiniElementV1.h"

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

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
MiniElementV1::MiniElementV1 (const Psana::CsPad::MiniElementV1& elem,
    const int16_t* data,
    const float* common_mode)
  : Psana::CsPad::MiniElementV1()
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
  , m_common_mode()
  , m_sb_temp_shape(elem.sb_temp_shape())
  , m_data_shape(elem.data_shape())
{
  std::copy(elem.sb_temp(), elem.sb_temp()+Nsbtemp, m_sb_temp);
  std::copy(common_mode, common_mode+2, m_common_mode);
}

//--------------
// Destructor --
//--------------
MiniElementV1::~MiniElementV1 ()
{
  delete [] m_data;
}



} // namespace cspad_mod
