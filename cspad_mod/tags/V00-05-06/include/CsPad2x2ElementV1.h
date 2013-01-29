#ifndef CSPAD_MOD_CSPAD2X2ELEMENTV1_H
#define CSPAD_MOD_CSPAD2X2ELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ElementV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psddl_psana/cspad2x2.ddl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace cspad_mod {

/// @addtogroup cspad_mod

/**
 *  @ingroup cspad_mod
 *
 *  @brief Implementation of Psana::CsPad2x2::ElementV1 interface for
 *  calibrated data.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class CsPad2x2ElementV1 : public Psana::CsPad2x2::ElementV1 {
public:

  enum {
    Nsbtemp = Psana::CsPad2x2::ElementV1::Nsbtemp /**< Number of the elements in _sbtemp array. */
  };

  /**
   *  Constructor takes old object and calibrated data.
   *  Data array must be allocated with new[] and will be deallocated in destructor.
   *  Common_mode array is copied, caller keeps ownership.
   */
  CsPad2x2ElementV1 (const Psana::CsPad2x2::ElementV1& elem,
      int16_t* data,
      const float* common_mode) ;

  // Destructor
  virtual ~CsPad2x2ElementV1 () ;

  /** Virtual channel number. */
  virtual uint32_t virtual_channel() const { return m_virtual_channel; }
  /** Lane number. */
  virtual uint32_t lane() const { return m_lane; }
  virtual uint32_t tid() const { return m_tid; }
  virtual uint32_t acq_count() const { return m_acq_count; }
  virtual uint32_t op_code() const { return m_op_code; }
  /** Quadrant number. */
  virtual uint32_t quad() const { return m_quad; }
  virtual uint32_t seq_count() const { return m_seq_count; }
  virtual uint32_t ticks() const { return m_ticks; }
  virtual uint32_t fiducials() const { return m_fiducials; }
  virtual ndarray<uint16_t, 1> sb_temp() const { return make_ndarray(m_sb_temp, Nsbtemp); }
  virtual uint32_t frame_type() const { return m_frame_type; }
  virtual ndarray<int16_t, 3> data() const { return ndarray<int16_t, 3>(m_data, m_data_shape); }
  virtual float common_mode(uint32_t section) const { return m_common_mode[section]; }

protected:

private:

  uint32_t m_virtual_channel;
  uint32_t m_lane;
  uint32_t m_tid;
  uint32_t m_acq_count;
  uint32_t m_op_code;
  uint32_t m_quad;
  uint32_t m_seq_count;
  uint32_t m_ticks;
  uint32_t m_fiducials;
  mutable uint16_t m_sb_temp[Nsbtemp];
  uint32_t m_frame_type;
  int16_t* m_data;
  float m_common_mode[2];
  unsigned m_data_shape[3];

  // Copy constructor and assignment are disabled by default
  CsPad2x2ElementV1 ( const CsPad2x2ElementV1& ) ;
  CsPad2x2ElementV1& operator = ( const CsPad2x2ElementV1& ) ;

};

} // namespace cspad_mod

#endif // CSPAD_MOD_CSPAD2X2ELEMENTV1_H
