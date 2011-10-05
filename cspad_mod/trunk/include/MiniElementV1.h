#ifndef CSPAD_MOD_MINIELEMENTV1_H
#define CSPAD_MOD_MINIELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MiniElementV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psddl_psana/cspad.ddl.h"

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
 *  @brief Implementation of Psana::CsPad::MiniElementV1 interface for
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

class MiniElementV1 : public Psana::CsPad::MiniElementV1 {
public:

  enum {
    Nsbtemp = Psana::CsPad::MiniElementV1::Nsbtemp /**< Number of the elements in _sbtemp array. */
  };

  /**
   *  Constructor takes old object and calibrated data.
   *  Data array must be allocated with new[] and will be deallocated in destructor.
   *  Common_mode array is copied, caller keeps ownership.
   */
  MiniElementV1 (const Psana::CsPad::MiniElementV1& elem,
      const int16_t* data,
      const float* common_mode) ;

  // Destructor
  virtual ~MiniElementV1 () ;

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
  virtual const uint16_t* sb_temp() const { return m_sb_temp; }
  virtual uint32_t frame_type() const { return m_frame_type; }
  virtual const int16_t* data() const { return m_data; }
  virtual float common_mode(uint32_t section) const { return m_common_mode[section]; }
  /** Method which returns the shape (dimensions) of the data returned by sb_temp() method. */
  virtual std::vector<int> sb_temp_shape() const { return m_sb_temp_shape; }
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  virtual std::vector<int> data_shape() const { return m_data_shape; }

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
  uint16_t m_sb_temp[Nsbtemp];
  uint32_t m_frame_type;
  const int16_t* m_data;
  float m_common_mode[2];
  std::vector<int> m_sb_temp_shape;
  std::vector<int> m_data_shape;

  // Copy constructor and assignment are disabled by default
  MiniElementV1 ( const MiniElementV1& ) ;
  MiniElementV1& operator = ( const MiniElementV1& ) ;

};

} // namespace cspad_mod

#endif // CSPAD_MOD_MINIELEMENTV1_H
