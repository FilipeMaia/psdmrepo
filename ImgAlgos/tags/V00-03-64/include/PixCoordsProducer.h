#ifndef IMGALGOS_PIXCOORDSPRODUCER_H
#define IMGALGOS_PIXCOORDSPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsProducer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdint.h> // uint8_t, uint32_t, etc.

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSCalib/GeometryAccess.h"
#include "ndarray/ndarray.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Example module class for psana
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class PixCoordsProducer : public Module {
public:

  typedef double coord_t;
  typedef double area_t;
  typedef int mask_t;
  typedef unsigned coord_index_t;

  // Default constructor
  PixCoordsProducer (const std::string& name) ;

  // Destructor
  virtual ~PixCoordsProducer () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called at the end of the calibration cycle
  virtual void endCalibCycle(Event& evt, Env& env);

  /// Method which is called at the end of the run
  virtual void endRun(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

  void printInputParameters();

protected:

private:

  // Data members, this is for example purposes only
  
  Source      m_str_src;          // Data source set from config file. Ex.: "DetInfo(:Cspad)"
  std::string m_group;            // Ex.: "CsPad::CalibV1"
  Pds::Src    m_src;              // source address of the data object
  std::string m_key_out_x;        // Key for output pixel x-coordinate array
  std::string m_key_out_y;        // Key for output pixel y-coordinate array
  std::string m_key_out_z;        // Key for output pixel z-coordinate array
  std::string m_key_out_a;        // Key for output pixel area array
  std::string m_key_out_m;        // Key for output pixel mask array
  std::string m_key_out_ix;       // Key for output pixel index x-coordinate array
  std::string m_key_out_iy;       // Key for output pixel index y-coordinate array
  std::string m_key_fname;        // Key for the name of geometry calibration file used in evaluation transmitted as ndarray<char,1>
  std::string m_key_gfname;       // Key for the name of geometry calibration file used in evaluation transmitted as string
  int         m_x0_off_pix;       // x0 origin offset in number of pixels before evaluation of indexes
  int         m_y0_off_pix;       // y0 origin offset in number of pixels before evaluation of indexes
  unsigned    m_mask_bits;        // mask control bits
  unsigned    m_print_bits;       // verbosity

  long m_count_run;
  long m_count_event;
  long m_count_calibcycle;
  long m_count_warnings;
  long m_count_cfg;
  long m_count_clb;

  std::string m_config_vers;
  std::string m_fname;

  PSCalib::GeometryAccess* m_geometry;

  const coord_t* m_pixX;
  const coord_t* m_pixY;
  const coord_t* m_pixZ;
  const area_t*  m_pixA;
  const mask_t*  m_pixM;
  const coord_index_t* m_pixIX;
  const coord_index_t* m_pixIY;
  unsigned       m_size;
  unsigned       m_size_a;
  unsigned       m_size_m;
  unsigned       m_size_ind;

  ndarray<const coord_t,1>       m_ndaX;
  ndarray<const coord_t,1>       m_ndaY;
  ndarray<const coord_t,1>       m_ndaZ;
  ndarray<const area_t,1>        m_ndaA;
  ndarray<const mask_t,1>        m_ndaM;
  ndarray<const coord_index_t,1> m_ndaIX;
  ndarray<const coord_index_t,1> m_ndaIY;
  ndarray<const uint8_t,1>       m_ndafn;

  /// Regular check for available calibration parameters
  void checkCalibPars(Event& evt, Env& env);

  /// Method is used to get m_src
  bool getConfigPars(Env& env);

  /// Method retreives calibration parameters
  bool getCalibPars(Event& evt, Env& env);

  /// Method saves pixel coordinate arrays in the event store 
  void savePixCoordsInEvent(Event& evt);

  /// Method saves pixel coordinate arrays in the calibStore 
  void savePixCoordsInCalibStore(Env& env);

//-------------------
  /**
   * @brief Gets m_src from object like Psana::CsPad::ConfigV#.
   */
  template <typename T>
  bool getConfigParsForType(PSEnv::Env& env) {

        boost::shared_ptr<T> config = env.configStore().get(m_str_src, &m_src);
        if (config.get()) {
            ++ m_count_cfg;
            return true;
        }
        return false;
  }

//-------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_PIXCOORDSPRODUCER_H
