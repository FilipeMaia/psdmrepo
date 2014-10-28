#ifndef CSPADPIXCOORDS_CSPAD2X2NDARRRESHAPE_H
#define CSPADPIXCOORDS_CSPAD2X2NDARRRESHAPE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2NDArrReshape.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CSPadPixCoords/GlobalMethods.h"
//#include "CSPadPixCoords/CSPad2x2ConfigPars.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "PSEvt/Source.h"
#include "psddl_psana/cspad2x2.ddl.h"


//		---------------------
// 		-- Class Interface --
//		---------------------

using namespace std;

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief CSPad2x2NDArrReshape produces the CSPad data ndarray<T,3> array for each event and add it to the event in psana framework.
 *
 *  1) get cspad configuration and data from the event,
 *  2) produce the CSPad data ndarray<T,3> array,
 *  3) save array in the event for further modules.
 *
 *  This class should not be used directly in the code of users modules. 
 *  Instead, it should be added as a module in the psana.cfg file with appropriate parameters.
 *  Then, the produced Image2D object can be extracted from event and used in other modules.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see PixCoords2x1, PixCoordsQuad, PixCoordsCSPad, CSPadImageGetTest
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */


//struct KeysInOutStatus {
//  std::string key_in;
//  std::string key_out;
//  bool is_found;
//};


class CSPad2x2NDArrReshape : public Module {
public:

  //  typedef CSPadPixCoords::CSPad2x2ConfigPars CONFIG;

  const static uint32_t NRows2x1 = 185; 
  const static uint32_t NCols2x1 = 388; 
  const static uint32_t N2x1     = 2; 

  // Default constructor
  CSPad2x2NDArrReshape (const std::string& name) ;

  // Destructor
  virtual ~CSPad2x2NDArrReshape () ;

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


protected:

  void printInputParameters();
  bool procEvent(Event& evt);
  bool procCalib(Env& env);
  void put_keys_in_vectors();
  void print_in_out_keys();

private:

  Pds::Src    m_src;
  Source      m_source;
  std::string m_keys_in; 
  std::string m_keys_out;
  unsigned    m_print_bits;
  unsigned    m_count_evt;
  unsigned    m_count_clb;
  unsigned    m_count_msg;

  std::vector<std::string>     v_keys_in;
  std::vector<std::string>     v_keys_out;

//-------------------
  /**
   * @brief Re-shape and save ndarray for CSPAD2x2 
   * Returns false if data is missing.
   */
  template <typename T>
  ndarray<T,3> reshape(const T* data_in) {

      //const ndarray<const T,3>& data_ndarr = *shp_const.get();
      ndarray<T,3> out_ndarr = make_ndarray<T>(N2x1, NRows2x1, NCols2x1);

      for(size_t r=0; r<NRows2x1; ++r) {
      for(size_t c=0; c<NCols2x1; ++c) {
      for(size_t n=0; n<N2x1; ++n) out_ndarr[n][r][c] = *data_in++;
      }
      }

      return out_ndarr;
  }


//-------------------
  /**
   * @brief Process event for requested in/output type and re-shapes CSPAD2x2 from [185,388,2] to [2,185,388]
   * Returns false if data is missing.
   */

  template <typename T>
  bool procEventForType (Event& evt, size_t i) {

      // for CONST
      shared_ptr< ndarray<const T,3> > shp_const = evt.get(m_source, v_keys_in[i], &m_src); // get m_src here      
      if (shp_const) { save3DArrInEvent<T>(evt, m_src, v_keys_out[i], reshape<T>(shp_const->data())); return true; }


      // for NON-CONST
      shared_ptr< ndarray<T,3> > shp = evt.get(m_source, v_keys_in[i], &m_src); // get m_src here      
      if (shp) { save3DArrInEvent<T>(evt, m_src, v_keys_out[i], reshape<T>(shp->data())); return true; }

      return false;
  }


//-------------------
  /**
   * @brief Process event for requested in/output type and re-shapes CSPAD2x2 from [185,388,2] to [2,185,388]
   * Returns false if data is missing.
   */

  template <typename T>
  bool procCalibForType (Env& env, size_t i) {

      // for CONST
      shared_ptr< ndarray<const T,3> > shp_const = env.calibStore().get(m_source, &m_src, v_keys_in[i]); // get m_src here      
      if (shp_const) { save3DArrayInEnv<T>(env, m_src, v_keys_out[i], reshape<T>(shp_const->data())); return true; }


      // for NON-CONST
      shared_ptr< ndarray<T,3> > shp = env.calibStore().get(m_source, &m_src, v_keys_in[i]); // get m_src here      
      if (shp) { save3DArrayInEnv<T>(env, m_src, v_keys_out[i], reshape<T>(shp->data())); return true; }

      return false;
  }


//-------------------

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPAD2X2NDARRRESHAPE_H
