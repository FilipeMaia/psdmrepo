#ifndef CSPADPIXCOORDS_CSPAD2X2NDARRPRODUCER_H
#define CSPADPIXCOORDS_CSPAD2X2NDARRPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2NDArrProducer.
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
#include "CSPadPixCoords/CSPad2x2ConfigPars.h"

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
 *  @brief CSPad2x2NDArrProducer produces the CSPad data ndarray<T,3> array for each event and add it to the event in psana framework.
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

class CSPad2x2NDArrProducer : public Module {
public:

  typedef CSPadPixCoords::CSPad2x2ConfigPars CONFIG;

  const static uint32_t NRows2x1 = 185; 
  const static uint32_t NCols2x1 = 388; 
  const static uint32_t N2x1     = 2; 

  // Default constructor
  CSPad2x2NDArrProducer (const std::string& name) ;

  // Destructor
  virtual ~CSPad2x2NDArrProducer () ;

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
  void getConfigParameters(Event& evt, Env& env);
  void procEvent(Event& evt, Env& env);
  void checkTypeImplementation();

private:

  // Data members, this is for example purposes only

  Pds::Src    m_src;
  Source      m_source;         // Data source set from config file
  std::string m_inkey; 
  std::string m_outkey;
  std::string m_outtype;
  unsigned    m_print_bits;
  long        m_count;

  DATA_TYPE   m_dtype;
  CONFIG*     m_config;

  // Parameters form Psana::CsPad2x2::ConfigV# object
  uint32_t m_roiMask;
  uint32_t m_numAsicsStored;

  // Parameters form Psana::CsPad2x2::DataV# and Psana::CsPad::ElementV# object
  //uint32_t m_num2x1_actual;

//-------------------
  /**
   * @brief For requested m_source and m_inkey process Psana::CsPad::DataV1, or V2
   * Returns false if data is missing.
   */

  template <typename TELEMENT, typename TOUT>
  bool procCSPad2x2DataForType (Event& evt) {

      typedef int16_t data_cspad_t;

      shared_ptr<TELEMENT> elem = evt.get(m_source, m_inkey, &m_src); // get m_src here
      
      if (elem) {

        const ndarray<const data_cspad_t,3>& data_ndarr = elem->data();
 
        // Create and initialize the array of the same shape as data, but for all 2x1...
        //const unsigned shape[] = {NRows2x1, NCols2x1, N2x1};
        const unsigned* shape = data_ndarr.shape();
        ndarray<TOUT,3> out_ndarr(shape);
        //std::fill(out_ndarr.begin(), out_ndarr.end(), TOUT(0));    

        typename ndarray<TOUT,3>::iterator it_out = out_ndarr.begin(); 

	// pixel-by-pixel copy of quad data ndarray to output ndarray with type conversion:
        for ( ndarray<const data_cspad_t,3>::iterator it=data_ndarr.begin(); it!=data_ndarr.end(); ++it ) {
	  *it_out++ = (TOUT)*it;
      	}

	save3DArrInEvent<TOUT>(evt, m_src, m_outkey, out_ndarr);

        return true;
      } // if (shp.get())
      return false;
  }

//--------------------
  /**
   * @brief Process event for requested output type TOUT
   * Returns false if data is missing.
   */

  template <typename TOUT>
  bool procEventForOutputType (Event& evt) {

    // proc event for available type Psana::CsPad2x2::ElementV1 
      if ( procCSPad2x2DataForType  <Psana::CsPad2x2::ElementV1, TOUT> (evt) ) return true;
    //if ( procCSPad2x2DataForType  <Psana::CsPad2x2::ElementV2, TOUT> (evt) ) return true;
    return false;
  }

//-------------------

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPAD2X2NDARRPRODUCER_H
