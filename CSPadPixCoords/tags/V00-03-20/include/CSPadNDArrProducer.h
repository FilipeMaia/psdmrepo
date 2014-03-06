#ifndef CSPADPIXCOORDS_CSPADNDARRPRODUCER_H
#define CSPADPIXCOORDS_CSPADNDARRPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadNDArrProducer.
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
#include "CSPadPixCoords/CSPadConfigPars.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "PSEvt/Source.h"
#include "psddl_psana/cspad.ddl.h"


//		---------------------
// 		-- Class Interface --
//		---------------------

using namespace std;

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief CSPadNDArrProducer produces the CSPad data ndarray<T,3> array for each event and add it to the event in psana framework.
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

class CSPadNDArrProducer : public Module {
public:

  typedef CSPadPixCoords::CSPadConfigPars CONFIG;

  const static uint32_t NQuadsMax    = 4;
  const static uint32_t N2x1         = 8;
  const static uint32_t NRows2x1     = 185;
  const static uint32_t NCols2x1     = 388;
  const static uint32_t SizeOf2x1Arr = NRows2x1 * NCols2x1;

  //enum { NQuadsMax    = Psana::CsPad::MaxQuadsPerSensor  };  // 4
  //enum { N2x1         = Psana::CsPad::SectorsPerQuad     };  // 8
  //enum { NCols2x1     = Psana::CsPad::ColumnsPerASIC     };  // 185
  //enum { NRows2x1     = Psana::CsPad::MaxRowsPerASIC * 2 };  // 388
  //enum { SizeOf2x1Arr = NRows2x1 * NCols2x1              };  // 185*388;

  // Default constructor
  CSPadNDArrProducer (const std::string& name) ;

  // Destructor
  virtual ~CSPadNDArrProducer () ;

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
  void procEvent(Event& evt, Env& env);
  void checkTypeImplementation();

private:

  // Data members, this is for example purposes only

  Pds::Src    m_src;
  Source      m_source;         // Data source set from config file
  std::string m_inkey; 
  std::string m_outkey;
  std::string m_outtype;
  bool        m_is_fullsize;
  bool        m_is_2darray;
  unsigned    m_print_bits;
  long        m_count;
  DATA_TYPE   m_dtype;
  CONFIG*     m_config;

//-------------------
  /**
   * @brief For requested m_source and m_inkey process Psana::CsPad::DataV1, or V2
   * Returns false if data is missing.
   */

  template <typename TDATA, typename TELEMENT, typename TOUT>
  bool procCSPadDataForType (Event& evt) {

      typedef int16_t data_cspad_t;

      shared_ptr<TDATA> shp = evt.get(m_source, m_inkey, &m_src); // get m_src here
      
      if (shp.get()) {

        // Create and initialize the array of the same shape as data, but for all 2x1...
        const unsigned shape[] = {m_config->num2x1StoredInData(), NRows2x1, NCols2x1};
        ndarray<TOUT,3> out_ndarr(shape);
        std::fill(out_ndarr.begin(), out_ndarr.end(), TOUT(0));    

        typename ndarray<TOUT,3>::iterator it_out = out_ndarr.begin(); 
        //TOUT* it_out = out_ndarr.data();

        uint32_t numQuads = shp->quads_shape()[0];

        for (uint32_t q = 0; q < numQuads; ++ q) {
            const TELEMENT& el = shp->quads(q);      
            const ndarray<const data_cspad_t,3>& quad_ndarr = el.data();

	    // pixel-by-pixel copy of quad data ndarray to output ndarray with type conversion:
            for ( ndarray<const data_cspad_t,3>::iterator it=quad_ndarr.begin(); it!=quad_ndarr.end(); ++it, ++it_out) {
	      *it_out = (TOUT)*it;
      	    }
	}

        if (m_is_fullsize) {
             ndarray<TOUT,3> nda_det = m_config->getCSPadPixNDArrFromNDArrShapedAsData<TOUT>(out_ndarr);

	     if (m_is_2darray) {
               //ndarray<TOUT,2> arr2d = make_ndarray(nda_det.data(), NQuadsMax*N2x1*NRows2x1, NCols2x1);
               ndarray<TOUT,2> arr2d = make_ndarray<TOUT>(NQuadsMax*N2x1*NRows2x1, NCols2x1);
               std::memcpy(arr2d.begin(), nda_det.begin(), sizeof(TOUT)*32*SizeOf2x1Arr);    	       
	       save2DArrInEvent<TOUT>(evt, m_src, m_outkey, arr2d);
	     }
	     else save3DArrInEvent<TOUT>(evt, m_src, m_outkey, nda_det);
	}
	else save3DArrInEvent<TOUT>(evt, m_src, m_outkey, out_ndarr);

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

    // proc event for available types Psana::CsPad::DataV1, or V2
    if ( procCSPadDataForType  <Psana::CsPad::DataV1, Psana::CsPad::ElementV1, TOUT> (evt) ) return true;
    if ( procCSPadDataForType  <Psana::CsPad::DataV2, Psana::CsPad::ElementV2, TOUT> (evt) ) return true;
    return false;
  }

//-------------------

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPADNDARRPRODUCER_H
