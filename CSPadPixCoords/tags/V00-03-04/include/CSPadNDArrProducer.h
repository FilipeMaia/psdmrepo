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
//#include "PSCalib/CSPadCalibPars.h"

//#include "CSPadPixCoords/QuadParameters.h"
//#include "CSPadPixCoords/PixCoords2x1.h"
//#include "CSPadPixCoords/PixCoordsQuad.h"
//#include "CSPadPixCoords/PixCoordsCSPad.h"

//#include "CSPadPixCoords/Image2D.h"

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

  enum { m_n2x1         = Psana::CsPad::SectorsPerQuad     };  // 8
  enum { m_ncols2x1     = Psana::CsPad::ColumnsPerASIC     };  // 185
  enum { m_nrows2x1     = Psana::CsPad::MaxRowsPerASIC * 2 };  // 388
  enum { m_sizeOf2x1Arr = m_nrows2x1 * m_ncols2x1          };  // 185*388;

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
  void getQuadConfigPars(Env& env);
  void getCSPadConfigFromData(Event& evt);
  void procEvent(Event& evt, Env& env);
  void checkTypeImplementation();

private:

  // Data members, this is for example purposes only

  Pds::Src    m_src;
  Source      m_source;         // Data source set from config file
  std::string m_inkey; 
  std::string m_outkey;
  std::string m_outtype;
  unsigned m_print_bits;
  long     m_count;

  // Parameters form Psana::CsPad::ConfigV# object
  uint32_t m_numQuadsInConfig;
  uint32_t m_roiMask        [4];
  uint32_t m_numAsicsStored [4];

  // Parameters form Psana::CsPad::DataV# and Psana::CsPad::ElementV# object
  uint32_t m_num2x1_actual;
  uint32_t m_numQuads;
  uint32_t m_quadNumber     [4];
  uint32_t m_num2x1Stored   [4];

//-------------------
  /**
   * @brief Adds image in the event as ndarray<T,2> or Image2D<T>, depending on m_imgkey.
   * 
   * @param[in]  arr2d    pointer to the data array with image of type T.
   */

  template <typename T>
  void addNDArrInEventForType(Event& evt, ndarray<T,3>& ndarr)
  {
      shared_ptr< ndarray<T,3> > shp( new ndarray<T,3>(ndarr) );
      evt.put(shp, m_src, m_outkey);
  }

//-------------------
  /**
   * @brief Gets m_numQuadsInConfig, m_roiMask[q] and m_numAsicsStored[q] from the Psana::CsPad::ConfigV# object.
   * 
   */

  template <typename T>
  bool getQuadConfigParsForType(Env& env) {

        shared_ptr<T> config = env.configStore().get(m_source);
        if (config.get()) {
            m_numQuadsInConfig = config->numQuads();
            for (uint32_t q = 0; q < config->numQuads(); ++ q) {
              m_roiMask[q]         = config->roiMask(q);
              m_numAsicsStored[q]  = config->numAsicsStored(q);
            }
	    return true;
	}
	return false;
  }

//-------------------
  /**
   * @brief Gets m_num2x1_actual, m_numQuads, and m_quadNumber[q] from the Psana::CsPad::DataV# and ElementV# objects.
   * 
   */

  template <typename TDATA, typename TELEMENT>
  bool getCSPadConfigFromDataForType(Event& evt) {

    shared_ptr<TDATA> shp = evt.get(m_source, m_inkey, &m_src);
    if (shp.get()) {
      m_numQuads = shp->quads_shape()[0];

      m_num2x1_actual = 0;
      //cout << "m_numQuads = " << m_numQuads << "   m_quadNumber[q] = " << endl;
      for (uint32_t q = 0; q < m_numQuads; ++ q) {
        const TELEMENT& el = shp->quads(q);
        m_quadNumber[q]    = el.quad();
        m_num2x1Stored[q]  = el.data().shape()[0];
        m_num2x1_actual   += m_num2x1Stored[q];	
	//for(uint32_t sect=0; sect < m_n2x1; sect++)
	//  if( m_roiMask[q] & (1<<sect) ) m_num2x1_actual ++; 
      }
      return true;
    }
    return false;
  }

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
        const unsigned shape[] = {m_num2x1_actual, m_ncols2x1, m_nrows2x1};
        ndarray<TOUT,3> out_ndarr(shape);
        std::fill(out_ndarr.begin(), out_ndarr.end(), TOUT(0));    

        //ndarray<TOUT,3>::iterator it_out = out_ndarr.begin(); 
        TOUT* it_out = out_ndarr.data();

        for (uint32_t q = 0; q < m_numQuads; ++ q) {
            const TELEMENT& el = shp->quads(q);      
            const ndarray<const data_cspad_t,3>& quad_ndarr = el.data();

	    // pixel-by-pixel copy of quad data ndarray to output ndarray with type conversion:
            for ( ndarray<const data_cspad_t,3>::iterator it=quad_ndarr.begin(); it!=quad_ndarr.end(); ++it ) {
	      *it_out++ = (TOUT)*it;
      	    }
	}

	addNDArrInEventForType<TOUT>(evt, out_ndarr);

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
