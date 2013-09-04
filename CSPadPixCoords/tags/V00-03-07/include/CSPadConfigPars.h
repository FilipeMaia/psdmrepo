#ifndef CSPADPIXCOORDS_CSPADCONFIGPARS_H
#define CSPADPIXCOORDS_CSPADCONFIGPARS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadImageProducer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
//#include "psana/Module.h"
#include "MsgLogger/MsgLogger.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//#include "CSPadPixCoords/Image2D.h"
//#include "CSPadPixCoords/GlobalMethods.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include <boost/shared_ptr.hpp>

#include "PSEvt/Source.h"
#include "PSEnv/Env.h"
#include "PSEvt/Event.h"
#include "psddl_psana/cspad.ddl.h"

//#include "CSPadPixCoords/PixCoords2x1V2.h"
#include "ndarray/ndarray.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

using namespace std;

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief CSPadConfigPars is a store for CSPAD configuration parameters.
 *
 *  <h1>Interface Description</h1>
 *   
 *  @li  Include and typedef
 *  @code
 *  #include "CSPadPixCoords/CSPadConfigPars.h"
 *  typedef CSPadPixCoords::CSPadConfigPars CONFIG;
 *  @endcode
 *  
 *  @li  Instatiation\n
 *  Default constructor; may be used if all 32 2x1 are in working condition and are presented in data
 *  @code
 *      CONFIG* config = new CONFIG (); 
 *  @endcode
 *  \n
 *  Constructor from specified data source, defined in psana. Prefered use case of this class objects.
 *  @code
 *      CONFIG* config = new CONFIG ("DetInfo(CxiDs1.0:Cspad.0)");
 *  @endcode
 *  In this case configuration parameters need to be defined in psana module overloding methods like
 *  @code
 *      virtual void beginRun(PSEvt::Event& evt, PSEnv::Env& env); 
 *      virtual void beginCalibCycle(PSEvt::Event& evt, PSEnv::Env& env); 
 *      // or
 *      virtual void event(PSEvt::Event& evt, PSEnv::Env& env); 
 *  @endcode
 *  from the PSEvt::Event and PSEnv::Env variables using method
 *  @code
 *      config -> setCSPadConfigPars (evt, env); 
 *      // or its separate sub-methods
 *      config -> setCSPadConfigParsFromEnv (env); 
 *      config -> setCSPadConfigParsFromEvent (evt); 
 *  @endcode
 *  \n
 *  Constructor from explicitly defined configuration parameters. It is not recommended to use. Can be used for stable non-complete configuration of the detector or for test purpose.
 *  @code
 *      uint32_t numQuads     = 4;                
 *      uint32_t quadNumber[] = {0, 1, 2, 3};      
 *      uint32_t roiMask[]    = {0377, 0377, 0377, 0377};
 *      CONFIG* config = new CSPadConfigPars::CSPadConfigPars ( numQuads, quadNumber, roiMask );
 *  @endcode
 *  
 *  
 * @li  Print current configuration parameters
 *  @code
 *      config -> printCSPadConfigPars();
 *  @endcode
 * 
 *  
 * @li  Get methods for member data  \n
 *  @code
 *      uint32_t numQuads = config -> numberOfQuadsStored ();
 *      uint32_t qNum     = config -> quadNumber (iq);  
 *      uint32_t n2x1     = config -> numberOf2x1Stored (iq);
 *      uint32_t quadMask = config -> roiMask (iq);
 *  @endcode
 *  
 *  
 * @li  Conversion between entire CSPAD pixel array shaped as (32,185,388) and data array shaped as (N,185,388), where N<=32 \n
 * Conversion from (32,185,388) to (N,185,388)
 * @code
 *     ndarray<double,3> nda_det  = make_ndarray (p_pix_arr_det, N2X1_IN_DET, ROWS2X1, COLS2X1);
 *     ndarray<double,3> nda_data = cspad_configpars -> getCSPadPixNDArrShapedAsData<double> ( nda_det );
 * @endcode
 *  \n 
 * Conversion from (N,185,388) to (32,185,388)
 * @code
 *     ndarray<double,3> nda_data = make_ndarray (p_pix_arr_data, N2X1_IN_DATA, ROWS2X1, COLS2X1); 
 *     ndarray<double,3> nda_det  = cspad_configpars -> getCSPadPixNDArrFromNDArrShapedAsData<double> ( nda_data ); 
 * @endcode
 * 
 *  @version \$Id:$
 *
 *  @author Mikhail S. Dubrovin
 */

class CSPadConfigPars {
public:

  //typedef CSPadPixCoords::PixCoords2x1V2 PC2X1;
  //typedef PixCoords2x1V2 PC2X1;

  const static unsigned NQuadsMax = 4;

  /// @brief Default constructor
  CSPadConfigPars () ;

 /**
   * @brief Constructor using specified source as input parameter
   *  @param[in] source         (def.= "DetInfo(CxiDs1.0:Cspad.0)")                
   */
  CSPadConfigPars (PSEvt::Source source) ;

 /**
   *  @brief Constructor with explicit defenition of configuration parameters
   *  @param[in] numQuads         (def.= 4)                
   *  @param[in] quadNumber[]     (def.= {0,1,2,3})      
   *  @param[in] roiMask[]        (def.= {0377,0377,0377,0377}, or in decimal {255,255,255,255})
   */
   CSPadConfigPars ( uint32_t numQuads,     // 4
  		     uint32_t quadNumber[], // {0,1,2,3},
  		     uint32_t roiMask[]     // {0377,0377,0377,0377} // in octal or in decimal: {255,255,255,255}
		    ) ;

  /// @brief Destructor
  virtual ~CSPadConfigPars () ;

 /**
   *  @brief Sets CSPAD configuration parameters
   *  @param[in] evt pointer to the event store
   *  @param[in] env pointer to the environment store 
   */
  void setCSPadConfigPars(PSEvt::Event& evt, PSEnv::Env& env);

  /// Sets CSPAD configuration parameters to their default values
  void setCSPadConfigParsDefault();

  /// Prints CSPAD configuration parameters
  void printCSPadConfigPars();

  /// Returns the string with package name for logger. 
  std::string name() { return "pkg: CSPadPixCoords"; }  

  /// Returns the number of turned on (1) bits (2x1s) in the binary mask (def.= 8)
  uint32_t getNum2x1InMask(uint32_t mask);

  /// Returns the number of quads stored in data array (def.= 4)
  uint32_t numberOfQuadsStored () { return m_numQuads; }  

  /// Returns the quad number for its index iq in the range [0,m_numQuads]
  uint32_t quadNumber          (int iq) { return m_quadNumber[iq]; }

  /// Returns the number of 2x1s in the quad with index iq (def.= 8)
  uint32_t numberOf2x1Stored   (int iq) { return m_num2x1Stored[iq]; }

  /// Returns the mask for 2x1s in the quad with index iq (def.=0377 for all 8 2x1s)
  uint32_t roiMask             (int iq) { return m_roiMask[iq]; }

  /// Returns the number of 2x1s available in the CSPAD detector (def.= 32)
  uint32_t num2x1StoredInData  ()       { return m_num2x1StoredInData; }

  //uint32_t numberOfAsicsStored (int iq) { return m_numAsicsStored[iq]; }

//--------------------
 
protected:

  /// part of the setCSPadConfigPars(PSEvt::Event& evt, PSEnv::Env& env)
  /// @param[in] env pointer to the environment store 
  void setCSPadConfigParsFromEnv(PSEnv::Env& env);

  /// part of the setCSPadConfigPars(PSEvt::Event& evt, PSEnv::Env& env)
  /// @param[in] evt pointer to the event store
  void setCSPadConfigParsFromEvent(PSEvt::Event& evt);

private:

  PSEvt::Source m_source;        /// member data for data source set from config file
  Pds::Src      m_src;           /// source address as Pds::Src 
 
  // Parameters form Psana::CsPad::ConfigV# object
  uint32_t m_roiMask        [4]; /// mask for turrned on/off (1/0) 2x1s
  //uint32_t m_numAsicsStored [4];

  // Parameters form Psana::CsPad::DataV# and Psana::CsPad::ElementV# object
  uint32_t m_numQuads;           /// number of quads in data
  uint32_t m_quadNumber     [4]; /// quad numbers for index in the range [0, m_numQuads]
  uint32_t m_num2x1Stored   [4]; /// number of 2x1s in quad by index
  uint32_t m_num2x1StoredInData; /// number of 2x1s in CSPAD stored in data
 
//-------------------
  /**
   * @brief Gets m_src, m_roiMask[q] and m_num2x1Stored[q] from the Psana::CsPad::ConfigV# object.
   */
  template <typename T>
  bool getQuadConfigParsForType(PSEnv::Env& env) {

        boost::shared_ptr<T> config = env.configStore().get(m_source, &m_src);
        if (config.get()) {
            for (uint32_t q = 0; q < NQuadsMax; ++ q) {
              m_roiMask[q]         = config->roiMask(q);
              //m_numAsicsStored[q]  = config->numAsicsStored(q);
            }
	    return true;
	}
	return false;
  }

//-------------------
  /**
   * @brief Gets m_numQuads and m_quadNumber[q] and m_num2x1Stored[q] from the Psana::CsPad::DataV# and ElementV# objects.
   */
  template <typename TDATA, typename TELEMENT>
  bool getCSPadConfigFromDataForType(PSEvt::Event& evt) {

    std::string key=""; // FOR RAW CSPAD DATA

    boost::shared_ptr<TDATA> data = evt.get(m_source, key, &m_src);
    if (data.get()) {
      m_numQuads = data->quads_shape()[0];
      m_num2x1StoredInData = 0;
      for (uint32_t q = 0; q < m_numQuads; ++ q) {
        const TELEMENT& el = data->quads(q);
        m_quadNumber[q]    = el.quad();
        m_num2x1Stored[q]  = el.data().shape()[0];
        m_num2x1StoredInData += m_num2x1Stored[q]; 
       }
      return true;
    }
    return false;
  }

//-------------------
public:

  /**
   * @brief Convers entire CSPAD pixel array, ndarray<T,3> with shape (32,185,388),
   * to the data array, ndarray<T,3> shaped as (N,185,388)
   */
  template <typename T>
      ndarray<T,3> getCSPadPixNDArrShapedAsData( ndarray<T,3>& arr_det ) { 
      
      const unsigned* shape = arr_det.shape();
      unsigned size2x1 = shape[1]*shape[2];
      //cout << "size2x1 = " << size2x1 << endl;
      //const unsigned shape[] = {32, PC2X1:ROWS2X1, PC2X1:COLS2X1}; // (32,185,388)
      //unsigned int shape_data[] = {m_num2x1StoredInData, shape[1], shape[2]}; // (N,185,388)

      ndarray<T,3> arr_data = make_ndarray<T>(m_num2x1StoredInData, shape[1], shape[2]);

      uint32_t ind2x1_in_data = 0;
      for (uint32_t q = 0; q < m_numQuads; ++ q) {
      
          uint32_t qNum = m_quadNumber[q]; 
          uint32_t mask = m_roiMask[q];
      
          for(uint32_t sect=0; sect < 8; sect++) {
              bool bitIsOn = mask & (1<<sect);
              if( !bitIsOn ) continue; 
      
              int ind2x1_in_det = qNum*8 + sect;             
      	      std::memcpy(&arr_data[ind2x1_in_data][0][0], &arr_det[ind2x1_in_det][0][0], size2x1*sizeof(double));      
              ind2x1_in_data ++;
      	  }
      }        
      return arr_data;
  }

//-------------------

  /**
   * @brief Convers entire CSPAD pixel data array, ndarray<T,3> shaped as (N,185,388)
   * to the entire CSPAD pixel array, ndarray<T,3> with shape (32,185,388),
   */
  template <typename T>
    ndarray<T,3> getCSPadPixNDArrFromNDArrShapedAsData( ndarray<T,3>& arr_data, T default_value=0 ) { 

      const unsigned* shape = arr_data.shape();
      unsigned size2x1 = shape[1]*shape[2];
      //cout << "size2x1 = " << size2x1 << endl;

      int numPixTotal = 32*size2x1;
      ndarray<T,3> arr_det = make_ndarray<T>(32, shape[1], shape[2]);
      std::fill_n(&arr_det[0][0][0], numPixTotal, double(default_value));

      uint32_t ind2x1_in_data = 0;
      for (uint32_t q = 0; q < m_numQuads; ++ q) {

          uint32_t qNum = m_quadNumber[q]; 
          uint32_t mask = m_roiMask[q];

          for(uint32_t sect=0; sect < 8; sect++) {
              bool bitIsOn = mask & (1<<sect);
              if( !bitIsOn ) continue; 

              int ind2x1_in_det = qNum*8 + sect;             
      	      std::memcpy(&arr_det[ind2x1_in_det][0][0], &arr_data[ind2x1_in_data][0][0], size2x1*sizeof(double));
              ind2x1_in_data ++;
	  }
      }        
      return arr_det;
  }

//-------------------

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPADCONFIGPARS_H
