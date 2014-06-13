#ifndef CSPADPIXCOORDS_CSPAD2X2CONFIGPARS_H
#define CSPADPIXCOORDS_CSPAD2X2CONFIGPARS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2ConfigPars.
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
#include "psddl_psana/cspad2x2.ddl.h"

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
 *  @brief CSPad2x2ConfigPars is a store for CSPAD2x2 configuration parameters.
 *
 *  <h1>Interface Description</h1>
 *   
 *  @li  Include and typedef
 *  @code
 *  #include "CSPadPixCoords/CSPad2x2ConfigPars.h"
 *  typedef CSPadPixCoords::CSPad2x2ConfigPars CONFIG;
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
 *      CONFIG* config = new CONFIG ("DetInfo(MecTargetChamber.0:Cspad2x2.3)");
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
 *      bool is_set = config -> setCSPad2x2ConfigPars (evt, env); 
 *      // or its separate private methods
 *      bool is_set = config -> setCSPad2x2ConfigParsFromEnv (env); 
 *      bool is_set = config -> setCSPad2x2ConfigParsFromEvent (evt); 
 *  @endcode
 *  \n
 *  Constructor from explicitly defined configuration parameters. It is not recommended to use. Can be used for stable non-complete configuration of the detector or for test purpose.
 *  @code
 *      uint32_t roiMask = 0;
 *      CONFIG* config = new CSPad2x2ConfigPars::CSPad2x2ConfigPars ( roiMask );
 *  @endcode
 *  
 *  
 * @li  Print current configuration parameters
 *  @code
 *      config -> printCSPad2x2ConfigPars();
 *  @endcode
 * 
 *  
 * @li  Get methods for member data  \n
 *  @code
 *      uint32_t n2x1     = config -> num2x1Stored();
 *      uint32_t roiMask  = config -> roiMask();
 *      float common_mode = config -> commonMode();
 *  @endcode
 *
 */  


/*   
 * @li  Conversion between entire CSPAD2x2 pixel array shaped as (185,388,2) and data array shaped as (185,388,N), where N<=2 \n
 * Conversion from (185,388,2) to (185,388,N)
 * @code
 *     ndarray<double,3> nda_det  = make_ndarray (p_pix_arr_det, N2X1_IN_DET, ROWS2X1, COLS2X1);
 *     ndarray<double,3> nda_data = cspad_configpars -> getCSPadPixNDArrShapedAsData<double> ( nda_det );
 * @endcode
 *  \n 
 * Conversion from (185,388,N) to (185,388,2)
 * @code
 *     ndarray<double,3> nda_data = make_ndarray (p_pix_arr_data, N2X1_IN_DATA, ROWS2X1, COLS2X1); 
 *     ndarray<double,3> nda_det  = cspad_configpars -> getCSPadPixNDArrFromNDArrShapedAsData<double> ( nda_data ); 
 * @endcode
 */

  
/** 
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CSPad2x2ConfigPars {
public:

  const static uint32_t N2x1 = 2; 

  //typedef CSPadPixCoords::PixCoords2x1V2 PC2X1;
  //typedef PixCoords2x1V2 PC2X1;

  //const static unsigned NQuadsMax = 4;

  /// @brief Default constructor
  CSPad2x2ConfigPars () ;

 /**
   * @brief Constructor using specified source as input parameter
   *  @param[in] source         (def.= "DetInfo(CxiDs1.0:Cspad.0)")                
   */
  CSPad2x2ConfigPars (PSEvt::Source source) ;

 /**
   *  @brief Constructor with explicit defenition of configuration parameters
   *  @param[in] roiMask        (def.= 03, or in decimal 3)
   */
  CSPad2x2ConfigPars ( uint32_t roiMask ) ;

  /// @brief Destructor
  virtual ~CSPad2x2ConfigPars () ;

 /**
   *  @brief Sets CSPAD configuration parameters
   *  @param[in] evt pointer to the event store
   *  @param[in] env pointer to the environment store 
   */
  bool setCSPad2x2ConfigPars(PSEvt::Event& evt, PSEnv::Env& env);

  /// Sets CSPAD configuration parameters to their default values
  void setCSPad2x2ConfigParsDefault();

  /// Prints CSPAD configuration parameters
  void printCSPad2x2ConfigPars();

  /// Returns the string with package name for logger. 
  std::string name() { return "pkg: CSPadPixCoords"; }  

  /// Returns the number of turned on (1) bits (2x1s) in the binary mask (def.= 8)
  uint32_t getNum2x1InMask(uint32_t mask);

  /// Returns the mask for 2x1s in the quad with index iq (def.=03 for 2 2x1s)
  uint32_t roiMask() { return m_roiMask; }

  /// Returns the number of 2x1s available in the CSPAD detector (def.= 32)
  uint32_t num2x1Stored() { return m_num2x1Stored; }

  /// Returns status: true if configuration parameters are set from env and evt, otherwise false.
  bool isSet() { return m_is_set; }   

  /// Returns common mode for 2x1s sections from evt
  float commonMode(int sec) { return m_common_mode[sec]; }

//--------------------
 
protected:

  /// part of the setCSPad2x2ConfigPars(PSEvt::Event& evt, PSEnv::Env& env)
  /// @param[in] env pointer to the environment store 
  bool setCSPad2x2ConfigParsFromEnv(PSEnv::Env& env);

  /// part of the setCSPad2x2ConfigPars(PSEvt::Event& evt, PSEnv::Env& env)
  /// @param[in] evt pointer to the event store
  bool setCSPad2x2ConfigParsFromEvent(PSEvt::Event& evt);

private:

  PSEvt::Source m_source;        /// member data for data source set from config file
  Pds::Src      m_src;           /// source address as Pds::Src 
  unsigned      m_count_cfg;
  std::string   m_config_vers; 
  std::string   m_data_vers; 
  bool          m_is_set_for_evt;
  bool          m_is_set_for_env;
  bool          m_is_set;
 
  // Parameters form Psana::CsPad::ConfigV# object
  uint32_t m_roiMask; /// mask for turrned on/off (1/0) 2x1s
  uint32_t m_num2x1Stored; /// number of 2x1s in CSPAD2x2 stored in data
  uint32_t m_numAsicsStored; /// number of ASICs in CSPAD2x2 stored 

  // Parameters form Psana::CsPad::DataV# and Psana::CsPad::ElementV# object
  float m_common_mode[2]; /// common mode for 2 sensors
 
//-------------------
  /**
   * @brief Gets m_roiMask and m_numAsicsStored from the Psana::CsPad2x2::ConfigV# object.
   * 
   */
  template <typename T>
  bool getConfigParsForType(PSEnv::Env& env) {

      boost::shared_ptr<T> config = env.configStore().get(m_source, &m_src);
      if (config) {
          m_roiMask        = config->roiMask();
          m_numAsicsStored = config->numAsicsStored();
	  m_num2x1Stored   = getNum2x1InMask(m_roiMask);
          ++ m_count_cfg;
          return true;
      }
      return false;
  }

//-------------------
  /**
   * @brief Gets extra info (if necessary) from the Psana::CsPad2x2::DataV# and ElementV# object.
   */
  template <typename TELEMENT>
  bool getCSPadConfigFromDataForType(PSEvt::Event& evt) {

    std::string key=""; // FOR RAW CSPAD DATA
    boost::shared_ptr<TELEMENT> elem = evt.get(m_source, key, &m_src);
    if (elem) {
      const ndarray<const int16_t, 3>& data = elem->data();
      for (unsigned i = 0; i != data.shape()[2]; ++ i) {
         m_common_mode[i] = elem->common_mode(i);
      }
      return true;
    }
    return false;
  }

//-------------------

public:

//  /**
//   * @brief Convers entire CSPAD pixel array, ndarray<T,3> with shape (32,185,388),
//   * to the data array, ndarray<T,3> shaped as (N,185,388)
//   */
//  template <typename T>
//      ndarray<T,3> getCSPadPixNDArrShapedAsData( ndarray<T,3>& arr_det ) { 
//      
//      const unsigned* shape = arr_det.shape();
//      unsigned size2x1 = shape[1]*shape[2];
//      //cout << "size2x1 = " << size2x1 << endl;
//      //const unsigned shape[] = {32, PC2X1:ROWS2X1, PC2X1:COLS2X1}; // (32,185,388)
//      //unsigned int shape_data[] = {m_num2x1StoredInData, shape[1], shape[2]}; // (N,185,388)
//
//      ndarray<T,3> arr_data = make_ndarray<T>(m_num2x1StoredInData, shape[1], shape[2]);
//
//      uint32_t ind2x1_in_data = 0;
//      for (uint32_t q = 0; q < m_numQuads; ++ q) {
//      
//          uint32_t qNum = m_quadNumber[q]; 
//          uint32_t mask = m_roiMask[q];
//      
//          for(uint32_t sect=0; sect < 8; sect++) {
//              bool bitIsOn = mask & (1<<sect);
//              if( !bitIsOn ) continue; 
//      
//              int ind2x1_in_det = qNum*8 + sect;             
//      	      std::memcpy(&arr_data[ind2x1_in_data][0][0], &arr_det[ind2x1_in_det][0][0], size2x1*sizeof(double));      
//              ind2x1_in_data ++;
//      	  }
//      }        
//      return arr_data;
//  }
//
////-------------------
//
//  /**
//   * @brief Convers entire CSPAD pixel data array, ndarray<T,3> shaped as (N,185,388)
//   * to the entire CSPAD pixel array, ndarray<T,3> with shape (32,185,388),
//   */
//  template <typename T>
//    ndarray<T,3> getCSPadPixNDArrFromNDArrShapedAsData( ndarray<T,3>& arr_data, T default_value=0 ) { 
//
//      const unsigned* shape = arr_data.shape();
//      unsigned size2x1 = shape[1]*shape[2];
//      //cout << "size2x1 = " << size2x1 << endl;
//
//      int numPixTotal = 32*size2x1;
//      ndarray<T,3> arr_det = make_ndarray<T>(32, shape[1], shape[2]);
//      std::fill_n(&arr_det[0][0][0], numPixTotal, double(default_value));
//
//      uint32_t ind2x1_in_data = 0;
//      for (uint32_t q = 0; q < m_numQuads; ++ q) {
//
//          uint32_t qNum = m_quadNumber[q]; 
//          uint32_t mask = m_roiMask[q];
//
//          for(uint32_t sect=0; sect < 8; sect++) {
//              bool bitIsOn = mask & (1<<sect);
//              if( !bitIsOn ) continue; 
//
//              int ind2x1_in_det = qNum*8 + sect;             
//      	      std::memcpy(&arr_det[ind2x1_in_det][0][0], &arr_data[ind2x1_in_data][0][0], size2x1*sizeof(double));
//              ind2x1_in_data ++;
////	  }
//      }        
//      return arr_det;
//  }
//
////-------------------

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPAD2X2CONFIGPARS_H
