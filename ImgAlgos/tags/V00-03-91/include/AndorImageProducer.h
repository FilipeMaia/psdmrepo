#ifndef PSANA_EXAMPLES_ANDORIMAGEPRODUCER_H
#define PSANA_EXAMPLES_ANDORIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AndorImageProducer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"
#include "psddl_psana/andor.ddl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/GlobalMethods.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/**
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class AndorImageProducer : public Module {
public:

  /// Data type for detector image 
  typedef uint16_t data_t;

  // Default constructor
  AndorImageProducer (const std::string& name) ;

  // Destructor
  virtual ~AndorImageProducer () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);
  
protected:

  void printInputParameters();
  void procEvent(Event& evt, Env& env);
  void checkTypeImplementation();
 
private:

  Pds::Src    m_src;
  Source      m_str_src;
  std::string m_key_in; 
  std::string m_key_out;
  std::string m_outtype;
  unsigned    m_print_bits;
  unsigned    m_count; 
  unsigned    m_count_msg; 

  DATA_TYPE   m_dtype;

 
//--------------------
  /**
   * @brief Process event for requested output type TOUT
   * Returns false if data is missing.
   */

  template <typename TDATA, typename TOUT>
  bool procDataForIOTypes (Event& evt) {

      shared_ptr<TDATA> frame1 = evt.get(m_str_src, m_key_in, &m_src);
      if (frame1) {

          if( m_print_bits & 2 ) {      
        	cout << "\n  shotIdStart = " << frame1->shotIdStart()
        	     << "\n  readoutTime = " << frame1->readoutTime()
        	     << "\n  temperature = " << frame1->temperature()
                     << "\n  data:\n";
	  }      
      
	  const ndarray<const data_t, 2>& data_ndarr = frame1->data();
          if( m_print_bits & 2 ) {for (int i=0; i<10; ++i) cout << " " << data_ndarr[0][i]; cout << "\n"; }      

	  // Use ndarray directly from data
	  if(m_dtype == ASDATA) {
              save2DArrayInEvent<data_t> (evt, m_src, m_key_out, data_ndarr);
              return true;
	  } 

	  // Copy ndarray from data with type changing
          //const unsigned* shape = data_ndarr.shape();
          //ndarray<TOUT,2> out_ndarr = make_ndarray<TOUT>(shape[0], shape[1]);
          ndarray<TOUT,2> out_ndarr( data_ndarr.shape() );
          typename ndarray<TOUT,2>::iterator it_out = out_ndarr.begin(); 
          for ( ndarray<const data_t,3>::iterator it=data_ndarr.begin(); it!=data_ndarr.end(); ++it, ++it_out) {
              *it_out = (TOUT)*it;
          }

          if( m_print_bits & 2 ) {for (int i=0; i<10; ++i) cout << " " << out_ndarr[0][i]; cout << "\n"; }      
          save2DArrayInEvent<TOUT> (evt, m_src, m_key_out, out_ndarr);
 
          return true;
      }

    return false;
  }

  //--------------------
  /**
   * @brief Process event for requested output type  TOUT
   * Returns false if data is missing.
   */

  template <typename TOUT>
  bool procEventForOutputType (Event& evt) {

    if ( procDataForIOTypes <Psana::Andor::FrameV1, TOUT> (evt) ) return true;
    //if ( procDataForIOTypes <Psana::Camera::FrameV2, TOUT> (evt) ) return true;

    m_count_msg ++;
    if (m_count_msg < 11 && m_print_bits) {
      MsgLog(name(), warning, "Andor::FrameV1 object is not available in the event:" << m_count 
                              << " for source:" << m_str_src << " key:" << m_key_in);
      if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:" << m_str_src << " key:" << m_key_in);
    }
    return false;
  }

  //--------------------


//-------------------

};

} // namespace ImgAlgos

#endif // PSANA_EXAMPLES_ANDORIMAGEPRODUCER_H
