#ifndef PSANA_EXAMPLES_ANDORIMAGEPRODUCER_H
#define PSANA_EXAMPLES_ANDORIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AndorImageProducer.h 0001 2014-01-17 09:00:00Z dubrovin@SLAC.STANFORD.EDU $
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
 *  @version $Id: AndorImageProducer.h 0001 2012-07-06 09:00:00Z dubrovin@SLAC.STANFORD.EDU $
 *
 *  @author Mikhail Dubrovin
 */

class AndorImageProducer : public Module {
public:

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

  DATA_TYPE   m_dtype;
 
//--------------------
  /**
   * @brief Process event for requested output type TOUT
   * Returns false if data is missing.
   */

  template <typename TOUT>
  bool procEventForOutputType (Event& evt) {

      shared_ptr<Psana::Andor::FrameV1> frame1 = evt.get(m_str_src, m_key_in, &m_src);
      if (frame1) {

          if( m_print_bits & 2 ) {      
        	cout << "\n  shotIdStart = " << frame1->shotIdStart()
        	     << "\n  readoutTime = " << frame1->readoutTime()
        	     << "\n  temperature = " << frame1->temperature()
                     << "\n  data:\n";
	  }      
      
	  // Use ndarray directly from data
	  if(m_dtype == ASDATA) {

	      const ndarray<unsigned short, 2>& data_ndarr = frame1->data().copy(); // copy ... because need to get rid of const ?
              if( m_print_bits & 2 ) {for (int i=0; i<10; ++i) cout << " " << data_ndarr[0][i]; cout << "\n"; }      

              //save2DArrayInEvent<const unsigned short> (evt, m_src, m_key_out, data_ndarr);
              save2DArrayInEvent<unsigned short> (evt, m_src, m_key_out, data_ndarr);
              return true;
	  } 

	  // Copy ndarray from data with type changing
	  const ndarray<const unsigned short, 2>& data_ndarr = frame1->data();
          //const unsigned* shape = data_ndarr.shape();
          //ndarray<TOUT,2> out_ndarr = make_ndarray<TOUT>(shape[0], shape[1]);
          ndarray<TOUT,2> out_ndarr( data_ndarr.shape() );
          typename ndarray<TOUT,2>::iterator it_out = out_ndarr.begin(); 
          for ( ndarray<const unsigned short,3>::iterator it=data_ndarr.begin(); it!=data_ndarr.end(); ++it, ++it_out) {
              *it_out = (TOUT)*it;
          }

          if( m_print_bits & 2 ) {for (int i=0; i<10; ++i) cout << " " << out_ndarr[0][i]; cout << "\n"; }      
          save2DArrayInEvent<TOUT> (evt, m_src, m_key_out, out_ndarr);
 
          return true;
      }
      else
      {
          MsgLog(name(), warning, "Andor::FrameV1 object is not available in the event(...) for source:"
              << m_str_src << " key:" << m_key_in);
	  return false;
      }

    return false;
  }

//-------------------

};

} // namespace ImgAlgos

#endif // PSANA_EXAMPLES_ANDORIMAGEPRODUCER_H
