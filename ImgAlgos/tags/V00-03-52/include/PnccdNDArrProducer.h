#ifndef PSANA_EXAMPLES_PNCCDNDARRPRODUCER_H
#define PSANA_EXAMPLES_PNCCDNDARRPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdNDArrProducer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>  // for stringstream

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"
#include "psddl_psana/pnccd.ddl.h"

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

class PnccdNDArrProducer : public Module {
public:

  /// Data type for detector image 
  typedef uint16_t data_t;

  const static size_t   Segs   = 4; 
  const static size_t   Rows   = 512; 
  const static size_t   Cols   = 512; 
  const static size_t   FrSize = Rows*Cols; 
  const static size_t   Size   = Segs*Rows*Cols; 
   
  // Default constructor
  PnccdNDArrProducer (const std::string& name) ;

  // Destructor
  virtual ~PnccdNDArrProducer () ;

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
   * @brief Get pnccd four frames from Psana::PNCCD::FramesV1 data object and copy them in the ndarray<TOUT, 3> out_ndarr
   * Returns false if data is missing.
   */

  template <typename TOUT>
  bool procEventForOutputType (Event& evt) {

      shared_ptr<Psana::PNCCD::FramesV1> frames1 = evt.get(m_str_src, m_key_in, &m_src);
      if (frames1) {
	
	  //const unsigned shape = {Segs,Rows,Cols};
	  //ndarray<TOUT, 3> out_ndarr( shape );
          ndarray<TOUT, 3> out_ndarr = make_ndarray<TOUT>(Segs,Rows,Cols);
          typename ndarray<TOUT, 3>::iterator it_out = out_ndarr.begin(); 

          std::stringstream str; 

          if( m_print_bits & 2 ) str << "  numLinks = " << frames1->numLinks();

          for (unsigned i = 0 ; i != frames1->numLinks(); ++ i) {
          
              const Psana::PNCCD::FrameV1& frame = frames1->frame(i);          
              const ndarray<const data_t, 2> data = frame.data();

              if( m_print_bits & 2 ) {      
                str << "\n  Frame #" << i;          
                str << "\n    specialWord = " << frame.specialWord();
                str << "\n    frameNumber = " << frame.frameNumber();
                str << "\n    timeStampHi = " << frame.timeStampHi();
                str << "\n    timeStampLo = " << frame.timeStampLo();          
                str << "\n    frame size  = " << data.shape()[0] << 'x' << data.shape()[1];
	      }      

	      // Copy frame from data to output ndarray with changing type
              for ( ndarray<const data_t, 2>::iterator it=data.begin(); it!=data.end(); ++it, ++it_out) {
                  *it_out = (TOUT)*it;
              }
          }

          if( m_print_bits & 2 ) { str << "\n    out_ndarr:\n" << out_ndarr; MsgLog(name(), info, str.str() ); }

          save3DArrInEvent<TOUT>(evt, m_src, m_key_out, out_ndarr);
 
          return true;
      }
      else
      {
          if( m_print_bits & 16 ) MsgLog(name(), warning, "PNCCD::FramesV1 object is not available in the event(...) for source:"
              << m_str_src << " key:" << m_key_in);
	  return false;
      }

      return false;
  }

//-------------------

};

} // namespace ImgAlgos

#endif // PSANA_EXAMPLES_PNCCDNDARRPRODUCER_H
