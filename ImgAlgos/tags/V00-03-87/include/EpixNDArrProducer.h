#ifndef PSANA_EXAMPLES_EPIXNDARRPRODUCER_H
#define PSANA_EXAMPLES_EPIXNDARRPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixNDArrProducer.
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
#include "psddl_psana/epix.ddl.h"

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
 *  @brief psana module which gets Epix data object and saves its data ndarray in the event store.
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

class EpixNDArrProducer : public Module {
public:

  /// Data type for detector image 
  typedef uint16_t data_t;

  // Default constructor
  EpixNDArrProducer (const std::string& name) ;

  // Destructor
  virtual ~EpixNDArrProducer () ;

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
   * @brief Get Psana::Epix::ConfigV1, Config10KV1, Config100aV1 config object and print it for m_print_bits.
   * Returns true/false if config object is available/missing.
   */

  template <typename T>
  bool getConfigData (Env& env, const std::string& objname=std::string("ConfigV1")) {

    shared_ptr<T> config1 = env.configStore().get(m_str_src);
    if (config1.get()) {    
      if( m_print_bits & 4 ) {
        WithMsgLog(name(), info, str) {
          str << "Epix::" << objname;
          str << "\n  version                  = " << config1->version();
          str << "\n  digitalCardId0           = " << config1->digitalCardId0();
          str << "\n  digitalCardId1           = " << config1->digitalCardId1();
          str << "\n  analogCardId0            = " << config1->analogCardId0();
          str << "\n  analogCardId1            = " << config1->analogCardId1();
          //str << "\n  lastRowExclusions        = " << config1->lastRowExclusions(); //missing in Epix100a
          str << "\n  numberOfAsicsPerRow      = " << config1->numberOfAsicsPerRow();
          str << "\n  numberOfAsicsPerColumn   = " << config1->numberOfAsicsPerColumn();
          str << "\n  numberOfRowsPerAsic      = " << config1->numberOfRowsPerAsic();
          str << "\n  numberOfPixelsPerAsicRow = " << config1->numberOfPixelsPerAsicRow();
          str << "\n  baseClockFrequency       = " << config1->baseClockFrequency();
          str << "\n  asicMask                 = " << config1->asicMask();
          str << "\n  numberOfRows             = " << config1->numberOfRows();
          str << "\n  numberOfColumns          = " << config1->numberOfColumns();
          str << "\n  numberOfAsics            = " << config1->numberOfAsics();  
        }    
      }
      return true;
    }
    return false;
  }

//--------------------
  /**
   * @brief Get Epix Psana::Epix::ElementV1,V2 data object and copy it in the ndarray<TOUT, 2> out_ndarr
   * Returns false if data is missing.
   */

  template <typename TOUT>
  bool procEventForOutputType (Event& evt) {

      shared_ptr<Psana::Epix::ElementV1> data1 = evt.get(m_str_src, m_key_in, &m_src);
      if (data1) {
	
          const ndarray<const data_t, 2> data = data1->frame();

          std::stringstream str; 
          if( m_print_bits & 2 ) {      
            str << "Epix::ElementV1 at " << m_src;
            str << "\n  vc           = " << int(data1->vc());
            str << "\n  lane         = " << int(data1->lane());
            str << "\n  acqCount     = " << data1->acqCount();
            str << "\n  frameNumber  = " << data1->frameNumber();
            str << "\n  ticks        = " << data1->ticks();
            str << "\n  fiducials    = " << data1->fiducials();
            str << "\n  frame        = " << data1->frame();
            str << "\n  excludedRows = " << data1->excludedRows();
            str << "\n  temperatures = " << data1->temperatures();
            str << "\n  lastWord     = " << data1->lastWord();
            str << "\n  data_ndarr:\n"   << data; 
            MsgLog(name(), info, str.str());
	  }
	  
          if ( m_dtype == ASDATA ) { // return data of the same type
            save2DArrayInEvent<data_t>(evt, m_src, m_key_out, data);
            return true;
	  }
	  else { // copy and return data with type changing
	    //const unsigned shape = {Segs,Rows,Cols};
	    //ndarray<TOUT, 3> out_ndarr( shape );
            ndarray<TOUT, 2> out_ndarr = make_ndarray<TOUT>(data.shape()[0], data.shape()[1]);
            typename ndarray<TOUT, 2>::iterator it_out = out_ndarr.begin(); 
            for ( ndarray<const data_t, 2>::iterator it=data.begin(); it!=data.end(); ++it, ++it_out) {
              *it_out = (TOUT)*it;
            }
            save2DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_ndarr); 
            return true;
	  }
      }



      shared_ptr<Psana::Epix::ElementV2> data2 = evt.get(m_str_src, m_key_in, &m_src);
      if (data2) {
	
          const ndarray<const data_t, 2> data = data2->frame();

          std::stringstream str; 
          if( m_print_bits & 2 ) {      
            str << "Epix::ElementV2 at "      << m_src;
            str << "\n  vc                = " << int(data2->vc());
            str << "\n  lane              = " << int(data2->lane());
            str << "\n  acqCount          = " << data2->acqCount();
            str << "\n  frameNumber       = " << data2->frameNumber();
            str << "\n  ticks             = " << data2->ticks();
            str << "\n  fiducials         = " << data2->fiducials();
            str << "\n  frame             = " << data2->frame();
            str << "\n  calibrationRows   = " << data2->calibrationRows();      //New
            str << "\n  environmentalRows = " << data2->environmentalRows();    //New
            str << "\n  temperatures      = " << data2->temperatures();
            str << "\n  lastWord          = " << data2->lastWord();
            str << "\n  data_ndarr:\n"        << data;  
            MsgLog(name(), info, str.str());
	  }
	  
          if ( m_dtype == ASDATA ) { // return data of the same type
            save2DArrayInEvent<data_t>(evt, m_src, m_key_out, data);
            return true;
	  }
	  else { // copy and return data with type changing
            ndarray<TOUT, 2> out_ndarr = make_ndarray<TOUT>(data.shape()[0], data.shape()[1]);
            typename ndarray<TOUT, 2>::iterator it_out = out_ndarr.begin(); 
            for ( ndarray<const data_t, 2>::iterator it=data.begin(); it!=data.end(); ++it, ++it_out) {
              *it_out = (TOUT)*it;
            }
            save2DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_ndarr); 
            return true;
	  }
      }

      static unsigned counter = 0; ++counter;
      if(counter > 20) return false;
      if( m_print_bits & 16 ) MsgLog(name(), warning, "Epix::ElementV1/V2 object is not available in the event(...) for source:"
              << m_str_src << " key:" << m_key_in);
      return false;

  }

//-------------------

};

} // namespace ImgAlgos

#endif // PSANA_EXAMPLES_EPIXNDARRPRODUCER_H
