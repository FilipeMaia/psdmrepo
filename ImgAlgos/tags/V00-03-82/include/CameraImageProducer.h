#ifndef IMGALGOS_CAMERAIMAGEPRODUCER_H
#define IMGALGOS_CAMERAIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraImageProducer.
// 1. Get Camera data as uint8_t or uint16_t
// 2. Subtract (if necessary) the offset from frmData->offset();
// 3. Save image in the event as ndarray<TOUT,2>
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"
#include "psddl_psana/camera.ddl.h"
//#include "PSEvt/EventId.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/GlobalMethods.h"
#include "MsgLogger/MsgLogger.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CameraImageProducer : public Module {
public:

  /// Data type for detector image 
  typedef uint16_t data_t;
  typedef uint8_t  data8_t;

  // Default constructor
  CameraImageProducer (const std::string& name) ;

  // Destructor
  virtual ~CameraImageProducer () ;

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
  void procEvent(Event& evt, Env& env);
  void printInputParameters();
  void printEventRecord(Event& evt, std::string comment="");
  void printSummary(Event& evt, std::string comment="");
  void checkTypeImplementation();
 
private:

  Pds::Src       m_src;              // source address of the data object
  Source         m_str_src;          // string with source name
  std::string    m_key_in;           // string with key for input data
  std::string    m_key_out;          // string with key for output image
  std::string    m_outtype;          // string type of output data asdata(uint16), double, float, int, int16
  bool           m_subtract_offset;  // true - subtryct, false - do not subtract
  unsigned       m_print_bits;       // control print bits
  long           m_count;            // local event counter
  unsigned long  m_count_msg;        // number of messages counter

  DATA_TYPE      m_dtype;            // enumerated type of output data
  DETECTOR_TYPE  m_detector;         // enumerated type of detectors

//--------------------
  /**
   * @brief Process event for requested data typename TDATA and output type TOUT
   * Returns false if data is missing.
   */
  template <typename TDATA, typename TOUT>
  bool procDataForIOTypes (Event& evt)
  {
      shared_ptr<TDATA> frame = evt.get(m_str_src, m_key_in, &m_src);
      if (frame.get()) {

	  TOUT offset = (m_subtract_offset) ? (TOUT)frame->offset() : 0;
    
          const ndarray<const data_t, 2>& data16 = frame->data16();
          if (not data16.empty()) {

              if(m_dtype == ASDATA) {
                 save2DArrayInEvent<data_t> (evt, m_src, m_key_out, frame->data16());
                 return true; 
              } 
 	      
              if( m_print_bits & 8 ) MsgLog(name(), info, "procEvent(...): Get image as ndarray<const uint16_t,2>,"
                                                          <<" frame offset=" << offset);
              ndarray<TOUT, 2> data_out = make_ndarray<TOUT>(frame->height(), frame->width());
              typename ndarray<TOUT, 2>::iterator oit;
              typename ndarray<const data_t, 2>::iterator dit;
              // This loop consumes ~5 ms/event for Opal1000 camera with 1024x1024 image size 
	      
	      if(m_detector == FCCD960) { 
                // Do special processing for FCCD960 gain factor bits
                for(dit=data16.begin(), oit=data_out.begin(); dit!=data16.end(); ++dit, ++oit) { 

		  uint16_t code = *dit;
		  //std::cout << "  xx:" << (code>>14);
		  switch ((code>>14)&03) {
		    default :
		    case  0 : *oit = TOUT( code&017777 );      break; // gain 8 - max gain in electronics - use factor 1 
		    case  1 : *oit = TOUT((code&017777) << 2); break; // gain 2 - use factor 4
		    case  3 : *oit = TOUT((code&017777) << 3); break; // gain 1 - use factor 8
		  }
                }
	      }
	      else {
                for(dit=data16.begin(), oit=data_out.begin(); dit!=data16.end(); ++dit, ++oit) { *oit = TOUT(*dit) - offset; }
	      }
    	      
    	      save2DArrayInEvent<TOUT> (evt, m_src, m_key_out, data_out);
              return true;
          }

          const ndarray<const uint8_t, 2>& data8 = frame->data8();
          if (not data8.empty()) {

              if(m_dtype == ASDATA) {
                 save2DArrayInEvent<uint8_t> (evt, m_src, m_key_out, frame->data8());
                 return true; 
              } 
	      
              if( m_print_bits & 8 ) MsgLog(name(), info, "procEvent(...): Get image as ndarray<const uint8_t,2>, subtract offset=" << offset);
              ndarray<TOUT, 2> data_out = make_ndarray<TOUT>(frame->height(), frame->width());
              typename ndarray<TOUT, 2>::iterator oit;
              typename ndarray<const uint8_t, 2>::iterator dit;
              for(dit=data8.begin(), oit=data_out.begin(); dit!=data8.end(); ++dit, ++oit) { *oit = TOUT(*dit) - offset; }
    	      
              save2DArrayInEvent<TOUT> (evt, m_src, m_key_out, data_out);
              return true;
          }

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

    if ( procDataForIOTypes <Psana::Camera::FrameV1, TOUT> (evt) ) return true;
    //if ( procDataForIOTypes <Psana::Camera::FrameV2, TOUT> (evt) ) return true;
    //if ( procDataForIOTypes <Psana::Camera::FrameV3, TOUT> (evt) ) return true;
    //if ( procDataForIOTypes <Psana::Camera::FrameV4, TOUT> (evt) ) return true;

      m_count_msg ++;
      if (m_count_msg < 11 && m_print_bits) {
        MsgLog(name(), warning, "Camera::FrameV1 object is not available in the event:" << m_count << " for source:" << m_str_src << " key:" << m_key_in);
        if (m_count_msg == 10)
          MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:" << m_str_src << " key:" << m_key_in);
      }
    return false;
  }

  //--------------------


}; // class
//-------------------
} // namespace ImgAlgos

#endif // IMGALGOS_CAMERAIMAGEPRODUCER_H
