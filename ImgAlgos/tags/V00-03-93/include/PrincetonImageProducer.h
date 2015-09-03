#ifndef IMGALGOS_PRINCETONIMAGEPRODUCER_H
#define IMGALGOS_PRINCETONIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonImageProducer.
// 1. Get Camera data as uint16_t
// 2. Do nothing for now...
// 3. Save image in the event as ndarray<const m_dtype,2>
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"
#include "psddl_psana/princeton.ddl.h"
#include "psddl_psana/pimax.ddl.h"

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

class PrincetonImageProducer : public Module {
public:
      
  /// Data type for detector image 
   typedef uint16_t data_t;

  // Default constructor
  PrincetonImageProducer (const std::string& name) ;

  // Destructor
  virtual ~PrincetonImageProducer () ;

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
  void saveImageInEvent(Event& evt, double *p_data, const unsigned *shape);
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
  unsigned       m_print_bits;       // control print bits
  long           m_count;            // local event counter

  DATA_TYPE      m_dtype;            // enumerated type of output data
  unsigned long  m_count_msg;        // number of messages counter

  double*        m_data;             // pointer to output image data
  //ndarray<double,2>* m_ndarr; 

//--------------------

  /**
   * @brief Process event for requested data typename TDATA and output type TOUT
   * Returns false if data is missing.
   */
  template <typename TDATA, typename TOUT>
  bool procDataForIOTypes (Event& evt) {

      shared_ptr<TDATA> frame = evt.get(m_str_src, m_key_in, &m_src);
      if (frame.get()) {

          if(m_dtype == ASDATA) {
              save2DArrayInEvent<data_t> (evt, m_src, m_key_out, frame->data());
	  } 
	  else
	  {
	      // Get reference to data ndarray 
	      const ndarray<const data_t,2>& data = frame->data(); 

              // Create and initialize the array of the same shape as data, but for all 2x1...
              ndarray<TOUT,2> out_ndarr(data.shape());
              //std::fill(out_ndarr.begin(), out_ndarr.end(), TOUT(0));    
 
              // Pixel-by-pixel copy of data ndarray to output ndarray with type conversion:
              typename ndarray<TOUT,2>::iterator it_out = out_ndarr.begin(); 
              for ( ndarray<const data_t,2>::iterator it=data.begin(); it!=data.end(); ++it, ++it_out) {
                  *it_out = (TOUT)*it;
              } 
              save2DArrayInEvent<TOUT> (evt, m_src, m_key_out, out_ndarr);
	  }

          if( m_print_bits & 1 && m_count < 2 ) MsgLog( name(), info, " I/O data type: " << strOfDataTypeAndSize<data_t>() );
          if( m_print_bits & 8 ) MsgLog( name(), info, stringOf2DArrayData<data_t>(frame->data(), std::string(" data: ")) );
         
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
    if ( procDataForIOTypes <Psana::Princeton::FrameV1, TOUT> (evt) ) return true;
    if ( procDataForIOTypes <Psana::Princeton::FrameV2, TOUT> (evt) ) return true;
    if ( procDataForIOTypes <Psana::Pimax::FrameV1, TOUT> (evt) ) return true;

    m_count_msg ++;
    if (m_count_msg < 11 && m_print_bits) {
      MsgLog(name(), warning, "Princeton/Pimax::FrameV1/V2 object is not available in the event:" 
                              << m_count << " for source:" << m_str_src << " key:" << m_key_in);
      if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:" << m_str_src << " key:" << m_key_in);
    }
    return false;
  }

//-------------------
  }; // class
 
} // namespace ImgAlgos

#endif // IMGALGOS_PRINCETONIMAGEPRODUCER_H
