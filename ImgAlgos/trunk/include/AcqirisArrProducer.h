#ifndef IMGALGOS_ACQIRISARRPRODUCER_H
#define IMGALGOS_ACQIRISARRPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AcqirisArrProducer.h 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
//
// Description:
//	Class AcqirisArrProducer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"


namespace ImgAlgos {

/**
 *  @brief Gets Acqiris waveforms from data apply time corrections and put them in the evt store as ndarray<double, 2> for waveforms and timestamps.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id: AcqirisArrProducer.h 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
 *
 *  @author Mikhail Dubrovin
 */

class AcqirisArrProducer : public Module {
public:

  /// Data type for waveforms
  typedef double wform_t;

  /// Data type for timestamps
  typedef double wtime_t;

  /// Default constructor
  AcqirisArrProducer (const std::string& name) ;

  /// Destructor
  virtual ~AcqirisArrProducer () ;

  /// Method which is called at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);

  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);

  /// Prints values of input parameters
  void printInputParameters();


protected:

  /// Gets Acqiris configuration parameters and returns configuration in text format
  std::string getAcqirisConfig(Event& evt, Env& env);

  /// Prints part of the waveforms and associated meta-data
  void print_wf_in_event(Event& evt, Env& env);

  /// Prints info about wf indexes
  void print_wf_index_info(uint32_t indexFirstPoint, int32_t i0_seg, int32_t size);

  /// Gets the waveforms from data apply time corrections and put them in current event store
  void proc_and_put_wf_in_event(Event& evt, Env& env);


private:

  /// Source address of the data object
  Pds::Src        m_src;

  /// String with source name
  Source          m_str_src;

  /// String with key for input data
  std::string     m_key_in;

  /// String with key for output waveform 2-d array
  std::string     m_key_wform;

  /// String with key for output wavetime 2-d array
  std::string     m_key_wtime;

  /// String file name prefix
  std::string     m_fname_prefix;

  /// On/off switch for time correction in array index
  bool            m_correct_t;

  /// Bit mask for print options
  unsigned        m_print_bits;

  /// Local event counter
  long            m_count_event;

  /// Local calibcycle counter
  long            m_count_calib;

  /// String with run number
  std::string     m_str_runnum; 

  /// String with experiment name
  std::string     m_str_experiment;

  /// String file name common prefix 
  std::string     m_fname_common;

  /// Flag: true/false = do/not save Acqiris configuration parameters in file 
  bool m_do_save_config;
};

} // namespace ImgAlgos

#endif // IMGALGOS_ACQIRISARRPRODUCER_H
