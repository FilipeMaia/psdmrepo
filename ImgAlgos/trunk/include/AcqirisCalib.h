#ifndef IMGALGOS_ACQIRISCALIB_H
#define IMGALGOS_ACQIRISCALIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AcqirisCalib.h 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
//
// Description:
//	Class AcqirisCalib.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"
#include "ndarray/ndarray.h"


#include "ImgAlgos/AcqirisArrProducer.h"

namespace ImgAlgos {

/**
 *  @brief Gets Acqiris waveforms from the evt store as ndarray<double, 2> subtracts baseline and put it back in the event. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id: AcqirisCalib.h 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
 *
 *  @author Mikhail Dubrovin
 */

class AcqirisCalib : public Module {
public:

  /// Data type for waveforms
  typedef AcqirisArrProducer::wform_t wform_t;

  /// Data type for timestamps
  typedef AcqirisArrProducer::wtime_t wtime_t;

  /// Default constructor
  AcqirisCalib (const std::string& name) ;

  /// Destructor
  virtual ~AcqirisCalib () ;

  /// Method which is called at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called at the end of the job
  virtual void endJob(Event& evt, Env& env);

  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);

  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);

  /// Prints values of input parameters
  void printInputParameters();


protected:

  bool procEvent(Event& evt, Env& env);
  bool isApprovedByCounters();
  void initInFirstEvent(Event& evt, Env& env);
  void loadBaseLine2DArrayFromFile();
  void printPartOfInputWaveform2DArray();
  void printPartOfBaseLine2DArray();

private:

  /// Source address of the data object
  Pds::Src        m_src;

  /// String with source name
  Source          m_str_src;

  /// String with key for input data
  std::string     m_key_in;

  /// String key for output waveform with subtracted base line
  std::string     m_key_out;

  /// String input file name for baseline subtraction
  std::string     m_fname_bline;

  /// String input file name with extension for experiment and run number for baseline subtraction
  std::string     m_fname_bline_ex;

  /// Number of events to skip in this module before start processing
  unsigned        m_skip_events;

  /// Number of events to process in this module before saving outut file with average
  unsigned        m_proc_events;

  /// Bit mask for print options
  unsigned        m_print_bits;

  /// Local event counter
  long            m_count_event;

  /// Local counter of gotten waveforms
  long            m_count_get;

  /// String with run number
  std::string     m_str_runnum; 

  /// String with experiment name
  std::string     m_str_experiment;

  /// String file name common prefix 
  std::string     m_fname_common;

  /// Boolean flag: true/false = do/not subtract baseline
  bool m_do_subtr_baseline;

  /// Number of Acqiris channels which produces waveforms (<=20)
  unsigned m_nbrChannels;

  /// Number of Acqiris channels samples along the waveform (order of 10000)
  unsigned m_nbrSamples;

  /// Size of the ndarray = m_nbrChannels * m_nbrSamples
  unsigned m_size;

  /// Last event for processing
  unsigned        m_last_event;

  /// Shared pointer to the waveform ndarray     
  shared_ptr< ndarray<wform_t,2> > sp_wf; 

  ndarray<wform_t,2> m_wf_bline;
  ndarray<wform_t,2> m_wf_data;
  ndarray<wform_t,2> m_wf;
};

} // namespace ImgAlgos

#endif // IMGALGOS_ACQIRISCALIB_H
