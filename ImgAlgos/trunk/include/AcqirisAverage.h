#ifndef IMGALGOS_ACQIRISAVERAGE_H
#define IMGALGOS_ACQIRISAVERAGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AcqirisAverage.h 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
//
// Description:
//	Class AcqirisAverage.
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
 *  @brief Gets Acqiris waveforms from the evt store as ndarray<double, 2> average them with threshold selection and save averaged waveform in file.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id: AcqirisAverage.h 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
 *
 *  @author Mikhail Dubrovin
 */

class AcqirisAverage : public Module {
public:

  /// Data type for waveforms
  typedef AcqirisArrProducer::wform_t wform_t;

  /// Data type for timestamps
  typedef AcqirisArrProducer::wtime_t wtime_t;

  /// Default constructor
  AcqirisAverage (const std::string& name) ;

  /// Destructor
  virtual ~AcqirisAverage () ;

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

  /// Gets Acqiris configuration parameters and returns configuration in text format
  //std::string getAcqirisConfig(Event& evt, Env& env);

  /// Gets the waveforms from data apply time corrections and put them in current event store
  //void proc_and_put_wf_in_event(Event& evt, Env& env);

  bool procEvent(Event& evt, Env& env);
  bool isApprovedByCounters();
  void initInFirstEvent(Event& evt, Env& env);
  void evaluateAverage(Event& evt);
  void printSelectionStatistics();
  void printVectorOfThresholds();
  
private:

  /// Source address of the data object
  Pds::Src        m_src;

  /// String with source name
  Source          m_str_src;

  /// String with key for input data
  std::string     m_key_in;

  /// String key for output averaged waveform 
  std::string     m_key_ave;

  /// String, user defined prefix of the output file name with averaged waveforms
  std::string     m_fname_ave_prefix;

  /// String for channel thresholds
  std::string     m_thresholds;

  /// List of flags, in string form: positive/negative signal true/false (leading positive edge or trailing negative edge/trailing positive edge or leading negative edge)
  std::string     m_is_positive_signal_list;

  /// List of flags, in string form, controlling selection algorithm
  std::string     m_do_inverse_selection_list;

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

  /// String output file name with averaged waveforms
  std::string     m_fname_ave;

  /// String with run number
  std::string     m_str_runnum; 

  /// String with experiment name
  std::string     m_str_experiment;

  /// String file name common prefix 
  std::string     m_fname_common;

   /// Boolean flag: true/false = do/not apply threshold
  bool m_do_threshold;

   /// Boolean flag: true/false = do/not save averaged waveforms in the file
  bool m_do_save_ave_file;

   /// Boolean flag: true/false = do/not save averaged waveforms in the evt store
  bool m_do_save_ave_evt;

  /// Boolean flag: true/false = average is done/or not
  bool m_average_is_done;

  /// Number of Acqiris channels which produces waveforms (<=20)
  unsigned m_nbrChannels;

  /// Number of Acqiris channels samples along the waveform (order of 10000)
  unsigned m_nbrSamples;

  /// Size of the ndarray = m_nbrChannels * m_nbrSamples
  unsigned m_size;

  /// Last event for processing
  unsigned        m_last_event;

  /// Statistics of selected waveforms (above threshold or below threshold for inversed selection)
  unsigned*       m_channel_stat;

  /// Vector of threshod values for all channels is filled from the string input parameter "thresholds"
  std::vector<wform_t> v_thresholds;

  /// Vector of bool for all channels is filled from the string input parameter "is_positive_signal_list"
  std::vector<bool> v_is_positive_signal;

  /// Vector of bool for all channels is filled from the string input parameter "do_inverse selection"
  std::vector<bool> v_do_inverse_selection;

  /// Shared pointer to the waveform ndarray     
  shared_ptr< ndarray<wform_t,2> > sp_wf; 

  /// Wavefors received from data
  //ndarray<wform_t,2> m_wf;

  /// Sum of wavefors received from data
  ndarray<wform_t,2> m_wf_sum;

  /// Averaged wavefors
  ndarray<wform_t,2> m_wf_ave;
};

} // namespace ImgAlgos

#endif // IMGALGOS_ACQIRISAVERAGE_H
