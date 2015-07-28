#ifndef IMGALGOS_IMGSAVEINFILE_H
#define IMGALGOS_IMGSAVEINFILE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgSaveInFile.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "PSEvt/Source.h"
#include "ImgAlgos/GlobalMethods.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief ImgSaveInFile is a test/example module for psana framework.
 *
 *  ImgSaveInFile psana module class works after CSPadImageProducer.
 *  It gets the Image2D object from the event.
 *  This image object may be used in data processing.
 *  For the test purpose, the image of particular event is saved in the text file.
 *  This event number is defined in the psana.cfg configuration file. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadImageProducer
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class ImgSaveInFile : public Module {
public:

  // Default constructor
  ImgSaveInFile (const std::string& name) ;

  // Destructor
  virtual ~ImgSaveInFile () ;

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

  void setFileMode();
  void saveImageInFile(Event& evt);
  void printInputParameters();

private:

  /// Source address of the data object
   Pds::Src    m_src;

  /// String-like object with source name
  Source      m_str_src;

  /// String with key for input data
  std::string m_key;

  /// Event number starting from 1 to be saved in file
  unsigned    m_eventSave;

  /// Should be true to save all events in files
  bool        m_saveAll;

  /// Common prefix of the file name
  std::string m_fname;

  /// File type "txt" or "bin" 
  std::string m_file_type;

  /// Bit mask for print options
  unsigned    m_print_bits;

  /// Event counter
  long        m_count;

  /// Message counter to constrain printout
  long        m_count_msg;

  /// Enumerated file type for "txt", "bin", etc. 
  FILE_MODE   m_file_mode;

  /// String with run number
  std::string m_str_runnum; 

  /// String with experiment name
  std::string m_str_experiment;

  /// String file name common prefix 
  std::string m_fname_common;
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGSAVEINFILE_H
