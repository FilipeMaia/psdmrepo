#ifndef CSPADPIXCOORDS_CSPADIMAGEGETTEST_H
#define CSPADPIXCOORDS_CSPADIMAGEGETTEST_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadImageGetTest.
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


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief CSPadImageGetTest is a test/example module for psana framework.
 *
 *  CSPadImageGetTest psana module class works after CSPadImageProducer.
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

class CSPadImageGetTest : public Module {
public:

  // Default constructor
  CSPadImageGetTest (const std::string& name) ;

  // Destructor
  virtual ~CSPadImageGetTest () ;

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

  std::string strTimeStamp(Event& evt);
  std::string strRunNumber(Event& evt);
  std::string strEventCounter();
  void saveImageInFile(Event& evt);

private:

  // Data members, this is for example purposes only

  //Source      m_src;       // Data source set from config file
  Pds::Src    m_src;
  std::string m_str_src;     // i.e. CxiDs1.0:Cspad.0
  std::string m_key;         // i.e. Image2D
  unsigned    m_eventSave;   // event number starting from 1 to be saved in file
  bool        m_saveAll;     // should be true to save all events in files
  std::string m_fname;       // common part of the file name
  long        m_count;

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPADIMAGEGETTEST_H
