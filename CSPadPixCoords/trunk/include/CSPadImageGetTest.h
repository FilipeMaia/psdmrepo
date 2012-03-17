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

  void saveImageInFile(Event& evt);

private:

  // Data members, this is for example purposes only

  std::string m_source;      // i.e. CxiDs1.0:Cspad.0
  Source      m_src;         // Data source set from config file
  std::string m_key;         // i.e. Image2D
  Pds::Src    m_actualSrc;
  unsigned    m_maxEvents;
  unsigned    m_eventSave;
  bool        m_saveAll;
  long        m_count;

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPADIMAGEGETTEST_H
