#ifndef O2OTRANSLATOR_O2OHDF5WRITER_H
#define O2OTRANSLATOR_O2OHDF5WRITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OHdf5Writer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <memory>
#include <stack>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/O2OXtcScannerI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"
#include "H5DataTypes/XtcClockTimeStamp.h"
#include "LusiTime/Time.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/CalibObjectStore.h"
#include "O2OTranslator/DataTypeCvtI.h"
#include "O2OTranslator/O2OCvtFactory.h"
#include "pdsdata/xtc/TransitionId.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class O2OFileNameFactory ;
class O2ODataTypeCvtI ;
class O2OMetaData ;

/**
 *  Scanner class that sends all data to HDF5 file.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class O2OHdf5Writer : public O2OXtcScannerI {
public:

  enum SplitMode { NoSplit, Family, SplitScan } ;

  // current XTC state
  enum State { Undefined, Configured, Running, CalibCycle,
               NumberOfStates } ;

  /**
   *  @brief Make writer instance
   *  
   *  @param[in] nameFactory  Instance of the factory class which creates output file names
   *  @param[in] overwrite    If true then allow overwriting of the outpur files
   *  @param[in] split        Output file splitting mode
   *  @param[in] splitSize    Size of the files at which to split
   *  @param[in] compression  Compression level, give negative number to disable compression
   *  @param[in] extGroups    If true generate extended group names with :NNNN suffix 
   *  @param[in] metadata     Object which keeps metadata (run number, experiment name, etc.)
   *  @param[in] finalDir     If non-empty then move files to this directory after closing
   *  @param[in] backupExt    If non empty then used as backup extension for existing files
   *  @param[in] fullTimeStamp If true then full timestamp will be stored (including fiducials)
   */
  O2OHdf5Writer ( const O2OFileNameFactory& nameFactory,
                  bool overwrite,
                  SplitMode split,
                  hsize_t splitSize,
                  int compression,
                  bool extGroups,
                  const O2OMetaData& metadata,
                  const std::string& finalDir,
                  const std::string& backupExt,
                  bool fullTimeStamp) ;

  // Destructor
  virtual ~O2OHdf5Writer () ;

  // signal start/end of the event (datagram)
  virtual bool eventStart ( const Pds::Dgram& dgram ) ;
  virtual void eventEnd ( const Pds::Dgram& dgram ) ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) ;
  virtual void levelEnd ( const Pds::Src& src ) ;

  // visit the data object
  virtual void dataObject ( const void* data, size_t size,
      const Pds::TypeId& typeId, const O2OXtcSrc& src ) ;

  // visit the data object in configure or begincalibcycle transitions
  virtual void configObject(const void* data, size_t size,
      const Pds::TypeId& typeId, const O2OXtcSrc& src);

protected:

  // Construct a group name
  std::string groupName( State state, unsigned counter ) const ;

  void openGroup ( const Pds::Dgram& dgram, State state ) ;
  void closeGroup ( const Pds::Dgram& dgram, State state, bool updateCounters = true ) ;

  // open new file
  void openFile();
  
  // close file/move to the final dir
  void closeFile();

  // store all configuration object from special store to a file
  void storeConfig0(); 

private:

  typedef unsigned StateCounters[NumberOfStates] ;
  typedef LusiTime::Time TransitionClock[Pds::TransitionId::NumberOf] ;

  // Data members
  const O2OFileNameFactory& m_nameFactory ;
  bool m_overwrite ;
  SplitMode m_split;
  hsize_t m_splitSize;
  int m_compression ;
  bool m_extGroups ;
  const O2OMetaData& m_metadata ;
  const std::string m_finalDir;
  const std::string m_backupExt;
  const bool m_fullTimeStamp;

  hdf5pp::File m_file ;
  std::stack<State> m_state ;
  std::stack<hdf5pp::Group> m_groups ;
  H5DataTypes::XtcClockTimeStamp m_eventTime ;
  StateCounters m_stateCounters ;
  Pds::TransitionId::Value m_transition;
  ConfigObjectStore m_configStore0;  // This contains only objects from latest Configure
  ConfigObjectStore m_configStore;   // This contains all configuration objects
  CalibObjectStore m_calibStore;
  O2OCvtFactory m_cvtFactory;        // must be after m_configStore and m_calibStore
  TransitionClock m_transClock;
  bool m_reopen;                     // if true then means time to reopen file in SplitScan mode
  unsigned m_serialScan;             // serial scan number, incremented on every EndCalibCycle
  unsigned m_scanAtOpen;             // scan number used to open the file

  // close all containers
  void closeContainers() ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OHDF5WRITER_H
