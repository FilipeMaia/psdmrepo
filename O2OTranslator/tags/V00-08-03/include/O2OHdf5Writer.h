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
#include "H5DataTypes/XtcClockTime.h"
#include "LusiTime/Time.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/DataTypeCvtI.h"
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

  enum SplitMode { NoSplit, Family } ;

  // current XTC state
  enum State { Undefined, Mapped, Configured, Running, CalibCycle,
               NumberOfStates } ;

  // Default constructor
  O2OHdf5Writer ( const O2OFileNameFactory& nameFactory,
                  bool overwrite,
                  SplitMode split,
                  hsize_t splitSize,
                  int compression,
                  bool extGroups,
                  const O2OMetaData& metadata ) ;

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

protected:

  // Construct a group name
  std::string groupName( State state, unsigned counter ) const ;

  void openGroup ( const Pds::Dgram& dgram, State state ) ;
  void closeGroup ( const Pds::Dgram& dgram, State state ) ;

private:

  typedef boost::shared_ptr<DataTypeCvtI> DataTypeCvtPtr ;
  typedef std::multimap<uint32_t, DataTypeCvtPtr> CvtMap ;
  typedef unsigned StateCounters[NumberOfStates] ;
  typedef LusiTime::Time TransitionClock[Pds::TransitionId::NumberOf] ;

  // Data members
  const O2OFileNameFactory& m_nameFactory ;
  hdf5pp::File m_file ;
  std::stack<State> m_state ;
  std::stack<hdf5pp::Group> m_groups ;
  H5DataTypes::XtcClockTime m_eventTime ;
  CvtMap m_cvtMap ;
  int m_compression ;
  bool m_extGroups ;
  const O2OMetaData& m_metadata ;
  StateCounters m_stateCounters ;
  Pds::TransitionId::Value m_transition;
  ConfigObjectStore m_configStore;
  TransitionClock m_transClock;

  // close all containers
  void closeContainers() ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OHDF5WRITER_H
