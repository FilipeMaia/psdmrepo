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
#include "O2OTranslator/DataTypeCvtFactoryI.h"

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
  enum State { Undefined, Mapped, Configured, Running } ;

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
  virtual void eventStart ( const Pds::Dgram& dgram ) ;
  virtual void eventEnd ( const Pds::Dgram& dgram ) ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) ;
  virtual void levelEnd ( const Pds::Src& src ) ;

  // visit the data object
  virtual void dataObject ( const void* data, const Pds::TypeId& typeId, const Pds::DetInfo& detInfo ) ;

protected:

  // Construct a group name
  std::string groupName( const std::string& prefix, const Pds::ClockTime& clock ) const ;

  void map ( const Pds::Dgram& dgram ) ;
  void configure ( const Pds::Dgram& dgram ) ;
  void startRun ( const Pds::Dgram& dgram ) ;
  void endRun ( const Pds::Dgram& dgram ) ;
  void unconfigure ( const Pds::Dgram& dgram ) ;
  void unmap ( const Pds::Dgram& dgram ) ;

private:

  typedef boost::shared_ptr<DataTypeCvtFactoryI> DataTypeCvtFactoryPtr ;
  typedef std::multimap<uint32_t, DataTypeCvtFactoryPtr> CvtMap ;

  // Data members
  const O2OFileNameFactory& m_nameFactory ;
  hdf5pp::File m_file ;
  std::stack<State> m_state ;
  hdf5pp::Group m_mapGroup ;      // Group for current Map transition
  hdf5pp::Group m_cfgGroup ;      // Group for current Configure transition
  hdf5pp::Group m_runGroup ;      // Group for current Run transition
  H5DataTypes::XtcClockTime m_eventTime ;
  CvtMap m_cvtMap ;
  int m_compression ;
  bool m_extGroups ;
  const O2OMetaData& m_metadata ;

  // close all containers
  void closeContainers() ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OHDF5WRITER_H
