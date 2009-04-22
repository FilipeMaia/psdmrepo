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

/**
 *  Scanner file that sends all data to HDF5 file
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

namespace O2OTranslator {

class O2OFileNameFactory ;
class O2ODataTypeCvtI ;

class O2OHdf5Writer : public O2OXtcScannerI {
public:

  enum SplitMode { NoSplit, Family } ;

  // current XTC state
  enum State { Undefined, Mapped, Configured, Running } ;

  // Default constructor
  O2OHdf5Writer ( const O2OFileNameFactory& nameFactory,
                  bool overwrite=false,
                  SplitMode split=NoSplit,
                  hsize_t splitSize=0xFFFFFFFF,
                  bool ignoreUnknowXtc=false,
                  int compression = -1 ) ;

  // Destructor
  virtual ~O2OHdf5Writer () ;

  // signal start/end of the event (datagram)
  virtual void eventStart ( const Pds::Sequence& seq ) ;
  virtual void eventEnd ( const Pds::Sequence& seq ) ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) ;
  virtual void levelEnd ( const Pds::Src& src ) ;

  // visit the data object
  virtual void dataObject ( const void* data, const Pds::TypeId& typeId, const Pds::DetInfo& detInfo ) ;

protected:

private:

  typedef boost::shared_ptr<DataTypeCvtFactoryI> DataTypeCvtFactoryPtr ;
  typedef std::multimap<uint32_t, DataTypeCvtFactoryPtr> CvtMap ;

  // Data members
  const O2OFileNameFactory& m_nameFactory ;
  hdf5pp::File m_file ;
  State m_state ;
  hdf5pp::Group m_mapGroup ;      // Group for current Map transition
  H5DataTypes::XtcClockTime m_eventTime ;
  CvtMap m_cvtMap ;
  bool m_ignore ;
  int m_compression ;

  // close all containers
  void closeContainers() ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OHDF5WRITER_H
