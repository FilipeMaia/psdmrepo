#ifndef PSXTCINPUT_DGRAMREADER_H
#define PSXTCINPUT_DGRAMREADER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramReader.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <list>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PsXtcInput/XtcFileName.h"
#include "PsXtcInput/XtcStreamMerger.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PsXtcInput {

class DgramQueue ;

/**
 *  Thread which reads datagrams from the list of XTC files
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

class DgramReader : boost::noncopyable {
public:

  typedef std::list<XtcFileName> FileList ;

  // Default constructor
  DgramReader ( const FileList& files,
                DgramQueue& queue,
                size_t maxDgSize,
                XtcStreamMerger::MergeMode mode,
                bool skipDamaged,
                double l1OffsetSec = 0 ) ;

  // Destructor
  ~DgramReader () ;

  // this is the "run" method used by the Boost.thread
  void operator() () ;

protected:

private:

  // Data members
  FileList m_files ;
  DgramQueue& m_queue ;
  size_t m_maxDgSize ;
  XtcStreamMerger::MergeMode m_mode ;
  bool m_skipDamaged ;
  double m_l1OffsetSec ;

};

} // namespace PsXtcInput

#endif // PSXTCINPUT_DGRAMREADER_H
