#ifndef XTCINPUT_DGRAMREADER_H
#define XTCINPUT_DGRAMREADER_H

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

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/MergeMode.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

class DgramQueue ;

/**
 *  Thread which reads datagrams from the list of XTC files
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DgramReader {
public:

  typedef std::list<XtcFileName> FileList ;

  // Default constructor
  DgramReader ( const FileList& files,
                DgramQueue& queue,
                size_t maxDgSize,
                MergeMode mode,
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
  MergeMode m_mode ;
  bool m_skipDamaged ;
  double m_l1OffsetSec ;

};

} // namespace XtcInput

#endif // XTCINPUT_DGRAMREADER_H
