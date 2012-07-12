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
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/MergeMode.h"

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

  typedef std::vector<std::string> FileList ;

  // Default constructor
  template <typename Iter>
  DgramReader(Iter begin, Iter end, DgramQueue& queue, size_t maxDgSize, MergeMode mode,
      const std::string& liveDbConn, const std::string& liveTable, unsigned liveTimeout,
      double l1OffsetSec = 0 )
    : m_files(begin, end)
    , m_queue( queue )
    , m_maxDgSize( maxDgSize )
    , m_mode( mode )
    , m_liveDbConn(liveDbConn)
    , m_liveTable(liveTable)
    , m_liveTimeout(liveTimeout)
    , m_l1OffsetSec(l1OffsetSec)
  {}

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
  std::string m_liveDbConn;
  std::string m_liveTable;
  unsigned m_liveTimeout;
  double m_l1OffsetSec ;

};

} // namespace XtcInput

#endif // XTCINPUT_DGRAMREADER_H
