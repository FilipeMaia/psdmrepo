#ifndef XTCINPUT_XTCSTREAMMERGER_H
#define XTCINPUT_XTCSTREAMMERGER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcStreamMerger.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <list>
#include <vector>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"
#include "XtcInput/XtcDechunk.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/**
 *  Class responsible for merging of the datagrams from several
 *  XTC streams.
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

class XtcStreamMerger : boost::noncopyable {
public:

  /// Several merge modes supported:
  ///   OneStream - all files come from one stream, chunked
  ///   FilePerStream - single file per stream, no chunking
  ///   FileName - streams and chunks are determined from file names
  enum MergeMode { OneStream, NoChunking, FileName } ;

  // Default constructor
  XtcStreamMerger ( const std::list<XtcFileName>& files,
                 size_t maxDgSize,
                 MergeMode mode,
                 bool skipDamaged,
                 double l1OffsetSec = 0 ) ;

  // Destructor
  ~XtcStreamMerger () ;

  // read next datagram, return zero pointer after last file has been read,
  // throws exception for errors.
  Pds::Dgram* next() ;

protected:

  // update time in datagram
  void updateDgramTime(Pds::Dgram& dgram) const ;

private:

  // Data members
  std::vector<XtcDechunk*> m_streams ;
  std::vector<Pds::Dgram*> m_dgrams ;
  MergeMode m_mode ;
  int32_t m_l1OffsetSec ;
  int32_t m_l1OffsetNsec ;

};

} // namespace XtcInput

#endif // XTCINPUT_XTCSTREAMMERGER_H
