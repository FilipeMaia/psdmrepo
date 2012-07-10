#ifndef XTCINPUT_XTCMERGEITERATOR_H
#define XTCINPUT_XTCMERGEITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMergeIterator.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"
#include "XtcInput/RunFileIterI.h"
#include "XtcInput/XtcStreamMerger.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Iterator class which merges datagrams from several files.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class XtcMergeIterator : boost::noncopyable {
public:

  // Default constructor
  XtcMergeIterator(const boost::shared_ptr<RunFileIterI>& runIter, 
      size_t maxDgSize, bool skipDamaged, double l1OffsetSec = 0);

  // Destructor
  ~XtcMergeIterator () ;

  /**
   *  @brief Return next datagram.
   *
   *  Read next datagram, return zero pointer after last file has been read,
   *  throws exception for errors.
   *
   *  @return Shared pointer to datagram object
   *
   *  @throw FileOpenException Thrown in case chunk file cannot be open.
   *  @throw XTCReadException Thrown for any read errors
   *  @throw XTCLiveTimeout Thrown for timeout during live data reading
   */
  Dgram next() ;

protected:

private:
  
  boost::shared_ptr<RunFileIterI> m_runIter;
  size_t m_maxDgSize;
  bool m_skipDamaged;
  double m_l1OffsetSec;
  boost::shared_ptr<XtcStreamMerger> m_dgiter ;  ///< Datagram iterator for current run

};

} // namespace XtcInput

#endif // XTCINPUT_XTCMERGEITERATOR_H
