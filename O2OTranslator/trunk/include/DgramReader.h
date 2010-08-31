#ifndef O2OTRANSLATOR_DGRAMREADER_H
#define O2OTRANSLATOR_DGRAMREADER_H

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
#include "O2OTranslator/O2OXtcFileName.h"
#include "O2OTranslator/O2OXtcMerger.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

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

class DgramReader  {
public:

  typedef std::list<O2OXtcFileName> FileList ;

  // Default constructor
  DgramReader ( const FileList& files,
                DgramQueue& queue,
                size_t maxDgSize,
                O2OXtcMerger::MergeMode mode,
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
  O2OXtcMerger::MergeMode m_mode ;
  bool m_skipDamaged ;
  double m_l1OffsetSec ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_DGRAMREADER_H
