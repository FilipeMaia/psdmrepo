#ifndef O2OTRANSLATOR_O2OXTCDECHUNK_H
#define O2OTRANSLATOR_O2OXTCDECHUNK_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcDechunk.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <list>
#include <stdio.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"
#include "O2OTranslator/O2OXtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Class which merges the chunks from a single XTC stream
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

class O2OXtcDechunk  {
public:

  // Constructor accepts the list of files, the files will be sorted
  // based on the chunk number extracted from files name.
  O2OXtcDechunk ( const std::list<O2OXtcFileName>& files, size_t maxDgSize, bool skipDamaged ) ;

  // Destructor
  ~O2OXtcDechunk () ;

  // read next datagram, return zero pointer after last file has been read,
  // throws exception for errors.
  Pds::Dgram* next() ;

  // get current file name
  O2OXtcFileName chunkName() const ;

protected:

private:

  // Data members
  std::list<O2OXtcFileName> m_files ;
  size_t m_maxDgSize ;
  bool m_skipDamaged ;
  std::list<O2OXtcFileName>::const_iterator m_iter ;
  FILE* m_file ;
  Pds::XtcFileIterator* m_dgiter ;
  uint64_t m_count ;

  // Copy constructor and assignment are disabled by default
  O2OXtcDechunk ( const O2OXtcDechunk& ) ;
  O2OXtcDechunk& operator = ( const O2OXtcDechunk& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCDECHUNK_H
