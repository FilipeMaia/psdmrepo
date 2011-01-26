#ifndef XTCINPUT_XTCDECHUNK_H
#define XTCINPUT_XTCDECHUNK_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcDechunk.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <list>
#include <cstdio>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

class XtcDgIterator ;

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

class XtcDechunk : boost::noncopyable {
public:

  // Constructor accepts the list of files, the files will be sorted
  // based on the chunk number extracted from files name.
  XtcDechunk ( const std::list<XtcFileName>& files, size_t maxDgSize, bool skipDamaged ) ;

  // Destructor
  ~XtcDechunk () ;

  // read next datagram, return zero pointer after last file has been read,
  // throws exception for errors.
  Pds::Dgram* next() ;

  // get current file name
  XtcFileName chunkName() const ;

protected:

private:

  // Data members
  std::list<XtcFileName> m_files ;
  size_t m_maxDgSize ;
  bool m_skipDamaged ;
  std::list<XtcFileName>::const_iterator m_iter ;
  XtcDgIterator* m_dgiter ;
  uint64_t m_count ;

};

} // namespace XtcInput

#endif // XTCINPUT_XTCDECHUNK_H
