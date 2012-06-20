#ifndef XTCINPUT_XTCDGITERATOR_H
#define XTCINPUT_XTCDGITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcDgIterator.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "XtcInput/Dgram.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/**
 *  Datagram iterator - reads datagrams from file, replacement for
 *  XtcFileIterator.
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

class XtcDgIterator : boost::noncopyable {
public:

  // Default constructor
  XtcDgIterator (const std::string& path, size_t maxDgramSize) ;

  // Destructor
  ~XtcDgIterator () ;

  // Returns next datagram, zero on EOF, throws exceptions for errors
  Dgram::ptr next() ;

protected:

private:

  // Data members
  std::string m_path;
  int m_fd;
  size_t m_maxDgramSize ;

};

} // namespace XtcInput

#endif // XTCINPUT_XTCDGITERATOR_H
