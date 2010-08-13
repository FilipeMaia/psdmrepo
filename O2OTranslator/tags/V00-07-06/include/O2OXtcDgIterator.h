#ifndef O2OTRANSLATOR_O2OXTCDGITERATOR_H
#define O2OTRANSLATOR_O2OXTCDGITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcDgIterator.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <cstdio>

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/Dgram.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

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

class O2OXtcDgIterator  {
public:

  // Default constructor
  O2OXtcDgIterator (const std::string& path, size_t maxDgramSize) ;

  // Destructor
  ~O2OXtcDgIterator () ;

  // Returns next datagram, zero or EOF, throws exceptions for errors
  Pds::Dgram* next() ;

protected:

private:

  // Data members
  std::string m_path;
  FILE* m_file;
  size_t m_maxDgramSize ;
  char* m_buf ;

  // Copy constructor and assignment are disabled by default
  O2OXtcDgIterator ( const O2OXtcDgIterator& ) ;
  O2OXtcDgIterator& operator = ( const O2OXtcDgIterator& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCDGITERATOR_H
