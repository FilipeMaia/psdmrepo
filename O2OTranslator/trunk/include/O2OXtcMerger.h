#ifndef O2OTRANSLATOR_O2OXTCMERGER_H
#define O2OTRANSLATOR_O2OXTCMERGER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcMerger.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <list>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"
#include "O2OTranslator/O2OXtcDechunk.h"
#include "O2OTranslator/O2OXtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

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

class O2OXtcMerger  {
public:

  // Default constructor
  O2OXtcMerger ( const std::list<O2OXtcFileName>& files, size_t maxDgSize ) ;

  // Destructor
  ~O2OXtcMerger () ;

  // read next datagram, return zero pointer after last file has been read,
  // throws exception for errors.
  Pds::Dgram* next() ;

protected:

private:

  // Data members
  std::vector<O2OXtcDechunk*> m_streams ;
  std::vector<Pds::Dgram*> m_dgrams ;

  // Copy constructor and assignment are disabled by default
  O2OXtcMerger ( const O2OXtcMerger& ) ;
  O2OXtcMerger& operator = ( const O2OXtcMerger& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCMERGER_H
