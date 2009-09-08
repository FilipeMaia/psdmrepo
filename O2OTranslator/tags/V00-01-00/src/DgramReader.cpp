//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramReader...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/DgramReader.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/DgramQueue.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
DgramReader::DgramReader ( const StringList& files, DgramQueue& queue, size_t maxDgSize )
  : m_files( files )
  , m_queue( queue )
  , m_maxDgSize( maxDgSize )
{
}

//--------------
// Destructor --
//--------------
DgramReader::~DgramReader ()
{
}

// this is the "run" method used by the Boost.thread
void
DgramReader::operator() ()
{
  typedef StringList::const_iterator StringIter ;
  for ( StringIter eventFileIter = m_files.begin() ; eventFileIter != m_files.end() ; ++ eventFileIter ) {

    // get the file names
    const std::string& eventFile = *eventFileIter ;

    MsgLogRoot( info, "processing file: " << eventFile ) ;

    // open input xtc file
    FILE* xfile = fopen( eventFile.c_str(), "rb" );
    if ( ! xfile ) {
      MsgLogRoot( error, "failed to open input XTC file: " << eventFile ) ;
      // this is fatal error, stop here
      throw O2OFileOpenException(eventFile) ;
    }

    // iterate over events in xtc file
    Pds::XtcFileIterator iter( xfile, m_maxDgSize ) ;
    while ( Pds::Dgram* dg = iter.next() ) {

      // make a copy
      char* dgbuf = (char*)dg ;
      size_t dgsize = sizeof(Pds::Dgram) + dg->xtc.sizeofPayload();
      char* buf = new char[dgsize] ;
      std::copy( dgbuf, dgbuf+dgsize, buf ) ;

      // move it to the queue
      m_queue.push ( (Pds::Dgram*)buf ) ;

    }

  }

  // tell all we are done
  m_queue.push ( (Pds::Dgram*)0 ) ;

}


} // namespace O2OTranslator
