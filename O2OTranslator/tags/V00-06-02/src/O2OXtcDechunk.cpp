//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcDechunk...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OXtcDechunk.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OXtcDgIterator.h"
#include "pdsdata/xtc/Xtc.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "O2OXtcDechunk" ;

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2OXtcDechunk::O2OXtcDechunk ( const std::list<O2OXtcFileName>& files, size_t maxDgSize, bool skipDamaged )
  : m_files(files)
  , m_maxDgSize(maxDgSize)
  , m_skipDamaged(skipDamaged)
  , m_iter()
  , m_dgiter(0)
  , m_count(0)
{
  m_iter = m_files.begin();
}

//--------------
// Destructor --
//--------------
O2OXtcDechunk::~O2OXtcDechunk ()
{
  delete m_dgiter ;
}

// read next datagram, return zero pointer after last file has been read,
// throws exception for errors.
Pds::Dgram*
O2OXtcDechunk::next()
{
  Pds::Dgram* dgram = 0 ;
  while ( not dgram ) {

    if ( not m_dgiter ) {

      if ( m_iter == m_files.end() ) break ;

      // open next xtc file if there is none open
      MsgLog( logger, info, "processing file: " << m_iter->path() ) ;
      m_dgiter = new O2OXtcDgIterator ( m_iter->path(), m_maxDgSize ) ;
      m_count = 0 ;
    }

    // try to read next event from it
    dgram = m_dgiter->next() ;
    ++ m_count ;

    // if failed to read go to next file
    if ( not dgram ) {
      delete m_dgiter ;
      m_dgiter = 0 ;

      ++ m_iter ;
    } else if ( m_skipDamaged ) {

      // get rid of damaged datagrams
      const Pds::Xtc& xtc = dgram->xtc ;
      if ( xtc.damage.value() ) {
        MsgLog( logger, warning, "XTC damage: " << std::hex << xtc.damage.value() << std::dec
            << " level: " << int(xtc.src.level()) << '#' << Pds::Level::name(xtc.src.level())
            << " type: " << int(xtc.contains.id()) << '#' << Pds::TypeId::name(xtc.contains.id())
            << "/V" << xtc.contains.version()
            << "\n    Skipping damaged event -- file: " << m_iter->path() << " event: " << m_count ) ;
        dgram = 0 ;
      }

    }

  }

  return dgram ;
}

// get current file name
O2OXtcFileName
O2OXtcDechunk::chunkName() const
{
  if ( m_iter == m_files.end() ) return O2OXtcFileName() ;
  return *m_iter ;
}

} // namespace O2OTranslator
