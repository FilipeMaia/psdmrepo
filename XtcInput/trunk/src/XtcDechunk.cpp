//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcDechunk...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcDechunk.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/Exceptions.h"
#include "XtcInput/XtcDgIterator.h"
#include "pdsdata/xtc/Xtc.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcDechunk" ;

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcDechunk::XtcDechunk ( const std::list<XtcFileName>& files, size_t maxDgSize, bool skipDamaged )
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
XtcDechunk::~XtcDechunk ()
{
  delete m_dgiter ;
}

// read next datagram, return zero pointer after last file has been read,
// throws exception for errors.
Dgram::ptr
XtcDechunk::next()
{
  Dgram::ptr dgram;
  while ( not dgram.get() ) {

    if ( not m_dgiter ) {

      if ( m_iter == m_files.end() ) break ;

      // open next xtc file if there is none open
      MsgLog( logger, trace, "processing file: " << m_iter->path() ) ;
      m_dgiter = new XtcDgIterator ( m_iter->path(), m_maxDgSize ) ;
      m_count = 0 ;
    }

    // try to read next event from it
    dgram = m_dgiter->next() ;
    ++ m_count ;

    // if failed to read go to next file
    if ( not dgram.get() ) {
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
        dgram.reset();
      }

    }

  }

  return dgram ;
}

// get current file name
XtcFileName
XtcDechunk::chunkName() const
{
  if ( m_iter == m_files.end() ) return XtcFileName() ;
  return *m_iter ;
}

} // namespace XtcInput
