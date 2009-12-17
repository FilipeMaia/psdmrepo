//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcDgIterator...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OXtcDgIterator.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/O2OExceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "O2OXtcDgIterator";

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2OXtcDgIterator::O2OXtcDgIterator (const std::string& path, size_t maxDgramSize)
  : m_path(path)
  , m_file(0)
  , m_maxDgramSize(maxDgramSize)
  , m_buf(new char[maxDgramSize])
{
  m_file = fopen( m_path.c_str(), "rb" );
  if ( ! m_file ) {
    MsgLog( logger, error, "failed to open input XTC file: " << m_path ) ;
    throw O2OFileOpenException(m_path) ;
  }
}

//--------------
// Destructor --
//--------------
O2OXtcDgIterator::~O2OXtcDgIterator ()
{
  fclose( m_file );
  delete[] m_buf;
}

Pds::Dgram*
O2OXtcDgIterator::next()
{
  Pds::Dgram* dg = (Pds::Dgram*)m_buf;

  // read header
  if ( fread(dg, sizeof(Pds::Dgram), 1, m_file) != 1 ) {
    if ( feof(m_file) ) {
      return 0;
    } else {
      throw O2OXTCReadException( m_path );
    }
  }

  // check payload size
  size_t payloadSize = dg->xtc.sizeofPayload();
  if ((payloadSize+sizeof(dg))>m_maxDgramSize) {
    throw O2OXTCSizeLimitException(m_path, payloadSize+sizeof(dg), m_maxDgramSize);
    return 0;
  }

  // read rest of the data
  if ( fread(dg->xtc.payload(), payloadSize, 1, m_file) != 1 ) {
    if ( feof(m_file) ) {
      MsgLog( logger, error, "next -- EOF while reading datagram payload in file " << m_path ) ;
    }
    throw O2OXTCReadException(m_path);
  }

  return dg;
}

} // namespace O2OTranslator
