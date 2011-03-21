//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcDgIterator...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcDgIterator.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcDgIterator";

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcDgIterator::XtcDgIterator (const std::string& path, size_t maxDgramSize)
  : m_path(path)
  , m_file(0)
  , m_maxDgramSize(maxDgramSize)
{
  m_file = fopen( m_path.c_str(), "rb" );
  if ( ! m_file ) {
    MsgLog( logger, error, "failed to open input XTC file: " << m_path ) ;
    throw FileOpenException(m_path) ;
  }
}

//--------------
// Destructor --
//--------------
XtcDgIterator::~XtcDgIterator ()
{
  fclose( m_file );
}

Dgram::ptr
XtcDgIterator::next()
{
  Pds::Dgram* dg = (Pds::Dgram*)new char[m_maxDgramSize];

  // read header
  if ( fread(dg, sizeof(Pds::Dgram), 1, m_file) != 1 ) {
    if ( feof(m_file) ) {
      return Dgram::ptr();
    } else {
      throw XTCReadException( m_path );
    }
  }

  // check payload size
  size_t payloadSize = dg->xtc.sizeofPayload();
  if ((payloadSize+sizeof(dg))>m_maxDgramSize) {
    throw XTCSizeLimitException(m_path, payloadSize+sizeof(dg), m_maxDgramSize);
  }

  // read rest of the data
  if ( payloadSize ) {
    if ( fread(dg->xtc.payload(), payloadSize, 1, m_file) != 1 ) {
      if ( feof(m_file) ) {
        MsgLog(logger, error, "EOF while reading datagram payload from file: " << m_path);
        return Dgram::ptr();
      } else {
        throw XTCReadException(m_path);
      }
    }
  }

  return Dgram::make_ptr(dg);
}

} // namespace XtcInput
