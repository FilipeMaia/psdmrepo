#ifndef XTCINPUT_XTCCHUNKDGITER_H
#define XTCINPUT_XTCCHUNKDGITER_H

//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class XtcChunkDgIter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "XtcInput/DgHeader.h"
#include "XtcInput/SharedFile.h"
#include "XtcInput/XtcFileName.h"

//             ---------------------
//             -- Class Interface --
//             ---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Datagram iterator for datagrams in a single chunk file.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcChunkDgIter : boost::noncopyable {
public:

  /**
   *  @brief Create iterator instance
   *
   *  Constructor for iterator which reads datagrams from single XTC file.
   *  It tries to open specified path first, if it fails and path has
   *  extension ".inprogress" then it removes extension and tries to
   *  open file again. If that fails the exception will be thrown.
   *  If file has an extension ".inprogress" and live timeout is non-zero
   *  then it assumes that file can grow while we are reading it.
   *
   *  @param[in]  path         Path name for XTC file
   *  @param[in]  liveTimeout  If non-zero then defines timeout in seconds for reading live
   *                     data files, if zero assumes that files is closed already
   *
   *  @throw FileOpenException Thrown in case file cannot be open.
   */
  XtcChunkDgIter (const XtcFileName& path, unsigned liveTimeout = 0) ;

  // Destructor
  ~XtcChunkDgIter () ;

  /**
   *  @brief Returns next datagram header, zero on EOF, throws exceptions for errors
   *
   *  If reading ".inprogress" file stops at EOF and there is no
   *  new data for "live timeout" seconds then XTCLiveTimeout exception
   *  is generated.
   *  File is assumed to be closed and does not grow any more when
   *  it is renamed and ".inprogess" extension is dropped.
   *
   *  @return Shared pointer to datagram header object
   *
   *  @throw XTCReadException Thrown for any read errors
   *  @throw XTCLiveTimeout Thrown for timeout during live data reading
   *  @throw XTCExtentException Thrown if XTC header is corrupted and extend is below expected
   */
  boost::shared_ptr<DgHeader> next() ;

  /**
   *  @brief Returns next datagram header at offset, zero on EOF, throws exceptions for errors
   *
   *  Works the same as next() except reads datagram header from given offset. 
   *  No check is made that a offset is correct and a datagram starts there.
   *
   *  @return Shared pointer to datagram header object
   *
   *  @throw XTCReadException Thrown for any read errors
   *  @throw XTCLiveTimeout Thrown for timeout during live data reading
   *  @throw XTCExtentException Thrown if XTC header is corrupted and extend is below expected
   */
  boost::shared_ptr<DgHeader> nextAtOffset(off64_t offset);

  /**
   *  @brief Returns XtcFileName for this chunk.
   *
   *  @return the xtc filename object for the chunk
   */
  const XtcFileName & path() const { return m_file.path(); }

protected:

private:

  SharedFile m_file;    ///< Single chunk file
  off_t      m_off;     ///< offset in file of the next datagram to read

};

} // namespace XtcInput

#endif // XTCINPUT_XTCCHUNKDGITER_H
