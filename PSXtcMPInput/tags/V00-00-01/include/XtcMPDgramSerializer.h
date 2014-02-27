#ifndef PSXTCMPINPUT_XTCMPDGRAMSERIALIZER_H
#define PSXTCMPINPUT_XTCMPDGRAMSERIALIZER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMPDgramSerializer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcMPInput {

/// @addtogroup PSXtcMPInput

/**
 *  @ingroup PSXtcMPInput
 *
 *  @brief Class which knows how to serialize datagrams to a stream.
 *
 *  Protocol for sending data over the wire is very simple:
 *
 *  - 4-byte header, string "DGRM"
 *  - 4-byte integer (native byte order) for total size of the data
 *    (including header and these 4 bytes)
 *  - sequence of datagrams, each datagram is represented as:
 *     - 1 byte flag: 0 - non-event datagram, 1 - event datagram
 *     - file name for datagram, zero-terminated string
 *     - 0 to 3 padding bytes, next byte is aligned at 4-bytes boundary
 *     - datagram contents, datagram size is determined from datagram header
 *     - 0 to 3 padding bytes, next byte is aligned at 4-bytes boundary
 *  - trailing 4 byte string "MRGD"
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class XtcMPDgramSerializer {
public:

  // Default constructor
  XtcMPDgramSerializer (int fd) ;

  /// send all datagrams to a stream, throws exception in case of trouble.
  void serialize(const std::vector<XtcInput::Dgram>& eventDg, const std::vector<XtcInput::Dgram>& nonEventDg);
  
  /// retrieve all datagrams from a stream, throws exception in case of trouble.
  void deserialize(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg);
  
protected:

  void send(const void* buf, size_t size);

  // sends datagram, returns number of bytes sent
  int sendDg(const XtcInput::Dgram& dg, uint8_t flag);

  int recv(void* buf, size_t size);

private:

  int m_fd;    ///< Descriptor for reading/writing
};

} // namespace PSXtcMPInput

#endif // PSXTCMPINPUT_XTCMPDGRAMSERIALIZER_H
