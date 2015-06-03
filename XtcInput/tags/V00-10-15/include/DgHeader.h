#ifndef XTCINPUT_DGHEADER_H
#define XTCINPUT_DGHEADER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgHeader.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"
#include "XtcInput/SharedFile.h"
#include "XtcInput/Dgram.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Class representing datagram header read from a file.
 *
 *  Instance of this class represent a header of a datagram
 *  as read from a file and it also knows how to read remaining
 *  datagram from a file.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class DgHeader : boost::noncopyable {
public:

  /**
   *  Constructor
   *
   *  @param[in] header   Datagram header
   *  @param[in] file     File object
   *  @param[in] off      Location of this datagram in a file
   */
  DgHeader(const Pds::Dgram& header, const SharedFile& file, off_t off);

  /// Returns offset of the next header (if there is any)
  off_t nextOffset() const;

  /// Get transition type
  Pds::TransitionId::Value transition() const { return m_header.seq.service(); }

  /// Get transition time
  const Pds::ClockTime& clock() const { return m_header.seq.clock(); }

  /// Get damage
  Pds::Damage damage() const { return m_header.xtc.damage; }

  /// Reads complete datagram into memory
  Dgram::ptr dgram();

  /// Get file name for this header
  const XtcFileName& path() const { return m_file.path(); }

  /// Get the offset of this dgram in the file
  const off_t offset() const { return m_off; }
protected:

private:

  Pds::Dgram m_header; ///< Actual datagram header
  SharedFile m_file;   ///< File where this datagram header was read from
  off_t      m_off;    ///< Location of this datagram in a file
  
};

} // namespace XtcInput

#endif // XTCINPUT_DGHEADER_H
