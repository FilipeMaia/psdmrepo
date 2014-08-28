#ifndef XTCINPUT_CHUNKFILEITERLIVE_H
#define XTCINPUT_CHUNKFILEITERLIVE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ChunkFileIterLive.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "XtcInput/ChunkFileIterI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/LiveFilesDB.h"

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
 *  @brief Implementation of ChunkFileIterI interface which works with live data.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ChunkFileIterLive : public ChunkFileIterI {
public:

  // Default constructor
  ChunkFileIterLive (unsigned expNum, unsigned run, unsigned stream,
      unsigned liveTimeout, const boost::shared_ptr<LiveFilesDB>& filesdb) ;

  // Destructor
  virtual ~ChunkFileIterLive () ;

  /**
   *  @brief Return file name for next chunk.
   *
   *  Returns empty name after the last chunk.
   */
  virtual XtcFileName next();

  /**
   *  @brief Return live timeout value
   */
  virtual unsigned liveTimeout() const;

protected:

private:

  unsigned m_expNum;
  unsigned m_run;
  unsigned m_stream;
  unsigned m_liveTimeout;
  boost::shared_ptr<LiveFilesDB> m_filesdb;
  int m_chunk;
};

} // namespace XtcInput

#endif // XTCINPUT_CHUNKFILEITERLIVE_H
