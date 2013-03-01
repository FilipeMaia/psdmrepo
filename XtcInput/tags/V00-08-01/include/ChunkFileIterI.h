#ifndef XTCINPUT_CHUNKFILEITERI_H
#define XTCINPUT_CHUNKFILEITERI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ChunkFileIterI.
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
 *  @brief Interface for class providing a sequence of file names.
 *
 *  Instances of this interface will provide a sequence of file names
 *  belonging to individual chunks from the same stream.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ChunkFileIterI : boost::noncopyable {
public:

  // Destructor
  virtual ~ChunkFileIterI () {}

  /**
   *  @brief Return file name for next chunk.
   *
   *  Returns empty name after the last chunk.
   */
  virtual XtcFileName next() = 0;

  /**
   *  @brief Return live timeout value
   */
  virtual unsigned liveTimeout() const = 0;

protected:

  // Default constructor
  ChunkFileIterI () {}

private:

};

} // namespace XtcInput

#endif // XTCINPUT_CHUNKFILEITERI_H
