#ifndef XTCINPUT_STREAMFILEITERI_H
#define XTCINPUT_STREAMFILEITERI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class StreamFileIterI.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/ChunkFileIterI.h"

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
 *  belonging to individual streams.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class StreamFileIterI : boost::noncopyable {
public:

  // Destructor
  virtual ~StreamFileIterI () {}
  
  /**
   *  @brief Return chunk iterator for next stream.
   *  
   *  Zero pointer is returned after last stream.
   */
  virtual boost::shared_ptr<ChunkFileIterI> next() = 0;

  /**
   *  @brief Return stream number for the set of files returned from last next() call.
   */
  virtual unsigned stream() const = 0;

protected:

  // Default constructor
  StreamFileIterI () {}

private:

};

} // namespace XtcInput

#endif // XTCINPUT_STREAMFILEITERI_H
