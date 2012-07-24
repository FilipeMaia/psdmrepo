#ifndef XTCINPUT_RUNFILEITERI_H
#define XTCINPUT_RUNFILEITERI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RunFileIterI.
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
#include "XtcInput/StreamFileIterI.h"

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

class RunFileIterI : boost::noncopyable {
public:

  // Destructor
  virtual ~RunFileIterI () {}

  /**
   *  @brief Return stream iterator for next run.
   *  
   *  Zero pointer is returned after last run.
   */
  virtual boost::shared_ptr<StreamFileIterI> next() = 0;

  /**
   *  @brief Return run number for the set of files returned from last next() call.
   */
  virtual unsigned run() const = 0;
  
protected:

  // Default constructor
  RunFileIterI () {}

private:

};

} // namespace XtcInput

#endif // XTCINPUT_RUNFILEITERI_H
