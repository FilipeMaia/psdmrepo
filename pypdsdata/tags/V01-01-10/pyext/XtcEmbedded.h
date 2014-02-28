#ifndef PYPDSDATA_XTCEMBEDDED_H
#define PYPDSDATA_XTCEMBEDDED_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: XtcEmbedded.h 774 2010-02-14 08:38:18Z salnikov $
//
// Description:
//	Class XtcEmbedded.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include "python/Python.h"     // Python.h should appear first to suppress warnings about _POSIX_C_SOURCE
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "types/PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/Xtc.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id: XtcEmbedded.h 774 2010-02-14 08:38:18Z salnikov $
 *
 *  @author Andrei Salnikov
 */

class XtcEmbedded : public PdsDataTypeEmbedded<XtcEmbedded,boost::shared_ptr<Pds::Xtc> > {
public:

  typedef PdsDataTypeEmbedded<XtcEmbedded,boost::shared_ptr<Pds::Xtc> > BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_XTCEMBEDDED_H
