#ifndef PYPDSDATA_XTCFILTER_H
#define PYPDSDATA_XTCFILTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../types/PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "XtcInput/XtcFilter.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

// Class that makes C++ functor out of Python callable
class XtcFilter_CallableWrapper {
public:
  XtcFilter_CallableWrapper(PyObject* obj);
  XtcFilter_CallableWrapper(const XtcFilter_CallableWrapper& other);
  XtcFilter_CallableWrapper& operator=(const XtcFilter_CallableWrapper& other);
  ~XtcFilter_CallableWrapper();
  bool operator()(const Pds::Xtc* input) const;
private:
  PyObject* m_obj;
};

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  @brief Python wrapper for XtcInput/XtcFilter C++ class.
 *
 *  C++ class is a template class, this Python class instead takes any type of
 *  callable as an argument.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class XtcFilter : public PdsDataTypeEmbedded<XtcFilter, XtcInput::XtcFilter<XtcFilter_CallableWrapper> > {
public:

  typedef XtcInput::XtcFilter<XtcFilter_CallableWrapper> PdsType;
  typedef PdsDataTypeEmbedded<XtcFilter, XtcInput::XtcFilter<XtcFilter_CallableWrapper> > BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_XTCFILTER_H
