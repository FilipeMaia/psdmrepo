#ifndef XTCINPUT_XTCFILTERTYPEID_H
#define XTCINPUT_XTCFILTERTYPEID_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilterTypeId.
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
#include "pdsdata/xtc/Xtc.hh"

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
 *  @brief Functor class that filters contents of the XTCs based
 *  on the list of acceptable Type IDs.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class XtcFilterTypeId  {
public:

  /// List of type IDs.
  typedef std::vector<Pds::TypeId::Type>  IdList;

  /**
   *  @brief Constructor takes two lists of TypeIds.
   *
   *  If keep list is empty then all object will be kept except for those
   *  in discard list (this also covers empty discard list). If discard list
   *  is empty then all objects will be discarded except for those in keep
   *  list. If both lists are not empty then it will keep everything in keep
   *  list but not in discard list.
   *
   *  @param[in] keep    List of TypeId types to keep (all versions will kept)
   *  @param[in] discard List of TypeId types to throw away (all versions will removed)
   */
  XtcFilterTypeId(const IdList& keep, const IdList& discard);

  /**
   *  @brief Filter method
   *
   *  This method does actual filtering job. Note also that for some types
   *  of damage it may need to skip damaged data as well if the structure
   *  of XTC cannot be recovered. This happens independently of keep/discard
   *  filtering. The size of the buffer must be big enough to fit the data,
   *  output data cannot be larger than input XTC.
   *
   *  @param[in] input     XTC container object
   *  @return  True or false to accept/reject given XTC.
   */
  bool operator()(const Pds::Xtc* input) const;

protected:

private:

  // Data members
  IdList m_keep;
  IdList m_discard;
  
};

} // namespace XtcInput

#endif // XTCINPUT_XTCFILTERTYPEID_H
