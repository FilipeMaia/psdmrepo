#ifndef XTCINPUT_XTCFILTERTYPEIDSRC_H
#define XTCINPUT_XTCFILTERTYPEIDSRC_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilterTypeIdSrc.
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
 *  on specific combination of typeId and Src.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class XtcFilterTypeIdSrc  {
public:

  /// List of type IDs.
  typedef std::vector<Pds::TypeId::Type>  IdList;

  /**
   *  @brief Constructor takes two TypeId and Src.
   *
   *  @param[in] typeId  TypeId to keep
   *  @param[in] src     Src to keep
   */
  XtcFilterTypeIdSrc(Pds::TypeId::Type typeId, Pds::Src src) : m_typeId(typeId), m_src(src) {}

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
  bool operator()(const Pds::Xtc* input) const {
    return input->contains.id() == m_typeId and input->src == m_src;
  }

protected:

private:

  // Data members
  Pds::TypeId::Type m_typeId;
  Pds::Src m_src;
  
};

} // namespace XtcInput

#endif // XTCINPUT_XTCFILTERTYPEIDSRC_H
