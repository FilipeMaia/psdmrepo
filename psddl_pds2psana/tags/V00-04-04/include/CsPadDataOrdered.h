#ifndef PSDDL_PDS2PSANA_CSPADDATAORDERED_H
#define PSDDL_PDS2PSANA_CSPADDATAORDERED_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadDataOrdered.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_pds2psana {

/**
 *  @brief special implementation of CsPad data class with ordering of quadrants. 
 *
 *  Template parameter can be one of psddl_pds2psana::CsPad::DataV* classes. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename DataType, typename ElemType>
class CsPadDataOrdered : public DataType {
public:

  // constructor
  template <typename XtcTypePtr, typename CfgTypePtr>
  CsPadDataOrdered (const XtcTypePtr& xtcPtr, const CfgTypePtr& cfgPtr) 
    : DataType(xtcPtr, cfgPtr)
    , m_quads()
  {
    unsigned nq = this->quads_shape()[0];
    
    // copy pointers to elements
    m_quads.resize(nq, 0);
    for (unsigned iq = 0; iq < nq; ++ iq) {
      m_quads[iq] = &(this->DataType::quads(iq));
    }
    
    // sort them
    std::sort(m_quads.begin(), m_quads.end(), ElemCmp());
  }

  // Destructor
  virtual ~CsPadDataOrdered () {}

  // override quads method
  virtual const ElemType& quads(uint32_t i0) const {
    return *m_quads[i0];
  }
  
protected:

private:

  struct ElemCmp {
    bool operator()(const ElemType* lhs, const ElemType* rhs) const {
      return lhs->quad() < rhs->quad();
    }
  };
  
  std::vector<const ElemType*> m_quads;

};

} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_CSPADDATAORDERED_H
