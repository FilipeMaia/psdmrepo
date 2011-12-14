#ifndef CSPAD_MOD_DATAT_H
#define CSPAD_MOD_DATAT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataT.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------
#include "psddl_psana/cspad.ddl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace cspad_mod {

/// @addtogroup cspad_mod

/**
 *  @ingroup cspad_mod
 *
 *  @brief Implementation of Psana::CsPad::DataT interface for
 *  calibrated data.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename DataType, typename ElemType>
class DataT : public DataType {
public:

  typedef DataType IfaceType;

  // Default constructor
  DataT () ;

  // Destructor
  virtual ~DataT () ;

  /// Append new element, ownership is taken by this object
  void append(ElemType* elem) { m_elements.push_back(elem); }

  virtual const ElemType& quads(uint32_t i0) const { return *m_elements[i0]; }
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  virtual std::vector<int> quads_shape() const { return std::vector<int>(1, m_elements.size()); }

protected:

private:

  std::vector<ElemType*> m_elements;

  // Copy constructor and assignment are disabled by default
  DataT ( const DataT& ) ;
  DataT& operator = ( const DataT& ) ;

};

typedef DataT<Psana::CsPad::DataV1, Psana::CsPad::ElementV1> DataV1;
typedef DataT<Psana::CsPad::DataV2, Psana::CsPad::ElementV2> DataV2;

} // namespace cspad_mod

#endif // CSPAD_MOD_DATAT_H
