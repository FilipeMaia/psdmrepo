//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataT...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/DataT.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
template <typename DataType, typename ElemType>
DataT<DataType, ElemType>::DataT ()
  : DataType()
  , m_elements()
{
}

//--------------
// Destructor --
//--------------
template <typename DataType, typename ElemType>
DataT<DataType, ElemType>::~DataT ()
{
  for (typename std::vector<ElemType*>::iterator it = m_elements.begin(); it != m_elements.end(); ++ it) {
    delete *it;
  }
}

// explicit instatiation
template class DataT<Psana::CsPad::DataV1, Psana::CsPad::ElementV1>;
template class DataT<Psana::CsPad::DataV2, Psana::CsPad::ElementV2>;

} // namespace cspad_mod
