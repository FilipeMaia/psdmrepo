//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Type...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/Type.h"

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

namespace hdf5pp {

//----------------
// Constructors --
//----------------
Type::Type ()
  : m_id ()
{
}

Type::Type ( hid_t id, bool doClose )
  : m_id ( new hid_t(id), TypePtrDeleter(doClose) )
{
}

//--------------
// Destructor --
//--------------
Type::~Type ()
{
}

} // namespace hdf5pp
