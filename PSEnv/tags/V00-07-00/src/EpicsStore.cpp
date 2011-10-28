//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsStore...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSEnv/EpicsStore.h"

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

namespace PSEnv {

//----------------
// Constructors --
//----------------
EpicsStore::EpicsStore ()
  : m_impl(new EpicsStoreImpl())
{
}

//--------------
// Destructor --
//--------------
EpicsStore::~EpicsStore ()
{
}

} // namespace PSEnv
