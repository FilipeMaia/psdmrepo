//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataTypeCvtFactoryI...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/DataTypeCvtFactoryI.h"

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

namespace O2OTranslator {

// comparator for DetInfo objects
bool
DataTypeCvtFactoryI::CmpDetInfo::operator()( const Pds::DetInfo& lhs, const Pds::DetInfo& rhs ) const
{
  int llog = lhs.log();
  int rlog = rhs.log();
  if ( llog < rlog ) return true ;
  if ( llog > rlog ) return false ;

  int lphy = lhs.phy();
  int rphy = rhs.phy();
  if ( lphy < rphy ) return true ;
  if ( lphy > rphy ) return false ;

  return false ;
}


//--------------
// Destructor --
//--------------
DataTypeCvtFactoryI::~DataTypeCvtFactoryI ()
{
}

} // namespace O2OTranslator
