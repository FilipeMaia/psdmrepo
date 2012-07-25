//------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootColumn...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// Class Headers --
//-----------------

#include "RootHist/RootColumn.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "RootHist/RootTuple.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  const char* logger = "RootHist";
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RootHist {

RootColumn::RootColumn (RootTuple* tuple, const std::string& name, void* address, const std::string& columnlist)
 : PSHist::Column()
{
  m_column = tuple -> getTuplePointer() -> Branch(name.c_str(), address, columnlist.c_str()); 

  MsgLog(logger, debug, "Created tuple column '" << name 
       << "' with the list of parameters: " << columnlist.c_str());
}


void RootColumn::print( std::ostream& o ) const 
{ 
  o << "RootColumn(" << m_column->GetName() << ")";
}

} // namespace RootHist
