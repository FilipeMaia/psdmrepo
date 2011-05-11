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

using std::cout;
using std::endl;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RootHist {


RootColumn::RootColumn (RootTuple* tuple, const std::string &name, void* address, const std::string &columnlist)
 : PSHist::Column()
{
  int bufsize = 32000;
  int compress = -1; 
  
  //m_column = new TBranch(tuple->getTuplePointer(), name.c_str(), address, columnlist.c_str(), bufsize, compress); 

  m_column = tuple -> getTuplePointer() -> Branch(name.c_str(), address, columnlist.c_str()); 

  cout << "RootColumn::RootColumn(...) - Created the column '" << name 
       << "' with the list of parameters: " << columnlist.c_str() << endl;
}


void RootColumn::print( std::ostream &o ) const { 
  o << "RootColumn::print(...) " << endl;
}


  //void RootColumn::fill() {
  //  m_column->Fill();
  //} 


} // namespace RootHist
