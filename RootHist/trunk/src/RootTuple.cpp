//------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootTuple...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// Class Headers --
//-----------------

#include "RootHist/RootTuple.h"
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


RootTuple::RootTuple ( const std::string &name, const std::string &title )
 : PSHist::Tuple()
{
  m_tuple = new TTree( name.c_str(), title.c_str() ); 
  cout << "RootTuple::RootTuple(...) - Created the tuple " << name << " with title=" << title << endl;
}



PSHist::Column* RootTuple::column( const std::string &name, void* address, const std::string &columnlist ) {
  cout << "RootTuple::column(...) - create the RootColumn " << name << " with list of parameters: " << columnlist.c_str() << endl;
  return new RootColumn (this, name, address, columnlist);
}


void RootTuple::print( std::ostream &o ) const { 
  o << "RootTuple" << endl;
}


void RootTuple::fill() {
  m_tuple->Fill();
} 


void RootTuple::reset() {
  m_tuple->Reset();
}


} // namespace RootHist
