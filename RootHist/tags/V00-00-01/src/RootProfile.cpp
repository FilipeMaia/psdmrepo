//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootProfile...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// Class Headers --
//-----------------

#include "RootHist/RootProfile.h"

//-----------------
// C/C++ Headers --
//-----------------

using std::cout;
using std::endl;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RootHist {


RootProfile::RootProfile ( const std::string &name, const std::string &title, int nbins, double xlow, double xhigh, double ylow, double yhigh, const std::string &option )
 : PSHist::Profile()
{
  m_histp = new TProfile( name.c_str(), title.c_str(), nbins, xlow, xhigh, ylow, yhigh, option.c_str() ); 
  std::cout << "RootProfile::RootProfile(...) - Created the 1d profile histogram " << name << " with title=" << title << " and equal bin size" << std::endl;
}


RootProfile::RootProfile ( const std::string &name, const std::string &title, int nbins, double *xbinedges, double ylow, double yhigh, const std::string &option )
 : PSHist::Profile()
{
  m_histp = new TProfile( name.c_str(), title.c_str(), nbins, xbinedges, ylow, yhigh, option.c_str() ); 
  std::cout << "RootProfile::RootProfile(...) - Created the 1d profile histogram " << name << " with title=" << title << " and variable bin size" <<  std::endl;
}


RootProfile::RootProfile ( const std::string &name, const std::string &title, PSHist::Axis &axis, double ylow, double yhigh, const std::string &option )
 : PSHist::Profile()
{ 
  if(axis.edges()) m_histp = new TProfile( name.c_str(), title.c_str(), axis.nbins(), axis.edges(), ylow, yhigh, option.c_str() ); 
  else             m_histp = new TProfile( name.c_str(), title.c_str(), axis.nbins(), axis.amin(), axis.amax(), ylow, yhigh, option.c_str() );
  std::cout << "RootProfile::RootProfile(...) - Created the 1d profile histogram " << name << " with title=" << title << std::endl;
}


void RootProfile::print( std::ostream &o ) const
{ 
  o << "RootProfile" << endl;
}


void RootProfile::fill( double x, double y, double weight ) {

  m_histp->Fill( x, y, weight );
} 


void RootProfile::reset() {

  m_histp->Reset();
}


} // namespace RootHist
