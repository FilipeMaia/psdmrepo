//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootH1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// Class Headers --
//-----------------

#include "RootHist/RootH1.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <sstream> // for int to string conversion using std::stringstream 
#include <iomanip> // for formatted conversion std::setw(3) , std::setfill

using std::cout;
using std::endl;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RootHist {


template <typename HTYPE>
RootH1<HTYPE>::RootH1 ( const std::string &name, const std::string &title, PSHist::Axis &axis ) : PSHist::H1()
{ 
  if(axis.edges()) m_histp = new HTYPE( name.c_str(), title.c_str(), axis.nbins(), axis.edges() ); 
  else             m_histp = new HTYPE( name.c_str(), title.c_str(), axis.nbins(), axis.amin(), axis.amax() );
  std::cout << "RootH1::RootH1(...) - Created the 1d histogram " << name << " with title=" << title << std::endl;
}


template <typename HTYPE>
RootH1<HTYPE>::RootH1 ( const std::string &name, const std::string &title, int nbins, double xlow, double xhigh ) : PSHist::H1()
{
  m_histp = new HTYPE( name.c_str(), title.c_str(), nbins, xlow, xhigh ); 
  std::cout << "RootH1::RootH1(...) - Created the 1d histogram " << name << " with title=" << title << " and equal bin size" << std::endl;
}


template <typename HTYPE>
RootH1<HTYPE>::RootH1 ( const std::string &name, const std::string &title, int nbins, double *xbinedges ) : PSHist::H1()
{
  m_histp = new HTYPE( name.c_str(), title.c_str(), nbins, xbinedges ); 
  std::cout << "RootH1::RootH1(...) - Created the 1d histogram " << name << " with title=" << title << " and variable bin size" <<  std::endl;
}


template <typename HTYPE>
void RootH1<HTYPE>::print( std::ostream &o ) const
{ 
  o << "RootH1" << endl;
}


template <typename HTYPE>
void RootH1<HTYPE>::fill( double x, double weight ) {

  m_histp->Fill( x, weight );
} 


template <typename HTYPE>
void RootH1<HTYPE>::reset() {

  m_histp->Reset();
}


// RootH1<HTYPE>::getAutoGeneratedName() generates an automatic name for histogram, which looks like: H1_N0001
/*
template <typename HTYPE>
std::string RootH1<HTYPE>::getAutoGeneratedName() {
  std::stringstream stream_hname;
  stream_hname << "H1_N" << std::setw(4) << std::setfill('0') << m_hnumber;
  m_hname = stream_hname.str();
  return m_hname;
}
*/

//-----------------------------------
// Instatiation of templated classes
//-----------------------------------

template class RootHist::RootH1<TH1I>;
template class RootHist::RootH1<TH1F>;
template class RootHist::RootH1<TH1D>;

} // namespace RootHist
