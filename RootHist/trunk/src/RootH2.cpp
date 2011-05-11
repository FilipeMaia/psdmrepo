//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootH2...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// Class Headers --
//-----------------

#include "RootHist/RootH2.h"

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

  //int RootH2::s_number_of_booked_histograms = 0;

//----------------
// Constructors --
//----------------

template <typename HTYPE>
RootH2<HTYPE>::RootH2 ( const std::string &name, const std::string &title, PSHist::Axis &xaxis, PSHist::Axis &yaxis )
  : PSHist::H2()
{ 
  cout << "RootH2::RootH2(...) - Created the 2D histogram " << name << " with title=" << title << std::endl;
  if(xaxis.edges() and yaxis.edges()) 

                    m_histp = new HTYPE( name.c_str(), title.c_str(), xaxis.nbins(), xaxis.edges(), 
                                                                      yaxis.nbins(), yaxis.edges() ); 
  else if(xaxis.edges() and not yaxis.edges()) 
                    m_histp = new HTYPE( name.c_str(), title.c_str(), xaxis.nbins(), xaxis.edges(), 
                                                                      yaxis.nbins(), yaxis.amin(), yaxis.amax() );
  else if(not xaxis.edges() and yaxis.edges()) 
                    m_histp = new HTYPE( name.c_str(), title.c_str(), xaxis.nbins(), xaxis.amin(), xaxis.amax(),
                                                                      yaxis.nbins(), yaxis.edges() );
  else if(not xaxis.edges() and not yaxis.edges()) 
                    m_histp = new HTYPE( name.c_str(), title.c_str(), xaxis.nbins(), xaxis.amin(), xaxis.amax(),
                                                                      yaxis.nbins(), yaxis.amin(), yaxis.amax() );
}


template <typename HTYPE>
RootH2<HTYPE>::RootH2 ( const std::string &name, const std::string &title, int nbinsx, double xlow, double xhigh,
                                                                           int nbinsy, double ylow, double yhigh )
  : PSHist::H2()
{
  m_histp = new HTYPE( name.c_str(), title.c_str(), nbinsx, xlow, xhigh, nbinsy, ylow, yhigh ); 
  cout << "RootH2::RootH2(...) - Created the 2D histogram with equal bin size N" << name << " with title=" << title << std::endl;
}


template <typename HTYPE>
RootH2<HTYPE>::RootH2 ( const std::string &name, const std::string &title, int nbinsx, double xlow, double xhigh,
                                                                           int nbinsy, double *ybinedges )
  : PSHist::H2()
{
  m_histp = new HTYPE( name.c_str(), title.c_str(), nbinsx, xlow, xhigh, nbinsy, ybinedges );
  cout << "RootH2::RootH2(...) - Created the 2D histogram with mixed bin size N" << name << " with title=" << title << std::endl;
}


template <typename HTYPE>
RootH2<HTYPE>::RootH2 ( const std::string &name, const std::string &title, int nbinsx, double *xbinedges, 
                                                                           int nbinsy, double ylow, double yhigh )
  : PSHist::H2()
{
  m_histp = new HTYPE( name.c_str(), title.c_str(), nbinsx, xbinedges, nbinsy, ylow, yhigh );
  cout << "RootH2::RootH2(...) - Created the 2D histogram with mixed bin size N" << name << " with title=" << title << std::endl;
}


template <typename HTYPE>
RootH2<HTYPE>::RootH2 ( const std::string &name, const std::string &title, int nbinsx, double *xbinedges, 
                                                                           int nbinsy, double *ybinedges )
  : PSHist::H2()
{
  m_histp = new HTYPE( name.c_str(), title.c_str(), nbinsx, xbinedges, nbinsy, ybinedges );
  cout << "RootH2::RootH2(...) - Created the 2D histogram with variable bin size N" << name << " with title=" << title << std::endl;
}


template <typename HTYPE>
void RootH2<HTYPE>::print( std::ostream &o ) const
{ 
  o << "RootH2<HTYPE>::RootH2" << endl;
}


template <typename HTYPE>
void RootH2<HTYPE>::fill( double x, double y, double weight ) {

  m_histp->Fill( x, y, weight );
} 


template <typename HTYPE>
void RootH2<HTYPE>::reset() {

  m_histp->Reset();
}

//-----------------------------------
// Instatiation of templated classes
//-----------------------------------

template class RootHist::RootH2<TH2I>;
template class RootHist::RootH2<TH2F>;
template class RootHist::RootH2<TH2D>;

} // namespace RootHist


