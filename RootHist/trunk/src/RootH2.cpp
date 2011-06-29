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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

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

//----------------
// Constructors --
//----------------

template <typename HTYPE>
RootH2<HTYPE>::RootH2 ( const std::string& name, const std::string& title, 
                        const PSHist::Axis& xaxis, const PSHist::Axis& yaxis )
  : PSHist::H2()
  , m_histp(0)
{ 
  if(xaxis.edges() and yaxis.edges()) {
    
    m_histp = new HTYPE( name.c_str(), title.c_str(), 
                         xaxis.nbins(), xaxis.edges(), 
                         yaxis.nbins(), yaxis.edges() ); 
  
  } else if(xaxis.edges() and not yaxis.edges()) {
    
    m_histp = new HTYPE( name.c_str(), title.c_str(), 
                         xaxis.nbins(), xaxis.edges(), 
                         yaxis.nbins(), yaxis.amin(), yaxis.amax() );
  
  } else if(not xaxis.edges() and yaxis.edges()) {
    
    m_histp = new HTYPE( name.c_str(), title.c_str(), 
                         xaxis.nbins(), xaxis.amin(), xaxis.amax(), 
                         yaxis.nbins(), yaxis.edges() );
  
  } else if(not xaxis.edges() and not yaxis.edges()) {
    
    m_histp = new HTYPE( name.c_str(), title.c_str(), 
                         xaxis.nbins(), xaxis.amin(), xaxis.amax(),
                         yaxis.nbins(), yaxis.amin(), yaxis.amax() );
  
  }
  
  MsgLog(logger, debug, "Created 2D histogram " << name << " with title=" << title);
}

template <typename HTYPE>
RootH2<HTYPE>::~RootH2()
{
}

template <typename HTYPE>
void 
RootH2<HTYPE>::fill(double x, double y, double weight) 
{
  m_histp->Fill( x, y, weight );
} 

template <typename HTYPE>
void 
RootH2<HTYPE>::fillN(unsigned n, const double* x, const double* y, const double* weight) 
{
  m_histp->FillN(n, x, y, weight);
} 


template <typename HTYPE>
void RootH2<HTYPE>::reset() {

  m_histp->Reset();
}


template <typename HTYPE>
void 
RootH2<HTYPE>::print(std::ostream& o) const
{ 
  o << "RootH2(" << m_histp->GetName() << ")";
}


//-----------------------------------
// Instatiation of templated classes
//-----------------------------------

template class RootHist::RootH2<TH2I>;
template class RootHist::RootH2<TH2F>;
template class RootHist::RootH2<TH2D>;

} // namespace RootHist


