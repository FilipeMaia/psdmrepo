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
#include <iostream>

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


template <typename HTYPE>
RootH1<HTYPE>::RootH1 ( const std::string& name, const std::string& title, 
                        const PSHist::Axis& axis ) 
  : PSHist::H1()
  , m_histp(0)
{ 
  if(axis.edges()) {
    m_histp = new HTYPE( name.c_str(), title.c_str(), axis.nbins(), axis.edges() ); 
  } else {
    m_histp = new HTYPE( name.c_str(), title.c_str(), axis.nbins(), axis.amin(), axis.amax() );
  }
  MsgLog(logger, debug, "Created 1d histogram " << name << " with title=" << title);
}

template <typename HTYPE>
RootH1<HTYPE>::~RootH1()
{
}


template <typename HTYPE>
void 
RootH1<HTYPE>::print(std::ostream& o) const
{
  o << "RootH1(" << m_histp->GetName() << ")";
}


template <typename HTYPE>
void 
RootH1<HTYPE>::fill(double x, double weight) 
{
  m_histp->Fill( x, weight );
}

template <typename HTYPE>
void 
RootH1<HTYPE>::fillN(unsigned n, const double* x, const double* weight)
{
  m_histp->FillN( n, x, weight );
}

template <typename HTYPE>
void 
RootH1<HTYPE>::reset() 
{
  m_histp->Reset();
}


//-----------------------------------
// Instatiation of templated classes
//-----------------------------------

template class RootHist::RootH1<TH1I>;
template class RootHist::RootH1<TH1F>;
template class RootHist::RootH1<TH1D>;

} // namespace RootHist
