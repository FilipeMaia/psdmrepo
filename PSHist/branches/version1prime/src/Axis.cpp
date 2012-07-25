//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Axis...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHist/Axis.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSHist/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHist {

//----------------
// Constructors --
//----------------
Axis::Axis(unsigned nbins, double amin, double amax)
  : m_amin(amin)
  , m_amax(amax)
  , m_nbins(nbins)
  , m_edges()
{
  if (m_nbins == 0) throw ExceptionBins(ERR_LOC);
  if (m_amin >= m_amax) throw ExceptionAxisRange(ERR_LOC, m_amin, m_amax);
}

Axis::Axis(unsigned nbins, const double *edges)
  : m_amin()
  , m_amax()
  , m_nbins(nbins)
  , m_edges(edges, edges+nbins+1)
{
  if (m_nbins == 0) throw ExceptionBins(ERR_LOC);
  // check that it is in increasing order
  for (const double* edge = edges; edge != edges+nbins; ++ edge) {
    if (*edge >= *(edge+1)) throw ExceptionAxisEdgeOrder(ERR_LOC);
  }
}

const double* 
Axis::edges() const
{ 
  if (m_edges.empty()) return 0; 
  return &m_edges[0];
}

// print data members
void 
Axis::print(std::ostream& out) const
{
  out << "=========================================================\n";
  if (m_edges.empty()) { 
    out << "Axis with equal bin sizes\n";
  } else {
    out << "Axis with variable bin sizes\n";
  }
  out << "Axis::m_nbins=" << m_nbins << " m_amin="  << m_amin << " m_amax="  << m_amax  << '\n';
  if (not m_edges.empty()) {
    out << "Axis::m_edges[0]=" << m_edges[0] << " m_edges[N]=" << m_edges[m_nbins] << '\n';
  }
  out << "=========================================================\n";
}

} // namespace PSHist
