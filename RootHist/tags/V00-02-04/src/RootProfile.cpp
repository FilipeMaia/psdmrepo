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


RootProfile::RootProfile ( const std::string& name, const std::string& title, 
    const PSHist::Axis& axis, const std::string& option )
 : PSHist::Profile()
  , m_histp(0)
{ 
  if(axis.edges()) {
    m_histp = new TProfile( name.c_str(), title.c_str(), 
                            axis.nbins(), axis.edges(), option.c_str() ); 
  } else {
    m_histp = new TProfile( name.c_str(), title.c_str(), 
                            axis.nbins(), axis.amin(), axis.amax(), option.c_str() );
  }
  MsgLog(logger, debug, "Created 1d profile histogram " << name << " with title=" << title);
}

RootProfile::RootProfile ( const std::string& name, const std::string& title, 
    const PSHist::Axis& axis, double ylow, double yhigh, const std::string& option )
 : PSHist::Profile()
 , m_histp(0)
{ 
  if(axis.edges()) {
    m_histp = new TProfile(name.c_str(), title.c_str(), 
                           axis.nbins(), axis.edges(), 
                           ylow, yhigh, option.c_str() ); 
  } else {
    m_histp = new TProfile( name.c_str(), title.c_str(), 
                            axis.nbins(), axis.amin(), axis.amax(), 
                            ylow, yhigh, option.c_str() );
  }
  MsgLog(logger, debug, "Created 1d profile histogram " << name << " with title=" << title);
}

RootProfile::~RootProfile()
{
}


void 
RootProfile::print( std::ostream& o ) const
{ 
  o << "RootProfile(" << m_histp->GetName() << ")";
}


void 
RootProfile::fill( double x, double y, double weight ) 
{
  m_histp->Fill(x, y, weight);
} 


void 
RootProfile::fillN(unsigned n, const double* x, const double* y, const double* weight) 
{
  m_histp->FillN(n, x, y, weight);
} 


void 
RootProfile::reset() 
{
  m_histp->Reset();
}


} // namespace RootHist
