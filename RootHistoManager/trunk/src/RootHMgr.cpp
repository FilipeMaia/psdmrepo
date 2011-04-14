//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootHMgr...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "RootHistoManager/RootHMgr.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "root/TH1I.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  //const char* logger = "RootHMgr";
  
  // factory function for 1-dim histograms
  template <typename TH>
  TH* makeH1(const std::string& name, const std::string& title, const RootHistoManager::AxisDef& axis)
  {
    if (axis.nbins()) {
      return new TH(name.c_str(), title.c_str(), axis.nbins(), axis.amin(), axis.amax());
    } else if (not axis.edges().empty()) {
      unsigned nbins = axis.edges().size() - 1;
      return new TH(name.c_str(), title.c_str(), nbins, &axis.edges()[0]);
    } else {
      return 0; 
    }
  }  
  
  // factory function for 1-dim histograms
  template <typename TH>
  TH* makeH2(const std::string& name, const std::string& title, 
      const RootHistoManager::AxisDef& xaxis, const RootHistoManager::AxisDef& yaxis)
  {
    if (xaxis.nbins() and yaxis.nbins()) {
      return new TH(name.c_str(), title.c_str(), 
                    xaxis.nbins(), xaxis.amin(), xaxis.amax(),
                    yaxis.nbins(), yaxis.amin(), yaxis.amax());
    } else if (xaxis.nbins() and not yaxis.edges().empty()) {
      unsigned nybins = yaxis.edges().size() - 1;
      return new TH(name.c_str(), title.c_str(), 
                    xaxis.nbins(), xaxis.amin(), xaxis.amax(),
                    nybins, &yaxis.edges()[0]);
    } else if (yaxis.nbins() and not xaxis.edges().empty()) {
      unsigned nxbins = xaxis.edges().size() - 1;
      return new TH(name.c_str(), title.c_str(), 
                    nxbins, &xaxis.edges()[0],
                    yaxis.nbins(), yaxis.amin(), yaxis.amax());
    } else if (not xaxis.edges().empty() and not yaxis.edges().empty()) {
      unsigned nxbins = xaxis.edges().size() - 1;
      unsigned nybins = yaxis.edges().size() - 1;
      return new TH(name.c_str(), title.c_str(), 
                    nxbins, &xaxis.edges()[0],
                    nybins, &yaxis.edges()[0]);
    } else {
      return 0; 
    }
  }  
  
}



//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RootHistoManager {

//----------------
// Constructors --
//----------------
RootHMgr::RootHMgr (const std::string& path)
  : m_path(path)
  , m_file()
{
}

//--------------
// Destructor --
//--------------
RootHMgr::~RootHMgr ()
{
  TFile* f = m_file.get();
  if (f) {
    f->Write();
    f->Close();
  }
}

TFile* 
RootHMgr::file()
{
  TFile* f = m_file.get();
  if (f) return f;

  f = TFile::Open(m_path.c_str(), "RECREATE", m_path.c_str());
  if (f) {
    m_file.reset(f);
  }
  return f;
}

/// create new 1-dim histogram with 32-bin integer bin contents
TH1*
RootHMgr::h1i(const std::string& name, const std::string& title, const AxisDef& axis)
{
  TFile* f = file();
  if (not f) return 0;
  f->cd();

  return ::makeH1<TH1I>(name, title, axis);
}

/// create new 1-dim histogram with double (64-bin) bin contents
TH1*
RootHMgr::h1d(const std::string& name, const std::string& title, const AxisDef& axis)
{
  TFile* f = file();
  if (not f) return 0;
  f->cd();

  return ::makeH1<TH1D>(name, title, axis);
}

/// create new 1-dim histogram with floating (32-bin) bin contents
TH1*
RootHMgr::h1f(const std::string& name, const std::string& title, const AxisDef& axis)
{
  TFile* f = file();
  if (not f) return 0;
  f->cd();

  return ::makeH1<TH1F>(name, title, axis);
}

/// create new 2-dim histogram with 32-bin integer bin contents
TH2*
RootHMgr::h2i(const std::string& name, const std::string& title, 
    const AxisDef& xaxis, const AxisDef& yaxis)
{
  TFile* f = file();
  if (not f) return 0;
  f->cd();

  return ::makeH2<TH2I>(name, title, xaxis, yaxis);
}

/// create new 2-dim histogram with double (64-bin) bin contents
TH2*
RootHMgr::h2d(const std::string& name, const std::string& title, 
    const AxisDef& xaxis, const AxisDef& yaxis)
{
  TFile* f = file();
  if (not f) return 0;
  f->cd();

  return ::makeH2<TH2D>(name, title, xaxis, yaxis);
}

/// create new 2-dim histogram with floating (32-bin) bin contents
TH2*
RootHMgr::h2f(const std::string& name, const std::string& title, 
    const AxisDef& xaxis, const AxisDef& yaxis)
{
  TFile* f = file();
  if (not f) return 0;
  f->cd();

  return ::makeH2<TH2F>(name, title, xaxis, yaxis);
}

/// create new 1-dim profile histogram
TProfile* 
RootHMgr::profile(const std::string& name, const std::string& title, 
    const AxisDef& axis, const std::string& option)
{
  TFile* f = file();
  if (not f) return 0;
  f->cd();
  
  if (axis.nbins()) {
    return new TProfile(name.c_str(), title.c_str(), 
                        axis.nbins(), axis.amin(), axis.amax(), option.c_str());
  } else if (not axis.edges().empty()) {
    unsigned nbins = axis.edges().size() - 1;
    return new TProfile(name.c_str(), title.c_str(), nbins, &axis.edges()[0], option.c_str());
  } else {
    return 0; 
  }
}


} // namespace RootHistoManager
