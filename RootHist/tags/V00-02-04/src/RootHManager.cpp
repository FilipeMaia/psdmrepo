//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootHManager...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "RootHist/RootHManager.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSHist/Exceptions.h"
#include "RootHist/Exceptions.h"
#include "RootHist/RootH1.h"
#include "RootHist/RootH2.h"
#include "RootHist/RootProfile.h"
#include "root/TSystem.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  const char* logger = "RootHist";
 
  template <typename Map>
  void deleteValues (Map& map) 
  {
    for (typename Map::iterator it = map.begin() ; it != map.end() ; ++ it) {
      delete it->second;
    }
    
  }
  
  // ROOT installs too many signal handlers, reset some of them
  bool resetRootSignals() {
    gSystem->ResetSignal(kSigPipe);
    return true;
  }
  bool initRoot = resetRootSignals();
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RootHist {

//----------------
// Constructors --
//----------------

RootHManager::RootHManager ( const std::string& filename, const std::string&  filemode ) 
  : PSHist::HManager()
  , m_file(0)
  , m_h1s()
  , m_h2s()
  , m_profs()
{
  // try to open file
  m_file = TFile::Open(filename.c_str(), filemode.c_str(), "Created by the RootHManager");
  if (not m_file or m_file->IsZombie()) throw ExceptionFileOpen(ERR_LOC, filename);
  
  MsgLog(logger, debug, "Root file : " << m_file->GetName()  << " is open in mode " << filemode);
}


//--------------
// Destructor --
//--------------
RootHManager::~RootHManager () 
{
  MsgLog(logger, debug, "Root file : " << m_file->GetName() << " will be closed now.");
  m_file->Close();
  delete m_file;

  // delete all our objects too
  ::deleteValues(m_h1s);
  ::deleteValues(m_h2s);
  ::deleteValues(m_profs);
}


//--------------
// 1D histograms : equi-distant bins and variable bin sizes histograms for int, float and double data.
//--------------

PSHist::H1* 
RootHManager::hist1i(const std::string& name, const std::string& title, 
    int nbins, double low, double high) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1I> (name, title, PSHist::Axis(nbins, low, high));
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H1* 
RootHManager::hist1f(const std::string& name, const std::string& title, 
    int nbins, double low, double high) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1F> (name, title, PSHist::Axis(nbins, low, high));
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H1* 
RootHManager::hist1d(const std::string& name, const std::string& title, 
    int nbins, double low, double high) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1D> (name, title, PSHist::Axis(nbins, low, high));
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H1* 
RootHManager::hist1i(const std::string& name, const std::string& title, 
    int nbins, const double *xbinedges) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1I> (name, title, PSHist::Axis(nbins, xbinedges));
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H1* 
RootHManager::hist1f(const std::string& name, const std::string& title, 
    int nbins, const double *xbinedges) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1F> (name, title, PSHist::Axis(nbins, xbinedges));
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H1* 
RootHManager::hist1d(const std::string& name, const std::string& title, 
    int nbins, const double *xbinedges) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1D> (name, title, PSHist::Axis(nbins, xbinedges));
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H1* 
RootHManager::hist1i(const std::string& name, const std::string& title, 
    const PSHist::Axis& axis) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1I> (name, title, axis);
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H1* 
RootHManager::hist1f(const std::string& name, const std::string& title, 
    const PSHist::Axis& axis) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1F> (name, title, axis);
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H1* 
RootHManager::hist1d(const std::string& name, const std::string& title, 
    const PSHist::Axis& axis) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H1* hist = new RootH1<TH1D> (name, title, axis);
  m_h1s.insert(std::make_pair(name, hist));

  return hist;
}


//--------------
// 2D histograms : equi-distant bins and variable bin sizes histograms for int, float and double data.
//--------------
PSHist::H2* 
RootHManager::hist2i(const std::string& name, const std::string& title, 
    const PSHist::Axis& xaxis, const PSHist::Axis& yaxis ) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H2* hist = new RootH2<TH2I> (name, title, xaxis, yaxis);
  m_h2s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H2* 
RootHManager::hist2f(const std::string& name, const std::string& title, 
    const PSHist::Axis& xaxis, const PSHist::Axis& yaxis ) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H2* hist = new RootH2<TH2F> (name, title, xaxis, yaxis);
  m_h2s.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::H2* 
RootHManager::hist2d(const std::string& name, const std::string& title, 
    const PSHist::Axis& xaxis, const PSHist::Axis& yaxis ) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::H2* hist = new RootH2<TH2D> (name, title, xaxis, yaxis);
  m_h2s.insert(std::make_pair(name, hist));

  return hist;
}



//----------------------
// 1D profile histograms : equi-distant bins and variable bin size profile histograms.
//----------------------

PSHist::Profile* 
RootHManager::prof1(const std::string& name, const std::string& title, 
    int nbins, double low, double high, const std::string& option) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::Profile* hist = new RootProfile (name, title, PSHist::Axis(nbins, low, high), option);
  m_profs.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::Profile* 
RootHManager::prof1(const std::string& name, const std::string& title, 
    int nbins, const double *xbinedges, const std::string& option) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::Profile* hist = new RootProfile (name, title, PSHist::Axis(nbins, xbinedges), option);
  m_profs.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::Profile* 
RootHManager::prof1(const std::string& name, const std::string& title, 
    const PSHist::Axis& axis, const std::string& option) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::Profile* hist = new RootProfile (name, title, axis, option);
  m_profs.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::Profile* 
RootHManager::prof1(const std::string& name, const std::string& title, 
    int nbins, double low, double high, 
    double ylow, double yhigh, const std::string& option) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::Profile* hist = new RootProfile (name, title, PSHist::Axis(nbins, low, high), 
                                           ylow, yhigh, option);
  m_profs.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::Profile* 
RootHManager::prof1(const std::string& name, const std::string& title, 
    int nbins, const double *xbinedges, 
    double ylow, double yhigh, const std::string& option) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::Profile* hist = new RootProfile (name, title, PSHist::Axis(nbins, xbinedges), 
                                           ylow, yhigh, option);
  m_profs.insert(std::make_pair(name, hist));

  return hist;
}

PSHist::Profile* 
RootHManager::prof1(const std::string& name, const std::string& title, 
    const PSHist::Axis& axis, double ylow, double yhigh, const std::string& option) 
{
  checkName(name);
  
  m_file->cd();
  PSHist::Profile* hist = new RootProfile (name, title, axis, ylow, yhigh, option);
  m_profs.insert(std::make_pair(name, hist));

  return hist;
}


void
RootHManager::write() 
{
  MsgLog(logger, debug, "RootHManager::write() : Write all histograms to file");

  m_file->Write();
  if (m_file->TestBit(TFile::kWriteError)) {
    throw PSHist::ExceptionStore(ERR_LOC, "Failure while writing histograms to ROOT file");
  }
}

void 
RootHManager::checkName(const std::string& name)
{
  // check name
  if (m_h1s.find(name) != m_h1s.end() or
      m_h2s.find(name) != m_h2s.end() or
      m_profs.find(name) != m_profs.end()) {
    throw PSHist::ExceptionDuplicateName(ERR_LOC, name);
  }
}

} // namespace RootHist

extern "C" PSHist::HManager * CREATE_PSHIST_HMANAGER_FROM_CONST_CHAR_PTR(const char *rfname) {
  return new RootHist::RootHManager (std::string(rfname));
}
