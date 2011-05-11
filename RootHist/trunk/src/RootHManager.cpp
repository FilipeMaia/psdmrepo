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

#include <iostream>
using std::cout;
using std::endl;
//using std::ostream;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "RootHist/RootH1.h"
#include "RootHist/RootH2.h"
#include "RootHist/RootProfile.h"
#include "RootHist/RootTuple.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RootHist {

//----------------
// Constructors --
//----------------

RootHManager::RootHManager ( const std::string &filename, const std::string & filemode ) : PSHist::HManager() {
    m_file = new TFile( filename.c_str(), filemode.c_str(), "Created by the RootHManager" );
    cout << "RootHManager::RootHManager(...): Root file : " << m_file->GetName()  << " is open in mode " << filemode << endl;
}


//--------------
// Destructor --
//--------------
RootHManager::~RootHManager () {
    cout << "RootHManager::~RootHManager () : The Root file : " << m_file->GetName() << " will be closed now." << endl;  
    m_file -> Close();
    delete m_file;
}


//--------------
// 1D histograms : equi-distant bins and variable bin sizes histograms for int, float and double data.
//--------------

PSHist::H1* RootHManager::hist1i(const std::string &name, const std::string &title, int nbins, double low, double high) {
 //cout << "RootHManager::hist1i - equi-distant bins" << endl;  
    return new RootH1<TH1I> (name, title, nbins, low, high);
}

PSHist::H1* RootHManager::hist1f(const std::string &name, const std::string &title, int nbins, double low, double high) {
 //cout << "RootHManager::hist1f - equi-distant bins" << endl;  
    return new RootH1<TH1F> (name, title, nbins, low, high);
}

PSHist::H1* RootHManager::hist1d(const std::string &name, const std::string &title, int nbins, double low, double high) {
 //cout << "RootHManager::hist1d - equi-distant bins" << endl;  
    return new RootH1<TH1D> (name, title, nbins, low, high);
}

PSHist::H1* RootHManager::hist1i(const std::string &name, const std::string &title, int nbins, double *xbinedges) {
 //cout << "RootHManager::hist1i - variable bin sizes" << endl;
    return new RootH1<TH1I> (name, title, nbins, xbinedges);
}

PSHist::H1* RootHManager::hist1f(const std::string &name, const std::string &title, int nbins, double *xbinedges) {
 //cout << "RootHManager::hist1f - variable bin sizes" << endl;
    return new RootH1<TH1F> (name, title, nbins, xbinedges);
}

PSHist::H1* RootHManager::hist1d(const std::string &name, const std::string &title, int nbins, double *xbinedges) {
 //cout << "RootHManager::hist1d - variable bin sizes" << endl;
    return new RootH1<TH1D> (name, title, nbins, xbinedges);
}

PSHist::H1* RootHManager::hist1i(const std::string &name, const std::string &title, PSHist::Axis &axis) {
 //cout << "RootHManager::hist1i - both bin sizes" << endl;
    return new RootH1<TH1I> (name, title, axis);
}

PSHist::H1* RootHManager::hist1f(const std::string &name, const std::string &title, PSHist::Axis &axis) {
 //cout << "RootHManager::hist1f - both bin sizes" << endl;
    return new RootH1<TH1F> (name, title, axis);
}

PSHist::H1* RootHManager::hist1d(const std::string &name, const std::string &title, PSHist::Axis &axis) {
 //cout << "RootHManager::hist1d - both bin sizes" << endl;
    return new RootH1<TH1D> (name, title, axis);
}


//--------------
// 2D histograms : equi-distant bins and variable bin sizes histograms for int, float and double data.
//--------------

PSHist::H2* RootHManager::hist2i(const std::string &name, const std::string &title, PSHist::Axis &xaxis, PSHist::Axis &yaxis ) {
  return new RootH2<TH2I> (name, title, xaxis, yaxis);
}

PSHist::H2* RootHManager::hist2f(const std::string &name, const std::string &title, PSHist::Axis &xaxis, PSHist::Axis &yaxis ) {
  return new RootH2<TH2F> (name, title, xaxis, yaxis);
}

PSHist::H2* RootHManager::hist2d(const std::string &name, const std::string &title, PSHist::Axis &xaxis, PSHist::Axis &yaxis ) {
  return new RootH2<TH2D> (name, title, xaxis, yaxis);
}



//----------------------
// 1D profile histograms : equi-distant bins and variable bin size profile histograms.
//----------------------

PSHist::Profile* RootHManager::prof1(const std::string &name, const std::string &title, int nbins, double low, double high, 
                                                              double ylow, double yhigh, const std::string &option) {
    return new RootProfile (name, title, nbins, low, high, ylow, yhigh, option);
}

PSHist::Profile* RootHManager::prof1(const std::string &name, const std::string &title, int nbins, double *xbinedges, 
                                                              double ylow, double yhigh, const std::string &option) {
    return new RootProfile (name, title, nbins, xbinedges, ylow, yhigh, option);
}

PSHist::Profile* RootHManager::prof1(const std::string &name, const std::string &title, PSHist::Axis &axis, 
                                                              double ylow, double yhigh, const std::string &option) {
    return new RootProfile (name, title, axis, ylow, yhigh, option);
}


//--------------
// Tuple
//--------------

PSHist::Tuple* RootHManager::tuple(const std::string &name, const std::string &title) {
    return new RootTuple (name, title);
}

//--------------
//--------------
//--------------

int RootHManager::write() {
    cout << "RootHManager::write()" << endl;
    
    cout << "Write all histograms in file" << endl;
    m_file -> Write();

    return 0;
}

} // namespace RootHist

