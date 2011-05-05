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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "root/TFile.h"
//#include "root/TH1.h"
#include "RootHist/RootH1.h"
//#include "root/TTree.h"

using std::cout;
using std::endl;
//using std::ostream;

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

PSHist::H1* RootHManager::hist1i(const std::string &name, const std::string &title, int nbins, double low, double high) {
  //cout << "RootHManager::hist1i - equi-distant bins" << endl;  
    return new RootH1(type_int, name, title, nbins, low, high);
}

PSHist::H1* RootHManager::hist1i(const std::string &name, const std::string &title, int nbins, double *xbinedges) {
  //cout << "RootHManager::hist1i - variable bin sizes" << endl;
    return new RootH1(type_int, name, title, nbins, xbinedges);
}

PSHist::H1* RootHManager::hist1f(const std::string &name, const std::string &title, int nbins, double low, double high) {
  //cout << "RootHManager::hist1f - equi-distant bins" << endl;  
    return new RootH1(type_float, name, title, nbins, low, high);
}

PSHist::H1* RootHManager::hist1f(const std::string &name, const std::string &title, int nbins, double *xbinedges) {
  //cout << "RootHManager::hist1f - variable bin sizes" << endl;
    return new RootH1(type_float, name, title, nbins, xbinedges);
}

PSHist::H1* RootHManager::hist1d(const std::string &name, const std::string &title, int nbins, double low, double high) {
  //cout << "RootHManager::hist1d - equi-distant bins" << endl;  
    return new RootH1(type_double, name, title, nbins, low, high);
}

PSHist::H1* RootHManager::hist1d(const std::string &name, const std::string &title, int nbins, double *xbinedges) {
  //cout << "RootHManager::hist1d - variable bin sizes" << endl;
    return new RootH1(type_double, name, title, nbins, xbinedges);
}


int RootHManager::write() {
    cout << "RootHManager::write()" << endl;
    
    cout << "Write all histograms in file" << endl;
    m_file -> Write();

    return 0;
}


} // namespace RootHist
