#ifndef ROOTHIST_ROOTHMANAGER_H
#define ROOTHIST_ROOTHMANAGER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootHManager.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <vector>
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "PSHist/HManager.h"
#include "PSHist/H1.h"
//#include "RootHist/RootH1.h"
//#include "PSHist/Tuple.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

class TFile;

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RootHist {

/**
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class RootHManager : public PSHist::HManager {

public:

  // Default constructor
  //RootHManager ( const char*        filename = "psana.root", const char*        filemode = "RECREATE" ) ; // also works.
    RootHManager ( const std::string &filename = "psana.root", const std::string &filemode = "RECREATE" ) ;

  // Destructor
  virtual ~RootHManager () ;

  // Selectors (const)
  // Modifiers
  // Static data members

  virtual int write(); // = 0; !!!!!!! 

  virtual PSHist::H1* hist1i(const std::string &name, const std::string &title, int nbins, double xlow, double xhigh);
  virtual PSHist::H1* hist1i(const std::string &name, const std::string &title, int nbins, double *xbinedges);

  virtual PSHist::H1* hist1f(const std::string &name, const std::string &title, int nbins, double xlow, double xhigh);
  virtual PSHist::H1* hist1f(const std::string &name, const std::string &title, int nbins, double *xbinedges);

  virtual PSHist::H1* hist1d(const std::string &name, const std::string &title, int nbins, double xlow, double xhigh);
  virtual PSHist::H1* hist1d(const std::string &name, const std::string &title, int nbins, double *xbinedges);

private:

  // Data members
  TFile   *m_file;

  //H1      *m_histp;  
  //RootH1  *m_histp;  

  // Copy constructor and assignment are disabled by default
  RootHManager ( const RootHManager& ) ;
  RootHManager& operator = ( const RootHManager& ) ;

};

} // namespace RootHist

#endif // ROOTHIST_ROOTHMANAGER_H
