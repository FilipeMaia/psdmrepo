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
#include "PSHist/Axis.h"
#include "PSHist/H1.h"
#include "PSHist/H2.h"
#include "PSHist/Profile.h"
#include "PSHist/Tuple.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "root/TFile.h"

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
  //RootHManager (const char*        filename = "psana.root", const char*        filemode = "RECREATE") ; // also works.
    RootHManager (const std::string &filename = "psana.root", const std::string &filemode = "RECREATE") ;

  // Destructor
  virtual ~RootHManager () ;

  // Selectors (const)
  // Modifiers
  // Static data members

  virtual int write(); // = 0; !!!!!!! 

  // 1-d histograms

  virtual PSHist::H1* hist1i(const std::string &name, const std::string &title, int nbins, double xlow, double xhigh);
  virtual PSHist::H1* hist1f(const std::string &name, const std::string &title, int nbins, double xlow, double xhigh);
  virtual PSHist::H1* hist1d(const std::string &name, const std::string &title, int nbins, double xlow, double xhigh);

  virtual PSHist::H1* hist1i(const std::string &name, const std::string &title, int nbins, double *xbinedges);
  virtual PSHist::H1* hist1f(const std::string &name, const std::string &title, int nbins, double *xbinedges);
  virtual PSHist::H1* hist1d(const std::string &name, const std::string &title, int nbins, double *xbinedges);

  virtual PSHist::H1* hist1i(const std::string &name, const std::string &title, PSHist::Axis &axis);
  virtual PSHist::H1* hist1f(const std::string &name, const std::string &title, PSHist::Axis &axis);
  virtual PSHist::H1* hist1d(const std::string &name, const std::string &title, PSHist::Axis &axis);


  // 2-d histograms

  virtual PSHist::H2* hist2i(const std::string &name, const std::string &title, PSHist::Axis &xaxis, PSHist::Axis &yaxis);
  virtual PSHist::H2* hist2f(const std::string &name, const std::string &title, PSHist::Axis &xaxis, PSHist::Axis &yaxis);
  virtual PSHist::H2* hist2d(const std::string &name, const std::string &title, PSHist::Axis &xaxis, PSHist::Axis &yaxis);


  // 1-d profile histograms

  virtual PSHist::Profile* prof1(const std::string &name, const std::string &title, int nbinsx, double xlow, double xhigh, 
                                                          double ylow, double yhigh, const std::string &option="");

  virtual PSHist::Profile* prof1(const std::string &name, const std::string &title, int nbins, double *xbinedges, 
                                                          double ylow, double yhigh, const std::string &option="");

  virtual PSHist::Profile* prof1(const std::string &name, const std::string &title, PSHist::Axis &axis, 
                                                          double ylow, double yhigh, const std::string &option="");

  // Tuple

  virtual PSHist::Tuple* tuple(const std::string &name, const std::string &title);


private:

  // Data members
  TFile   *m_file;

  // Copy constructor and assignment are disabled by default
  RootHManager             ( const RootHManager& ) ;
  RootHManager& operator = ( const RootHManager& ) ;

};

} // namespace RootHist

#endif // ROOTHIST_ROOTHMANAGER_H
