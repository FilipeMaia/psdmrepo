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
#include <map>
#include <string>

//----------------------
// Base Class Headers --
//----------------------

#include "PSHist/HManager.h"
#include "PSHist/Axis.h"
#include "PSHist/H1.h"
#include "PSHist/H2.h"
#include "PSHist/Profile.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "root/TFile.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RootHist {

/**
 *  @defgroup RootHist RootHist package
 *  
 *  @brief Implementation of the histogramming service based on ROOT.
 *  
 *  This package contains implementation for PSHist interfaces which is 
 *  based on ROOT histograms and tuples. 
 */

/**
 *  @ingroup RootHist
 *  
 *  @brief Implementation of PSHist::HManager interface.
 *  
 *  RootHManager is implemented as a wrapper for TFile. Histograms and 
 *  tuples created by this class all end up ion the same ROOT file.
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see PSHist::HManager
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class RootHManager : public PSHist::HManager {

public:

  /**
   *  @brief Create manager instance
   *  
   *  @param[in] filename  Name of the ROOT file 
   *  @param[in] filemode  open mode
   *  
   *  @throw ExceptionFileOpen thrown if cannot open a file
   */
  RootHManager (const std::string& filename = "psana.root", const std::string& filemode = "RECREATE") ;

  // Destructor
  virtual ~RootHManager () ;


  // 1-d histograms

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1i(const std::string& name, const std::string& title, 
      int nbins, double xlow, double xhigh);

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1f(const std::string& name, const std::string& title, 
      int nbins, double xlow, double xhigh);

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1d(const std::string& name, const std::string& title, 
      int nbins, double xlow, double xhigh);


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1i(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges);

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1f(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges);

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1d(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges);


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1i(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis);

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1f(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis);

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H1* hist1d(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis);


  // 2-d histograms


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H2* hist2i(const std::string& name, const std::string& title, 
      const PSHist::Axis& xaxis, const PSHist::Axis& yaxis);

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H2* hist2f(const std::string& name, const std::string& title, 
      const PSHist::Axis& xaxis, const PSHist::Axis& yaxis);

  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::H2* hist2d(const std::string& name, const std::string& title, 
      const PSHist::Axis& xaxis, const PSHist::Axis& yaxis);


  // 1-d profile histograms


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::Profile* prof1(const std::string& name, const std::string& title, 
      int nbinsx, double xlow, double xhigh, const std::string& option="");


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::Profile* prof1(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges, const std::string& option="");


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::Profile* prof1(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis, const std::string& option="");


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::Profile* prof1(const std::string& name, const std::string& title, 
      int nbinsx, double xlow, double xhigh, double ylow, double yhigh, 
      const std::string& option="");


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::Profile* prof1(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges, double ylow, double yhigh, 
      const std::string& option="");


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual PSHist::Profile* prof1(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis, double ylow, double yhigh, const std::string& option="");


  /// Implementation of the corresponding method from PSHist::HManager interface.
  virtual void write();

private:

  /**
   *  @brief Check that name is unique.
   *  
   *  If the name is already known then throw an exception.
   *  
   *  @param[in] name Histogram or tuple name
   *  
   *  @throw PSHist::ExceptionDuplicateName
   */
  void checkName(const std::string& name);


  TFile   *m_file;               ///< ROOT file which stores all histograms
  std::map<std::string, PSHist::H1*> m_h1s;  ///< 1-dimensional histograms
  std::map<std::string, PSHist::H2*> m_h2s;  ///< 2-dimensional histograms
  std::map<std::string, PSHist::Profile*> m_profs;  ///< 1-dim profile histograms
};

} // namespace RootHist

#endif // ROOTHIST_ROOTHMANAGER_H
