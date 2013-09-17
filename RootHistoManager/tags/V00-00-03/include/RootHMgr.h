#ifndef ROOTHISTOMANAGER_ROOTHMGR_H
#define ROOTHISTOMANAGER_ROOTHMGR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootHMgr.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RootHistoManager/AxisDef.h"
#include "root/TFile.h"
#include "root/TH1.h"
#include "root/TH2.h"
#include "root/TH3.h"
#include "root/TProfile.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  @defgroup RootHistoManager RootHistoManager package
 *  
 *  @brief Interim implementation of the histogramming service based on pure ROOT classes.
 *  
 *  This package is a thin wrapper around standard ROOT package. It just adds one more
 *  class RootHMgr which is a factory class for ROOT histograms and also manages the 
 *  lifetime of the associated ROOT file and histograms that it creates.
 */


namespace RootHistoManager {

/**
 *  @ingroup RootHistoManager
 *  
 *  @brief ROOT-specific histogram manager for psana. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see PSEnv::Env
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class RootHMgr : boost::noncopyable {
public:

  /**
   *  @brief Constructor takes the name of the ROOT file which will be 
   *  overwritten if it exists already.
   */
  RootHMgr (const std::string& path) ;

  // Destructor
  ~RootHMgr () ;

  /// create new 1-dimensional histogram with 32-bin integer bin contents
  TH1* h1i(const std::string& name, const std::string& title, const AxisDef& axis);

  /// create new 1-dimensional histogram with double (64-bin) bin contents
  TH1* h1d(const std::string& name, const std::string& title, const AxisDef& axis);

  /// create new 1-dimensional histogram with floating (32-bin) bin contents
  TH1* h1f(const std::string& name, const std::string& title, const AxisDef& axis);

  /// create new 2-dimensional histogram with 32-bin integer bin contents
  TH2* h2i(const std::string& name, const std::string& title, 
      const AxisDef& xaxis, const AxisDef& yaxis);

  /// create new 2-dimensional histogram with double (64-bin) bin contents
  TH2* h2d(const std::string& name, const std::string& title, 
      const AxisDef& xaxis, const AxisDef& yaxis);

  /// create new 2-dimensional histogram with floating (32-bin) bin contents
  TH2* h2f(const std::string& name, const std::string& title, 
      const AxisDef& xaxis, const AxisDef& yaxis);

  /// create new 1-dim profile histogram
  TProfile* profile(const std::string& name, const std::string& title, 
      const AxisDef& xaxis, const std::string& option="");

protected:

  /// Create file if needed, return pointer.
  TFile* file();
  
private:
  
  std::string m_path;              ///< File name
  boost::scoped_ptr<TFile> m_file; ///< Do not use directly, use file() method

};

} // namespace RootHistoManager

#endif // ROOTHISTOMANAGER_ROOTHMGR_H
