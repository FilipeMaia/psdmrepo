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

namespace RootHistoManager {

/**
 *  ROOT-specific histogram manager for psana. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class RootHMgr : boost::noncopyable {
public:

  // Default constructor
  RootHMgr (const std::string& path) ;

  // Destructor
  ~RootHMgr () ;

  /// create new 1-dim histogram with 32-bin integer bin contents
  TH1* h1i(const std::string& name, const std::string& title, const AxisDef& axis);

  /// create new 1-dim histogram with double (64-bin) bin contents
  TH1* h1d(const std::string& name, const std::string& title, const AxisDef& axis);

  /// create new 1-dim histogram with floating (32-bin) bin contents
  TH1* h1f(const std::string& name, const std::string& title, const AxisDef& axis);

  /// create new 2-dim histogram with 32-bin integer bin contents
  TH2* h2i(const std::string& name, const std::string& title, 
      const AxisDef& xaxis, const AxisDef& yaxis);

  /// create new 2-dim histogram with double (64-bin) bin contents
  TH2* h2d(const std::string& name, const std::string& title, 
      const AxisDef& xaxis, const AxisDef& yaxis);

  /// create new 1-dim histogram with floating (32-bin) bin contents
  TH2* h2f(const std::string& name, const std::string& title, 
      const AxisDef& xaxis, const AxisDef& yaxis);

  /// create new 1-dim profile histogram
  TProfile* profile(const std::string& name, const std::string& title, 
      const AxisDef& xaxis, const std::string& option="");

protected:

  TFile* file();
  
private:
  
  std::string m_path;
  boost::scoped_ptr<TFile> m_file;

};

} // namespace RootHistoManager

#endif // ROOTHISTOMANAGER_ROOTHMGR_H
