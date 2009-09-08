#ifndef O2OTRANSLATOR_CVTGROUPMAP_H
#define O2OTRANSLATOR_CVTGROUPMAP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtGroupMap.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/xtc/DetInfo.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Class which provides mapping from a DetInfo obejct to a corresponding
 *  detector group in HDF5 file.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CvtGroupMap  {
public:

  typedef std::vector<hdf5pp::Group> GroupList ;

  // Default constructor
  CvtGroupMap () ;

  // Destructor
  ~CvtGroupMap () ;

  /// find a group for a given top group and DetInfo, return invalid
  /// object if not found
  hdf5pp::Group find( hdf5pp::Group top, Pds::DetInfo info ) const ;

  /// insert new mapping
  void insert ( hdf5pp::Group top, Pds::DetInfo info, hdf5pp::Group group ) ;

  /// remove all mappings for given top group
  void erase ( hdf5pp::Group top ) ;

  /// get the set of subgroups for a given top group
  GroupList groups( hdf5pp::Group top ) const ;

protected:

private:

  // comparison operator for DetInfo objects
  struct _DetInfoCmp {
    bool operator()( const Pds::DetInfo& lhs, const Pds::DetInfo& rhs ) const ;
  };

  typedef std::map<Pds::DetInfo,hdf5pp::Group,_DetInfoCmp> Det2Group ;
  typedef std::map<hdf5pp::Group,Det2Group> Group2Group ;

  // Data members
  Group2Group m_group2group ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTGROUPMAP_H
