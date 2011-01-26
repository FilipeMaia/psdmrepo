#ifndef O2OTRANSLATOR_CVTDATACONTAINER_H
#define O2OTRANSLATOR_CVTDATACONTAINER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtDataContainer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/ObjectContainer.h"
#include "hdf5pp/Group.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
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

template <typename Factory>
class CvtDataContainer {
public:

  // container type
  typedef typename Factory::container_type container_type ;

  // Constructor
  CvtDataContainer( const Factory& factory )
    : m_factory(factory)
    , m_contMap()
  {
  }

  // Destructor
  ~CvtDataContainer ()
  {
    // delete all remaining stuff if anything is left
    for ( typename ContMap::iterator x = m_contMap.begin() ; x != m_contMap.end() ; ++ x ) {
      delete x->second ;
    }
    m_contMap.clear() ;
  }

  // Get the container for a given parent group
  container_type* container( hdf5pp::Group group )
  {
    // if container already exist just return it
    typename ContMap::iterator it = m_contMap.find( group );
    if ( it != m_contMap.end() ) return it->second ;

    // create new container using factory object
    container_type* cont = m_factory.container( group ) ;

    // store it for the future
    m_contMap.insert ( typename ContMap::value_type( group, cont ) ) ;

    return cont ;
  }

  // Get the container for a given parent group, method that takes additional
  // parameter that gets passed to the factory
  template <typename Extra>
  container_type* container( hdf5pp::Group group, const Extra& extra )
  {
    // if container already exist just return it
    typename ContMap::iterator it = m_contMap.find( group );
    if ( it != m_contMap.end() ) return it->second ;

    // create new container using factory object
    container_type* cont = m_factory.container( group, extra ) ;

    // store it for the future
    m_contMap.insert ( typename ContMap::value_type( group, cont ) ) ;

    return cont ;
  }

  // close the container for a specific group
  void closeGroup( hdf5pp::Group group )
  {
    // delete container associated with the group
    typename ContMap::iterator it = m_contMap.find( group );
    if ( it != m_contMap.end() ) {
      container_type* cont = it->second ;
      m_contMap.erase( it ) ;
      delete cont ;
    }
  }

protected:

private:

  typedef typename std::map<hdf5pp::Group,container_type*> ContMap ;

  // Data members
  Factory m_factory ;
  ContMap m_contMap ;

  // Copy constructor and assignment are disabled by default
  CvtDataContainer ( const CvtDataContainer& ) ;
  CvtDataContainer operator = ( const CvtDataContainer& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTDATACONTAINER_H
