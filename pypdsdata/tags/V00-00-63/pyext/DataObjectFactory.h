#ifndef PYPDSDATA_DATAOBJECTFACTORY_H
#define PYPDSDATA_DATAOBJECTFACTORY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataObjectFactory.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <python/Python.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Xtc.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

/**
 *  Utility class which implements a factory method for building
 *  Python object corresponding to data classes from XTC.
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

class DataObjectFactory  {
public:

  /**
   * Factory method which creates Python objects from XTC
   */
  static PyObject* makeObject( const Pds::Xtc& xtc, PyObject* parent );

protected:

  // Default constructor
  DataObjectFactory () {}

  // Destructor
  ~DataObjectFactory () {}

private:

  // Copy constructor and assignment are disabled by default
  DataObjectFactory ( const DataObjectFactory& ) ;
  DataObjectFactory& operator = ( const DataObjectFactory& ) ;

};

} // namespace pypdsdata

#endif // PYPDSDATA_DATAOBJECTFACTORY_H
