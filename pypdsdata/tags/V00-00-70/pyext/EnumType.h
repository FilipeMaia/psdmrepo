#ifndef PYPDSDATA_ENUMTYPE_H
#define PYPDSDATA_ENUMTYPE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EnumType.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include "python/Python.h"
#include <map>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

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

class EnumType {
public:

  struct Enum { const char* name; int value; };

  // Constructor takes two arguments: name which may be a dotted
  // name including package and module name, and array of enums,
  // last array entry must have zero name pointer
  EnumType ( const char* name, Enum* enums ) ;

  // Destructor
  ~EnumType () ;

  // Returns _borrowed_ reference to Python type object
  PyObject* type() const {
    return static_cast<PyObject*>((void*)&m_type); 
  }

  // Make instance of this type, returns new reference
  PyObject* Enum_FromLong( long value ) const;
  PyObject* Enum_FromString( const char* name ) const;

protected:

private:

  typedef std::map<long,PyObject*> Int2Enum;

  // Data members
  PyTypeObject m_type ;
  Int2Enum m_int2enum;

  // Copy constructor and assignment are disabled by default
  EnumType ( const EnumType& ) ;
  EnumType& operator = ( const EnumType& ) ;

};

} // namespace pypdsdata

#endif // PYPDSDATA_ENUMTYPE_H
