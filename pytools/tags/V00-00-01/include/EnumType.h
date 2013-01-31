#ifndef PYTOOLS_ENUMTYPE_H
#define PYTOOLS_ENUMTYPE_H

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
#include <boost/utility.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pytools {

/// @addtogroup pytools

/**
 *  @ingroup pytools
 *
 *  @brief Emulation of C++ enum type for Python.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class EnumType : boost::noncopyable {
public:

  /// Enum is a name plus integer value
  struct Enum { const char* name; int value; };

  /**
   *  This constructor takes the name of this type which may be a dotted
   *  name including package and module name.
   */
  EnumType(const char* typeName);

  /**
   *  This constructor takes two arguments: name which may be a dotted
   *  name including package and module name, and array of enums,
   *  last array entry must have zero name pointer
   */
  EnumType(const char* typeName, Enum* enums);

  /**
   *  Add one more enum value to the type.
   */
  void addEnum(const std::string& name, int value);

  /**
   *  Returns type name
   */
  const char* typeName() const { return m_typeName; }

  // Destructor
  ~EnumType () ;

  // Returns _borrowed_ reference to Python type object
  PyObject* type() const {
    return static_cast<PyObject*>((void*)&m_type); 
  }

  // Make instance of this type, returns new reference
  PyObject* Enum_FromLong(long value) const;
  PyObject* Enum_FromString(const char* name) const;

protected:

  void initType(const char* typeName);
  void makeDocString();

private:

  typedef std::map<long,PyObject*> Int2Enum;

  // Data members
  char* m_typeName;
  PyTypeObject m_type;
  Int2Enum m_int2enum;
  char* m_docString;

};

} // namespace pytools

#endif // PYTOOLS_ENUMTYPE_H
