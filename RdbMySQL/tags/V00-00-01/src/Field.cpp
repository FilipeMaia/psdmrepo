//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Row
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Andy Salnikov
//
// Copyright Information:
//      Copyright (C) 2005 SLAC
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "RdbMySQL/Field.h"

//---------------
// C++ Headers --
//---------------
#include <mysql/mysql.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RdbMySQL {

/// get the name of the field
const char* 
Field::name() const
{ 
  return _field->name ; 
}

/// get the name of the table for the field, of empty string for computed fields
const char* 
Field::table() const
{ 
  return _field->table ; 
}

/// The default value of this field, as a null-terminated string
const char* 
Field::def() const
{ 
  return _field->def ; 
}

/// get the type of the field (FIELD_TYPE_*)
int 
Field::type() const
{ 
  return int(_field->type) ; 
}

/// The width of the field, as specified in the table definition
unsigned int 
Field::length() const
{ 
  return _field->length ; 
}

/// The maximum width of the field for the result set 
unsigned int 
Field::max_length() const
{ 
  return _field->max_length ; 
}

/// Different bit-flags for the field
unsigned int 
Field::flags() const
{ 
  return _field->flags ; 
}

} // namespace RdbMySQL
