#ifndef RDBMYSQL_FIELD_HH
#define RDBMYSQL_FIELD_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Field.
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//
// Author List:
//      Andy Salnikov
//
// Copyright Information:
//      Copyright (C) 2005 SLAC
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
struct st_mysql_field ;

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RdbMySQL {

/**
 *  This class represents the heared of the result set.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Field {

public:

  /**
   *  Constructor takes MYSQL_FIELD list
   */
  Field( st_mysql_field* field ) : _field(field) {}

  // Destructor
  ~Field () {}

  /// get the name of the field
  const char* name() const ;

  /// get the name of the table for the field, of empty string for computed fields
  const char* table() const ;

  /// The default value of this field, as a null-terminated string
  const char* def() const ;

  /// get the type of the field (FIELD_TYPE_*)
  int type() const ;

  /// The width of the field, as specified in the table definition
  unsigned int length() const ;

  /// The maximum width of the field for the result set 
  unsigned int max_length() const ;

  /// Different bit-flags for the field
  unsigned int flags() const ;

protected:

  // Helper functions

private:

  // Friends

  // Data members
  st_mysql_field*  _field ;

};

} // namespace RdbMySQL

#endif // RDBMYSQL_FIELD_HH
