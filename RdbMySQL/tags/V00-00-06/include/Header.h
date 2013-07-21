#ifndef RDBMYSQL__HEADER_HH
#define RDBMYSQL__HEADER_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Header.
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
#include "RdbMySQL/Field.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
struct st_mysql_res ;
namespace RdbMySQL {
class Client ;
}

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

class Header {

public:

  /**
   *  Constructor takes MYSQL_FIELD list
   */
  Header( st_mysql_res* res, Client* client ) : _res(res), _client(client) {}

  /// Destructor
  ~Header () {}

  ///  Returns the numbr of fields in a row
  unsigned int size() const ;

  /// Returns true for empty header
  bool empty() const { return size() == 0 ; }

  /// get the field at nth position
  Field field( unsigned int i ) const ;

  /// get the index of the field from its name, returns -1 if no such field exist
  int index ( const char* name ) const ;
  int index ( const std::string& name ) const ;

protected:

  // Helper functions

private:

  // Friends

  // Data members
  st_mysql_res*  _res ;
  Client* _client ;

  // find index from the field name, or -1
  int name2idx ( std::string& name ) const ;

};

} // namespace RdbMySQL

#endif // RDBMYSQL__HEADER_HH
