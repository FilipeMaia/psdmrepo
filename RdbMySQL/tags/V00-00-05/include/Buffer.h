#ifndef RDBMYSQL_BUFFER_HH
#define RDBMYSQL_BUFFER_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Buffer.
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

namespace RdbMySQL {

/**
 *  This is a "string" which uses existing data buffers for data storage 
 *  instead of copying it.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see Query
 *  @see QueryBuf
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Buffer {

public:

  typedef unsigned long size_type ;


  /**
   *  Constructor takes pointer to a data buffer and data size.
   */
  Buffer ( const char* data, size_type size ) : _data(data), _size(size) {}

  // Destructor
  ~Buffer () {}

  /// get the pointer to the data
  const char* data() const { return _data ; }

  /// get data size 
  size_type size() const { return _size ; }

private:

  // Friends

  // Data members
  const char* _data ;
  size_type   _size ;

};

} // namespace RdbMySQL

#endif // RDBMYSQL_BUFFER_HH
