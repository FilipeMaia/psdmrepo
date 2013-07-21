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
#include "RdbMySQL/RowIter.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/Result.h"
#include "RdbMySQL/Row.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RdbMySQL {

/// advance, return true if there is a row to extract
bool 
RowIter::next()
{
  _row = _res.fetch_row() ;
  return _row ;
}

/// get the current row
Row
RowIter::row() const
{
  unsigned long* lengths = _res.fetch_lengths();
  return Row ( _row, lengths ) ;
}

} // namespace RdbMySQL
