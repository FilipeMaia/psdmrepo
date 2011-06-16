#ifndef ROOTHIST_ROOTCOLUMN_H
#define ROOTHIST_ROOTCOLUMN_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RootColumn.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <iostream>
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//#include "PSHist/Tuple.h"
#include "PSHist/Column.h"
#include "RootHist/RootTuple.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "root/TBranch.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

class TBranch;

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RootHist {

/**
 *  This class implements the tuple parameter (Column) which is defined in ROOT as TBranch.
 *  
 *  Essentially, constructor of this class creates the type-independent pointer TBranch *m_column.
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class RootColumn : public PSHist::Column {
public:

  // Constructors
  RootColumn () {}
  RootColumn ( RootTuple* tuple, const std::string &name, void* address, const std::string &columnlist );

  // Destructor
  virtual ~RootColumn(){}

  // Methods

  virtual void print(std::ostream &o) const;

private:

  // Data members
  TBranch *m_column;

  // Static members

  // Copy constructor and assignment are disabled by default
  RootColumn              ( const RootColumn& );
  RootColumn & operator = ( const RootColumn& );

};

} // namespace RootHist

#endif // ROOTHIST_ROOTCOLUMN_H
