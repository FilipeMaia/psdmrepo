#ifndef H5DATATYPES_ACQIRISDATADESCV1_H
#define H5DATATYPES_ACQIRISDATADESCV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisDataDescV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

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

/**
 *  C++ source file code template. The first sentence is a brief summary of 
 *  what the class is for. It is followed by more detailed information
 *  about how to use the class. This doc comment must immediately preceed the 
 *  class definition.
 *
 *  Additional paragraphs with more details may follow; separate paragraphs
 *  with a blank line. The last paragraph before the tags (preceded by @) 
 *  should be the identification and copyright, as below.
 *
 *  Please note that KDOC comments must start with /** (a forward slash
 *  followed by TWO asterisks). Interface members should be documented
 *  with KDOC comments, as should be protected members that may be of interest
 *  to those deriving from your class. Private implementation should
 *  be commented with C++-style // (double forward slash) comments.
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

namespace H5DataTypes {

class AcqirisDataDescV1  {
public:

  // Default constructor
  AcqirisDataDescV1 () ;

  // Destructor
  virtual ~AcqirisDataDescV1 () ;

protected:

private:

  // Data members
  
  int m_memberVariable;  // private members start with m_

  // Copy constructor and assignment are disabled by default
  AcqirisDataDescV1 ( const AcqirisDataDescV1& ) ;
  AcqirisDataDescV1& operator = ( const AcqirisDataDescV1& ) ;

//------------------
// Static Members --
//------------------

public:

  // Selectors (const)

  // Modifiers

private:

  // Data members
  static int s_staticVariable;     // Static data member starts with s_.

};

} // namespace H5DataTypes

#endif // H5DATATYPES_ACQIRISDATADESCV1_H
