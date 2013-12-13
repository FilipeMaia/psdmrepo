#ifndef H5DATATYPES_ALIASCONFIGV1_H
#define H5DATATYPES_ALIASCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AliasConfigV1.
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
#include "hdf5pp/Group.h"
#include "pdsdata/psddl/alias.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
//  Helper class for Pds::Src class
//
struct AliasPdsSrc {
public:

  AliasPdsSrc () {}
  AliasPdsSrc ( const Pds::Src& src ) : log(src.log()), phy(src.phy()) {}

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  uint32_t log;
  uint32_t phy;
};

//
//  Helper class for Pds::Alias::SrcAlias class
//
struct AliasSrcAlias {
public:

  enum { AliasNameMax = Pds::Alias::SrcAlias::AliasNameMax };

  AliasSrcAlias () {}
  AliasSrcAlias ( const Pds::Alias::SrcAlias& srcAlias ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  AliasPdsSrc src;
  char aliasName[AliasNameMax];
};

//
//  Helper class for Pds::Alias::ConfigV1 class
//
class AliasConfigV1  {
public:

  typedef Pds::Alias::ConfigV1 XtcType ;

  AliasConfigV1 () {}
  AliasConfigV1 ( const XtcType& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof() ; }

protected:

private:

  uint32_t numSrcAlias;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_ALIASCONFIGV1_H
