//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class hdf5_type_info...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <iomanip>

//----------------------
// Base Class Headers --
//----------------------
#include "hdf5/hdf5.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std ;

namespace {

  void print_type_info ( hid_t tid, const char* name )
  {
    cout << name << '\n' ;
    cout << "    H5Tget_class      : " << int(H5Tget_class(tid)) << '\n' ;
    cout << "    H5Tget_size       : " << H5Tget_size(tid) << '\n' ;
    cout << "    H5Tis_variable_str: " << int(H5Tis_variable_str( tid )) << '\n' ;
    if ( H5Tget_class(tid) == H5T_STRING ) {
      cout << "    H5Tget_strpad     : " << int(H5Tget_strpad( tid )) << '\n' ;
    }
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

int main( int, char** )
{
  hid_t mystr_t = H5Tcopy ( H5T_C_S1 ) ;
  H5Tset_size( mystr_t, H5T_VARIABLE ) ;

  print_type_info ( mystr_t, "mystr_t" ) ;
  print_type_info ( H5T_C_S1, "H5T_C_S1" ) ;
  print_type_info ( H5T_FORTRAN_S1, "H5T_FORTRAN_S1" ) ;

  print_type_info ( H5T_IEEE_F32BE, "H5T_IEEE_F32BE" ) ;
  print_type_info ( H5T_IEEE_F32LE, "H5T_IEEE_F32LE" ) ;
  print_type_info ( H5T_IEEE_F64BE, "H5T_IEEE_F64BE" ) ;
  print_type_info ( H5T_IEEE_F64LE, "H5T_IEEE_F64LE" ) ;

  print_type_info ( H5T_STD_I8BE, "H5T_STD_I8BE" ) ;
  print_type_info ( H5T_STD_I8LE, "H5T_STD_I8LE" ) ;
  print_type_info ( H5T_STD_I16BE, "H5T_STD_I16BE" ) ;
  print_type_info ( H5T_STD_I16LE, "H5T_STD_I16LE" ) ;
  print_type_info ( H5T_STD_I32BE, "H5T_STD_I32BE" ) ;
  print_type_info ( H5T_STD_I32LE, "H5T_STD_I32LE" ) ;
  print_type_info ( H5T_STD_I64BE, "H5T_STD_I64BE" ) ;
  print_type_info ( H5T_STD_I64LE, "H5T_STD_I64LE" ) ;
  print_type_info ( H5T_STD_U8BE, "H5T_STD_U8BE" ) ;
  print_type_info ( H5T_STD_U8LE, "H5T_STD_U8LE" ) ;
  print_type_info ( H5T_STD_U16BE, "H5T_STD_U16BE" ) ;
  print_type_info ( H5T_STD_U16LE, "H5T_STD_U16LE" ) ;
  print_type_info ( H5T_STD_U32BE, "H5T_STD_U32BE" ) ;
  print_type_info ( H5T_STD_U32LE, "H5T_STD_U32LE" ) ;
  print_type_info ( H5T_STD_U64BE, "H5T_STD_U64BE" ) ;
  print_type_info ( H5T_STD_U64LE, "H5T_STD_U64LE" ) ;
  print_type_info ( H5T_STD_B8BE, "H5T_STD_B8BE" ) ;
  print_type_info ( H5T_STD_B8LE, "H5T_STD_B8LE" ) ;
  print_type_info ( H5T_STD_B16BE, "H5T_STD_B16BE" ) ;
  print_type_info ( H5T_STD_B16LE, "H5T_STD_B16LE" ) ;
  print_type_info ( H5T_STD_B32BE, "H5T_STD_B32BE" ) ;
  print_type_info ( H5T_STD_B32LE, "H5T_STD_B32LE" ) ;
  print_type_info ( H5T_STD_B64BE, "H5T_STD_B64BE" ) ;
  print_type_info ( H5T_STD_B64LE, "H5T_STD_B64LE" ) ;
  print_type_info ( H5T_STD_REF_OBJ, "H5T_STD_REF_OBJ" ) ;
  print_type_info ( H5T_STD_REF_DSETREG, "H5T_STD_REF_DSETREG" ) ;

  print_type_info ( H5T_UNIX_D32BE, "H5T_UNIX_D32BE" ) ;
  print_type_info ( H5T_UNIX_D32LE, "H5T_UNIX_D32LE" ) ;
  print_type_info ( H5T_UNIX_D64BE, "H5T_UNIX_D64BE" ) ;
  print_type_info ( H5T_UNIX_D64LE, "H5T_UNIX_D64LE" ) ;

}


