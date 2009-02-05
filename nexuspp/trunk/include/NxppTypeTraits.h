#ifndef NEXUSPP_NXPPTYPETRAITS_H
#define NEXUSPP_NXPPTYPETRAITS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NxppTypeTraits.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <stdint.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "nexus/napi.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Type traits package.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace nexuspp {

template <typename T>
struct NxppTypeTraits  {
};

template <typename T, int NxType>
struct NxppTypeTraitsNum {

  // nxtype determines nexus type, not known for unspecialized version
  enum { nxtype = NxType } ;

  // get the address of the data item
  static void* dataAddress( T* ptr ) { return static_cast<void*>(ptr); }
  static const void* dataAddress( const T& val ) { return static_cast<const void*>(&val); }

  // size() returns the size of the data value
  // in terms of the 'nxtype' type
  static int size( T val ) { return 1 ; }
};

template <>
struct NxppTypeTraits<float> : public NxppTypeTraitsNum<float,NX_FLOAT32> {
};

template <>
struct NxppTypeTraits<double> : public NxppTypeTraitsNum<double,NX_FLOAT64> {
};

template <>
struct NxppTypeTraits<int8_t> : public NxppTypeTraitsNum<int8_t,NX_INT8> {
};

template <>
struct NxppTypeTraits<uint8_t> : public NxppTypeTraitsNum<uint8_t,NX_UINT8> {
};

template <>
struct NxppTypeTraits<int16_t> : public NxppTypeTraitsNum<int16_t,NX_INT16> {
};

template <>
struct NxppTypeTraits<uint16_t> : public NxppTypeTraitsNum<uint16_t,NX_UINT16> {
};

template <>
struct NxppTypeTraits<int32_t> : public NxppTypeTraitsNum<int32_t,NX_INT32> {
};

template <>
struct NxppTypeTraits<uint32_t> : public NxppTypeTraitsNum<uint32_t,NX_UINT32> {
};

template <>
struct NxppTypeTraits<std::string>  {
  enum { nxtype = NX_CHAR } ;
  static void* dataAddress( const std::string& val ) { return (void*)(val.c_str()); }
  static int size( const std::string& val ) { return val.size() ; }
};


} // namespace nexuspp

#endif // NEXUSPP_NXPPTYPETRAITS_H
