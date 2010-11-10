#ifndef H5DATATYPES_CSPADPEDESTALSV1_H
#define H5DATATYPES_CSPADPEDESTALSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPedestalsV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "pdscalibdata/CsPadPedestalsV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for CsPadPedestalsV1
//
struct CsPadPedestalsV1_Data  {

  enum { Quads = pdscalibdata::CsPadPedestalsV1::Quads };
  enum { Sections = pdscalibdata::CsPadPedestalsV1::Sections };
  enum { Columns = pdscalibdata::CsPadPedestalsV1::Columns };
  enum { Rows = pdscalibdata::CsPadPedestalsV1::Rows };
  enum { Size = Quads*Sections*Columns*Rows };
  
  float pedestals[Quads][Sections][Columns][Rows];
};

/**
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPadPedestalsV1  {
public:

  typedef pdscalibdata::CsPadPedestalsV1 DataType ;
  
  // Default constructor
  CsPadPedestalsV1 () ;
  
  // read pedestals from file
  CsPadPedestalsV1 (const DataType& data) ;

  // Destructor
  ~CsPadPedestalsV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single object at specified location
  static void store( const DataType& data, hdf5pp::Group location ) ;
  
protected:

private:

  // Data members
  CsPadPedestalsV1_Data m_data;

  // Copy constructor and assignment are disabled by default
  CsPadPedestalsV1 ( const CsPadPedestalsV1& ) ;
  CsPadPedestalsV1& operator = ( const CsPadPedestalsV1& ) ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADPEDESTALSV1_H
