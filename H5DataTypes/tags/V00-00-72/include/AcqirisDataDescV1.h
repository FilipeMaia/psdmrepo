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
#include "hdf5pp/Group.h"
#include "pdsdata/acqiris/ConfigV1.hh"
#include "pdsdata/acqiris/DataDescV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
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

struct AcqirisDataDescV1_Data  {
};

class AcqirisDataDescV1  {
public:

  typedef Pds::Acqiris::DataDescV1 XtcType ;

  AcqirisDataDescV1 () ;
  AcqirisDataDescV1 ( const XtcType& xtcData ) ;

  static hdf5pp::Type timestampType( const Pds::Acqiris::ConfigV1& config ) ;
  static hdf5pp::Type waveformType( const Pds::Acqiris::ConfigV1& config ) ;

protected:
private:

  AcqirisDataDescV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_ACQIRISDATADESCV1_H
