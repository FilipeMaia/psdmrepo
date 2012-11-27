#ifndef H5DATATYPES_EVRDATAV3_H
#define H5DATATYPES_EVRDATAV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrDataV3.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/evr/DataV3.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct EvrDataV3_Data {
  size_t numFifoEvents;
  const Pds::EvrData::DataV3::FIFOEvent* fifoEvents;
};

class EvrDataV3 : boost::noncopyable {
public:

  typedef Pds::EvrData::DataV3 XtcType ;

  EvrDataV3 () ;
  EvrDataV3 ( const XtcType& data ) ;

      // destructor
  void destroy();

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size(); }

protected:

private:

  EvrDataV3_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRDATAV3_H
