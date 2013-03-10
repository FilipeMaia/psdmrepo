//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5IterData...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5IterData.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHdf5Input {

/// Standard stream insertion operator for enum type
std::ostream&
operator<<(std::ostream& out, Hdf5IterData::EventType type)
{
  const char* str = "*Unknown*";
  switch(type) {
  case Hdf5IterData::Configure:
    str = "Configure";
    break;
  case Hdf5IterData::BeginRun:
    str = "BeginRun";
    break;
  case Hdf5IterData::BeginCalibCycle:
    str = "BeginCalibCycle";
    break;
  case Hdf5IterData::Event:
    str = "Event";
    break;
  case Hdf5IterData::EndCalibCycle:
    str = "EndCalibCycle";
    break;
  case Hdf5IterData::EndRun:
    str = "EndRun";
    break;
  case Hdf5IterData::UnConfigure:
    str = "UnConfigure";
    break;
  case Hdf5IterData::Stop:
    str = "Stop";
    break;
  }
  return out << str;
}

/// Standard stream insertion operator for data type
std::ostream&
operator<<(std::ostream& out, const Hdf5IterData& data)
{
  out << "Hdf5IterData(type=" << data.type() ;
  if (data.run() >= 0) out << ", run=" << data.run() ;
  if (data.time() != PSTime::Time()) out << ", time=" << data.time() ;
  const Hdf5IterData::seq_type& seq = data.data();
  for (Hdf5IterData::const_iterator it = seq.begin(); it != seq.end(); ++it) {
    out << ", ";
    out << it->group.name() << '[' << it->index << "]";
  }
  return out << ")";
}

} // namespace PSHdf5Input
