//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5FileListIter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5FileListIter.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSHdf5Input/Hdf5FileIter.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHdf5Input {

//----------------
// Constructors --
//----------------
Hdf5FileListIter::Hdf5FileListIter (const std::list<std::string>& fileNames)
  : m_fileNames(fileNames)
  , m_fileIter()
{
}

//--------------
// Destructor --
//--------------
Hdf5FileListIter::~Hdf5FileListIter ()
{
}

/// Returns next object
Hdf5FileListIter::value_type 
Hdf5FileListIter::next()
{
  while (true) {

    if (not m_fileIter.get()) {
    
      // no more files left - we are done
      if (m_fileNames.empty()) return value_type(value_type::Stop);

      // open next file
      std::string fileName = m_fileNames.front();
      m_fileNames.pop_front();      
      m_fileIter.reset(new Hdf5FileIter(fileName));
      
    }
    
    // read next event from file
    value_type res = m_fileIter->next();

    // close the file if it sends us Stop
    if (res.type() == value_type::Stop) {
      m_fileIter.reset();
    } else {
      return res;
    }
  }

}

} // namespace PSHdf5Input
