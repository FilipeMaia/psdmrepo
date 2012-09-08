//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaData...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CorAnaData.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/CorAnaInputParameters.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CorAnaData::CorAnaData(): m_olog(INPARS->get_ostream())
{
  m_olog << "C-tor: CorAnaData()\n";

  readMetadataFile();

}

//--------------
// Destructor --
//--------------
CorAnaData::~CorAnaData ()
{
}

//----------------

void
CorAnaData::readMetadataFile()
{
  std::vector<std::string>&  v_names = INPARS -> get_vector_fnames();    
  std::string& fname = v_names[0];
  std::string  fname_com = fname.substr ( 0, fname.rfind("-b") );
  std::string  fname_med = fname_com; fname_med += ".med";

  m_olog << "Data file name: " << fname     << "\n";
  m_olog << "Comm file name: " << fname_com << "\n";
  m_olog << "Med  file name: " << fname_med << "\n";
}

//----------------
//----------------
//----------------
//----------------

} // namespace ImgAlgos
