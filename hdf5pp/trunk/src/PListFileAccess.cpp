//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListFileAccess...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/PListFileAccess.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

//----------------
// Constructors --
//----------------
PListFileAccess::PListFileAccess ()
  : m_impl()
{
}

//--------------
// Destructor --
//--------------
PListFileAccess::~PListFileAccess ()
{
}

// use family driver
void
PListFileAccess::set_family_driver ( hsize_t memb_size, const PListFileAccess& memb_fapl )
{
  m_impl.setClass(H5P_FILE_ACCESS);
  herr_t stat = H5Pset_fapl_family ( m_impl.id(), memb_size, memb_fapl.plist() ) ;
  if ( stat < 0 ) {
    throw Hdf5CallException ( ERR_LOC, "H5Pset_fapl_family" ) ;
  }
}

// define chuink cache parameters, see for parameter documentation
void
PListFileAccess::set_cache(size_t rdcc_nelmts, size_t rdcc_nbytes, double rdcc_w0)
{
  m_impl.setClass(H5P_FILE_ACCESS);
  herr_t stat = H5Pset_cache(m_impl.id(), 0, rdcc_nelmts, rdcc_nbytes, rdcc_w0);
  if ( stat < 0 ) {
    throw Hdf5CallException ( ERR_LOC, "H5Pset_cache" ) ;
  }
}

void
PListFileAccess::set_fclose_degree(CloseDegree degree)
{
  m_impl.setClass(H5P_FILE_ACCESS);
  H5F_close_degree_t fc_degree;
  bool paramCorrect = true;
  switch (degree) {
  case CloseWeak:
    fc_degree = H5F_CLOSE_WEAK;
    break;
  case CloseSemi:
    fc_degree = H5F_CLOSE_SEMI;
    break;
  case CloseStrong:
    fc_degree = H5F_CLOSE_STRONG;
    break;
  case CloseDefault:
    fc_degree = H5F_CLOSE_DEFAULT;
    break;
  default:
    paramCorrect = false;
  }
   
  if (not paramCorrect) {
    throw Exception ( ERR_LOC,"PListFileAccess::set_fclose_degree",
                      "incorrect degreeClose parameter value" ) ;
  }

  herr_t stat = H5Pset_fclose_degree(m_impl.id(), fc_degree);
  if ( stat < 0 ) {
    throw Hdf5CallException ( ERR_LOC, "H5Pset_fclose_degree" ) ;
  }
}

} // namespace hdf5pp
