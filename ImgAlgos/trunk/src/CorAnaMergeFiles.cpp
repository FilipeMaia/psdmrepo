//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaMergeFiles...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CorAnaMergeFiles.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iomanip>
#include <cmath> // for sqrt, atan2

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/GlobalMethods.h"

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
CorAnaMergeFiles::CorAnaMergeFiles(): CorAna ()
{
  m_timer1 = new TimeInterval();
  m_log << "CorAnaMergeFiles::CorAnaMergeFiles(): Start merging of blocks vs index -> image vs index of tau " 
        << m_timer1->strStartTime() << "\n";

  openFiles();
  mergeFiles();
  closeFiles();

  m_log << "CorAnaMergeFiles::CorAnaMergeFiles(): Merging time: " 
        << m_timer1->getCurrentTimeInterval() << "sec\n";
  m_log << "CorAnaMergeFiles::CorAnaMergeFiles(): resulting file for image vs index of tau is:" << m_fname_result_img << "\n";
}

//--------------
// Destructor --
//--------------
CorAnaMergeFiles::~CorAnaMergeFiles ()
{
}

//----------------

void
CorAnaMergeFiles::openFiles()
{
  m_log << "\nCorAnaMergeFiles::openFiles():";

  p_inp = new std::ifstream [m_nfiles];
  ios_base::openmode mode = ios_base::in | ios_base::binary;

  for (unsigned b=0; b<m_nfiles; b++) {
    std::string fname = m_fname_com + "-b" + stringFromUint(b,4) + "-result.bin";
    m_log << "\n    open inp file: " << fname;
    p_inp[b].open(fname.c_str(), mode);
  }

  m_log << "\n    open out file: " << m_fname_result_img;
  p_out.open((m_fname_result_img).c_str(), ios_base::out | ios_base::binary);
}

//----------------

void 
CorAnaMergeFiles::closeFiles()
{
  for(unsigned b=0; b<m_nfiles; b++) p_inp[b].close();
  p_out.close();
}

//----------------

void
CorAnaMergeFiles::mergeFiles()
{
  m_log << "\nCorAnaMergeFiles::mergeFiles(): Merge " 
        << m_nfiles << " blocks for " 
        << m_npoints_tau << " points in tau in a single image vs tau index \n";

  cor_t* buf = new cor_t [m_blk_size]; // the same size for gi, gf, and g2 correlators

  for(unsigned itau=0; itau<m_npoints_tau; itau++) {

    for(unsigned ig=0; ig<3; ig++) { // loop over gi, gf, and g2 blocks

      for(unsigned b=0; b<m_nfiles; b++) {

	unsigned buf_size = m_blk_size*sizeof(cor_t);
        p_inp[b].read((char*)buf, buf_size);
        if(!p_inp[b].good()) {
          m_log << "CorAnaMergeFiles::mergeFiles(): Input buffer is not good for itau:" << itau << " ig:" << ig  << " blk:" << b << "\n";
          	break;
        }
        p_out.write(reinterpret_cast<const char*>(buf), buf_size); 
      }
    }
  }
}

//----------------
//----------------

} // namespace ImgAlgos
