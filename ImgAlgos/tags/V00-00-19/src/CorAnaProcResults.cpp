//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaProcResults...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CorAnaProcResults.h"

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
CorAnaProcResults::CorAnaProcResults(): CorAna ()
{
  m_timer1 = new TimeInterval();
  m_log << "CorAnaProcResults::CorAnaProcResults(): Start job at" 
        << m_timer1->strStartTime() << "\n";

  readCorFile();
  fillHistogram();
  saveHistogramInFile();

  m_log << "CorAnaProcResults::CorAnaProcResults(): Finish job in " 
        << m_timer1->getCurrentTimeInterval() << "sec\n";
}

//--------------
// Destructor --
//--------------
CorAnaProcResults::~CorAnaProcResults ()
{
}

//----------------

void
CorAnaProcResults::readCorFile()
{
  m_log << "CorAnaProcResults::readCorFile(): Read correlations vs tau index from file: " << m_fname_result_img << "\n";

  std::fstream inf(m_fname_result_img.c_str(), std::ios::in | std::ios::binary);
  if (!inf.is_open()) {
     const std::string msg = "CorAnaProcResults::readCorFile(): Unable to open file: " + m_fname_result_img + "\n"; 
     m_log << msg;  
     abort();
  }

  m_cor = new cor_t [m_img_size * m_npoints_tau];
  inf.read((char*)m_cor, sizeof(cor_t) * m_img_size * m_npoints_tau); 
  inf.close();

  m_log << "CorAnaProcResults::readCorFile(): Array is loaded from file.\n";
}

//----------------

unsigned
CorAnaProcResults::getBinInImg(unsigned pix)
{
  int  r = (pix < m_img_cols) ?   0 : pix/m_img_cols;
  int  c = (pix < m_img_cols) ? pix : pix%m_img_cols;
  int dx = r - m_row_c; 
  int dy = c - m_col_c; 
  double  R = std::sqrt(dx*dx+dy*dy);
  return (R < m_radmax) ? static_cast<unsigned>(R/m_radbin) : m_nbins-1;
}

//----------------

void
CorAnaProcResults::fillHistogram()
{
  m_log << "CorAnaProcResults::fillHistograms()\n";

  m_nbins  = 12;
  m_hsize  = m_npoints_tau * m_nbins;
  m_row_c  = m_img_rows/2;
  m_col_c  = m_img_cols/2;
  m_radmax = std::min(m_row_c, m_col_c);
  m_radbin = m_radmax/m_nbins;

  m_sum0 = new unsigned [m_hsize];
  m_sum1 = new double   [m_hsize];
  m_hist = new hist_t   [m_hsize];

  std::fill_n(m_sum0, m_hsize, unsigned(0));
  std::fill_n(m_sum1, m_hsize, double(0));
  std::fill_n(m_hist, m_hsize, hist_t(0));

  for(unsigned itau=0; itau<m_npoints_tau; itau++) {

    cor_t* p_img_cor = &m_cor[m_img_size*itau];

    for(unsigned pix=0; pix<m_img_size; pix++) {

      unsigned hind = m_nbins * itau + getBinInImg(pix);
      m_sum0[hind] += 1; 
      m_sum1[hind] += p_img_cor[pix]; 
    }

    for(unsigned hind=0; hind<m_hsize; hind++) {
      m_hist[hind] = (m_sum0[hind]) ? m_sum1[hind]/m_sum0[hind] : 0;
    }
  }
}

//----------------

void
CorAnaProcResults::saveHistogramInFile()
{
  m_log << "CorAnaProcResults::saveHistogramInFile():" + m_fname_hist + "\n"; 

  std::ofstream out(m_fname_hist.c_str());

  for(unsigned itau=0; itau<m_npoints_tau; itau++) {

    out << std::setw(8) << v_ind_tau[itau];

    for(unsigned bin=0; bin<m_nbins; bin++) 
      out << std::fixed << std::setw(12) << std::setprecision(6) << 100*m_hist[m_nbins * itau + bin];
    out << " \n";
  }

  out.close();
}

//----------------
//----------------
//----------------
//----------------
//----------------
//----------------

} // namespace ImgAlgos
