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
//#include <ifstream>
#include <iomanip>
#include <stdlib.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

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
CorAnaData::CorAnaData(): CorAna ()
{
  m_timer1 = new TimeInterval();
  m_log << "CorAnaData::CorAnaData(): Job is started at " 
        << m_timer1->strStartTime() << "\n";
  readDataFile();
  m_log << "CorAnaData::CorAnaData(): Data reading and initialization time: " 
        << m_timer1->getCurrentTimeInterval() << "sec\n";

  printData();
  loopProcCorTau();
}

//--------------
// Destructor --
//--------------
CorAnaData::~CorAnaData ()
{
}

//----------------

void
CorAnaData::readDataFile()
{
  m_log << "\nCorAnaData::readDataFile(): Read data from file: " << m_fname << "\n";

  std::fstream inf(m_fname.c_str(), std::ios::in | std::ios::binary);
  if (!inf.is_open()) {
     const std::string msg = "CorAnaData::readDataFile(): Unable to open file: " + m_fname + "\n"; 
     m_log << msg;  
     abort();
  }

  m_data = new data_t [m_blk_size * m_nimgs];
  //inf.seekg(0);
  inf.read((char*)m_data, sizeof(data_t) * m_blk_size * m_nimgs); 

  inf.close();
}

//----------------

void
CorAnaData::printData()
{
  m_log << "\nCorAnaData::printData(): Data from file: " << m_fname;

  for(unsigned r=0; r<m_nimgs; r++) {
    if ( r<10 
      || (r<100 && r%10 == 0) 
      || r%100== 0
      || r==m_nimgs-1 )
      {
        m_log << "\nImg-blk " << std::setw(4) << r << ":" << std::setprecision(2);
        for(unsigned c=0; c<10; c++)                     m_log << " "  << std::setw(4) << m_data[r*m_blk_size + c];
	m_log << " ...";
        for(unsigned c=m_blk_size-10; c<m_blk_size; c++) m_log << " "  << std::setw(4) << m_data[r*m_blk_size + c];
      }
  }
  m_log << "\n";
}

//----------------

void
CorAnaData::loopProcCorTau()
{
  m_timer1->startTime();
  unsigned counter=0;
  initCorTau();

  std::ofstream ofs(m_fname_result.c_str());

  for(vector<unsigned>::const_iterator it = v_ind_tau.begin(); 
                                       it!= v_ind_tau.end(); it++) {
    evaluateCorTau(*it);
    saveCorTau(ofs);
    //printCorTau(*it);
    counter++; 
  }

  ofs.close();

  m_log << "\nCorAnaData::loopProcCorTau(): Correlation processing time =" << m_timer1->getCurrentTimeInterval() << "sec for "
        << counter << " intevals in tau.";
}

//----------------

void
CorAnaData::initCorTau()
{
  m_log << "\nCorAnaData::initCorTau(): " << m_fname;
  m_sum_gi = new double   [m_blk_size];
  m_sum_gf = new double   [m_blk_size];
  m_sum_g2 = new double   [m_blk_size];
  m_sum_st = new unsigned [m_blk_size];
  m_cor_gi = new cor_t    [m_blk_size];
  m_cor_gf = new cor_t    [m_blk_size];
  m_cor_g2 = new cor_t    [m_blk_size];
}

//----------------

void
CorAnaData::evaluateCorTau(unsigned tau) // tau in number of frames between images
{
  m_log << "\nCorAnaData::evaluateCorTau(tau): tau=" << tau;
  std::fill_n(m_sum_gi, m_blk_size, double(0));
  std::fill_n(m_sum_gf, m_blk_size, double(0));
  std::fill_n(m_sum_g2, m_blk_size, double(0));
  std::fill_n(m_sum_st, m_blk_size, unsigned(0));
  std::fill_n(m_cor_gi, m_blk_size, cor_t(0));
  std::fill_n(m_cor_gf, m_blk_size, cor_t(0));
  std::fill_n(m_cor_g2, m_blk_size, cor_t(0));


  for (unsigned ti=0; ti<m_tind_size-tau; ti++) {
       unsigned tf=ti+tau;

       if ( ! (tf<m_tind_size) ) break;
       
       // get the event index in array for time index
       int evi = m_tind_to_evind[ti];
       int evf = m_tind_to_evind[tf];

       // If the event does not exist for specified time index -> skip it in sum
       if(evi<0) continue;
       if(evf<0) continue;

       sumCorTau((unsigned)evi,(unsigned)evf);
  }
}

//----------------

void
CorAnaData::sumCorTau(unsigned i, unsigned f) // i and f are the event indexes
{
  /*
    if ( i<5 
      || i<50 && i%10 == 0 
      || i%100== 0
      || i==m_nimgs-(f-i)-1 ) 
      m_log << "\nCorAnaData::sumCorTau(i,f): tau=" << f-i << "  i, f=" << i << ", " << f;
  */

   data_t* p_i = &m_data[i*m_blk_size];
   data_t* p_f = &m_data[f*m_blk_size];
   double Ii, If;

   for(unsigned pix=0; pix<m_blk_size; pix++) {
     Ii = p_i[pix]; 
     If = p_f[pix]; 
     m_sum_gi[pix] += Ii; 
     m_sum_gf[pix] += If; 
     m_sum_g2[pix] += Ii*If; 
     m_sum_st[pix] ++; 
   }
}

//----------------

void
CorAnaData::saveCorTau(std::ostream& out)
{
   m_log << "  ->  CorAnaData::saveCorTau(tau)";

   cor_t gi, gf;

   m_notzero = 1; // !!!!!!!!!!!!!!!! HADRWIRED

   for(unsigned pix=0; pix<m_blk_size; pix++) {
       if(m_sum_st[pix]<1) continue;
       gi = cor_t( m_sum_gi[pix] / m_sum_st[pix] ); 
       gf = cor_t( m_sum_gf[pix] / m_sum_st[pix] ); 
       m_cor_gi[pix] = (gi<m_notzero) ? m_notzero : gi; 
       m_cor_gf[pix] = (gf<m_notzero) ? m_notzero : gf; 
       m_cor_g2[pix] = cor_t( m_sum_g2[pix] / m_sum_st[pix] ); 
   }

   out.write(reinterpret_cast<const char*>(m_cor_gi), m_blk_size*sizeof(cor_t));
   out.write(reinterpret_cast<const char*>(m_cor_gf), m_blk_size*sizeof(cor_t));
   out.write(reinterpret_cast<const char*>(m_cor_g2), m_blk_size*sizeof(cor_t));
}

//----------------

void
CorAnaData::printCorTau(unsigned tau)
{
  m_log << "\nCorAnaData::printCorTau(tau): tau = " << tau << std::setprecision(3) << std::setw(6) << std::left << "\n";
  unsigned c0 = 8;
  unsigned c = 0;
  unsigned r = 0;

  for(unsigned pix=0; pix<m_blk_size; pix++) {
    if(c==0)             m_log << "\nRow=" << std::setw(4) << r << ": "; 
    if(c<c0)             m_log << " " << std::setw(6) << m_cor_g2[pix];
    if(c==c0)            m_log << " ...";
    if(c>=m_img_cols-c0) m_log << " " << std::setw(6) << m_cor_g2[pix];
    c++; if(c==m_img_cols) {c=0; r++;}
  }
  m_log << "\n" << std::setprecision(8) << std::setw(10);
}

//----------------
//----------------
//----------------

} // namespace ImgAlgos
