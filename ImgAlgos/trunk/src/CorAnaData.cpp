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
//#include <sstream>
#include <iomanip>
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
CorAnaData::CorAnaData(): m_log(INPARS->get_ostream())
{
  m_log << "C-tor: CorAnaData()\n";

  m_timer1 = new TimeInterval();
  m_log << "Job is started at " << m_timer1->strStartTime() << "\n";

  readMetadataFile();
  readDataFile();

  m_log << "Data reading time =" << m_timer1->getCurrentTimeInterval() << "sec\n";

  printMetadata();
  printData();

  makeIndTau();
  printIndTau();
  saveIndTauInFile();

  m_timer1->startTime();

  initCorTau();
  unsigned tau=10;
  evaluateCorTau(tau);

  m_log << "\nCorrelation processing time =" << m_timer1->getCurrentTimeInterval() << "sec\n";
  printCorTau(tau);
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
  //std::string& fname = v_names[0];

  m_fname     =  v_names[0]; // std::string(fname);
  m_fname_com = m_fname.substr(0,m_fname.rfind("-b"));
  m_fname_med = m_fname_com + ".med";

  m_log << "Data file name:                " << m_fname     << "\n";
  m_log << "Commmon part of the file name: " << m_fname_com << "\n";
  m_log << "Metadata file name:            " << m_fname_med << "\n";

  //std::string line;
  std::string key;

  std::ifstream inf(m_fname_med.c_str());
  if (inf.is_open())
  {
    while ( inf.good() )
    {
      //getline (inf,line);
      //std::cout << line << "\n";
      inf >> key;
             if (key=="IMAGE_ROWS")      inf >> m_img_rows;
	else if (key=="IMAGE_COLS")      inf >> m_img_cols;
	else if (key=="IMAGE_SIZE")      inf >> m_img_size;
	else if (key=="NUMBER_OF_FILES") inf >> m_nfiles;
	else if (key=="BLOCK_SIZE")      inf >> m_blk_size;
	else if (key=="REST_SIZE")       inf >> m_rst_size;
	else if (key=="NUMBER_OF_IMGS")  inf >> m_nimgs;
        else if (key=="FILE_TYPE")       inf >> m_file_type;
        else if (key=="DATA_TYPE")       inf >> m_data_type;
        else std::cout << "\nWARNING! The key: " << key
                       << " is not recognized in the metadata file: " << m_fname_med;
    }
    inf.close();
  }
  else m_log << "Unable to open file :" << m_fname_med;  
}

//----------------

void
CorAnaData::printMetadata()
{
    m_log   << "\nMetadata from input file: " << m_fname_med 
            << "\nIMAGE_ROWS      " << m_img_rows 
            << "\nIMAGE_COLS      " << m_img_cols
            << "\nIMAGE_SIZE      " << m_img_size
            << "\nNUMBER_OF_FILES " << m_nfiles
            << "\nBLOCK_SIZE      " << m_blk_size
            << "\nREST_SIZE       " << m_rst_size
            << "\nNUMBER_OF_IMGS  " << m_nimgs
            << "\nFILE_TYPE       " << m_file_type
            << "\nDATA_TYPE       " << m_data_type
            << "\n";
}

//----------------

void
CorAnaData::readDataFile()
{
  m_log << "Read data from file: " << m_fname << "\n";

  std::fstream inf(m_fname.c_str(), std::ios::in | std::ios::binary);
  if (!inf.is_open()) {
     const std::string msg = "Unable to open file :" + m_fname_med; 
     m_log << msg;  
     abort();
  }

  m_data = new data_t [m_blk_size * m_nimgs];
  //inf.seekg(0);
  inf.read((char*)m_data, sizeof(unsigned) * m_blk_size * m_nimgs); 

  inf.close();
}

//----------------

void
CorAnaData::printData()
{
  m_log << "Data red from file: " << m_fname;

  for(unsigned r=0; r<m_nimgs; r++) {
    if ( r<10 
      || r<100 && r%10 == 0 
      || r%100== 0
      || r==m_nimgs-1 )
      {
        m_log << "\nImg-blk " << std::setw(4) << r << ":";
        for(unsigned c=0; c<10; c++)                     m_log << " "  << std::setw(4) << m_data[r*m_blk_size + c];
	m_log << " ...";
        for(unsigned c=m_blk_size-10; c<m_blk_size; c++) m_log << " "  << std::setw(4) << m_data[r*m_blk_size + c];
      }
  }
  m_log << "\n";
}

//----------------

void
CorAnaData::initCorTau()
{
  m_log << "\ninitCorTau()" << m_fname;
  m_res_g2 = new double   [m_blk_size];
  m_sum_g2 = new double   [m_blk_size];
  m_sum_gi = new double   [m_blk_size];
  m_sum_gf = new double   [m_blk_size];
  m_sum_st = new unsigned [m_blk_size];
}

//----------------

void
CorAnaData::evaluateCorTau(unsigned tau) // tau in number of frames between images
{
  m_log << "\nevaluateCorTau: tau=" << tau;
  std::fill_n(m_res_g2, m_blk_size, double(0));
  std::fill_n(m_sum_g2, m_blk_size, double(0));
  std::fill_n(m_sum_gi, m_blk_size, double(0));
  std::fill_n(m_sum_gf, m_blk_size, double(0));
  std::fill_n(m_sum_st, m_blk_size, unsigned(0));

  for (unsigned i=0; i<m_nimgs-tau; i++) {
       unsigned f=i+tau;

       sumCorTau(i,f);

  }
       saveCorTau(tau);
}

//----------------

void
CorAnaData::sumCorTau(unsigned i, unsigned f)
{
    if ( i<10 
      || i<100 && i%10 == 0 
      || i%100== 0
      || i==m_nimgs-(f-i)-1 ) 
      m_log << "\nevaluateCorTau: tau=" << f-i 
            << "  i, f=" << i << ", " << f;

   data_t* p_i = &m_data[i*m_blk_size];
   data_t* p_f = &m_data[f*m_blk_size];

   for(unsigned pix=0; pix<m_blk_size; pix++) {
     m_sum_g2[pix] += p_i[pix]*p_f[pix]; 
     m_sum_gi[pix] += p_i[pix]; 
     m_sum_gf[pix] += p_f[pix]; 
     m_sum_st[pix] += 1; 
   }
}

//----------------

void
CorAnaData::saveCorTau(unsigned tau)
{
   m_log << "\nsaveCorTau: tau=" << tau;
   double den(0);
   for(unsigned pix=0; pix<m_blk_size; pix++) {
     den = m_sum_gi[pix] * m_sum_gf[pix];
     m_res_g2[pix] = (den != 0) ? m_sum_g2[pix] * m_sum_st[pix] / den : 0; 
   }
}

//----------------

void
CorAnaData::printCorTau(unsigned tau)
{
  m_log << "\nprintCorTau: tau = " << tau << std::setprecision(3) << std::setw(6) << std::left << "\n";
  unsigned c0 = 8;
  unsigned c = 0;
  unsigned r = 0;

  for(unsigned pix=0; pix<m_blk_size; pix++) {
    if(c==0)             m_log << "\nRow=" << std::setw(4) << r << ": "; 
    if(c<c0)             m_log << " " << std::setw(6) << m_res_g2[pix];
    if(c==c0)            m_log << " ...";
    if(c>=m_img_cols-c0) m_log << " " << std::setw(6) << m_res_g2[pix];
    c++; if(c==m_img_cols) {c=0; r++;}
  }
  m_log << "\n" << std::setprecision(8) << std::setw(10);
}

//----------------

void
CorAnaData::makeIndTau()
{
  //m_log << "\nmakeIndTau():\n";
  for(unsigned itau=1; itau<m_nimgs; itau++) {
     if( itau<100 
      || itau<1000 && itau%10 == 0 
      || itau%100 == 0 ) v_ind_tau.push_back(itau);
  }
}

//----------------

void
CorAnaData::printIndTau()
{
  m_log << "\nVector of indexes for tau: size =" << v_ind_tau.size()  << "\n";
  unsigned counter=0;
  for(vector<unsigned>::const_iterator it = v_ind_tau.begin(); 
                                       it!= v_ind_tau.end(); it++) {
    m_log << " "  << std::setw(5) << std::right << *it;
    counter++; if(counter>19) {counter=0; m_log << "\n";}
  }
  m_log << "\n";
}

//----------------

void
CorAnaData::saveIndTauInFile()
{
  m_fname_tau = m_fname_com + "-tau.txt";
  m_log << "\nsaveIndTauInFile(): " << m_fname_tau  << "\n";

  std::ofstream out(m_fname_tau.c_str());
  for(vector<unsigned>::const_iterator it = v_ind_tau.begin(); 
                                       it!= v_ind_tau.end(); it++)
  out << " " << *it;
  out << "\n";
  out.close();
}

//----------------
//----------------
//----------------
//----------------

} // namespace ImgAlgos
