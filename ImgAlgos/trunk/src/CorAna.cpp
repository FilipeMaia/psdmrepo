//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAna...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CorAna.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream> // for stringstream
#include <iomanip> // for setw, setprecision

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
CorAna::CorAna() : m_log(INPARS->get_ostream())
{
  defineFileNames();
  readMetadataFile();
  readTimeRecordsFile();
  defineIndTau();
  if(m_file_num_str == "b0000") saveIndTauInFile(); // for single job only!

  printFileNames();
  printMetadata();
  printTimeRecords();
  printIndTau();
}

//--------------
// Destructor --
//--------------
CorAna::~CorAna ()
{
}

//----------------

void
CorAna::defineFileNames()
{
  //std::vector<std::string>&  v_names = INPARS -> get_vector_fnames();    
  //m_fname     =  v_names[0];
  m_fname            = INPARS -> get_fname_data(); 
  int posb           = m_fname.rfind("-b");
  m_fname_com        = m_fname.substr(0,posb);
  m_fname_med        = m_fname_com + "-med.txt";
  m_fname_time       = m_fname_com + "-time.txt"; 
  m_fname_tau        = INPARS -> get_fname_tau(); 
  m_fname_tau_out    = m_fname_com + "-tau.txt";
  m_fname_result     = m_fname.substr(0,m_fname.rfind(".")) + "-result.bin";
  m_fname_result_img = m_fname_com + "-image-result.bin";
  m_file_num_str     = m_fname.substr(posb+1,5); // cut something like: "b0000"
}

//----------------

void
CorAna::printFileNames()
{
  m_log << "CorAna::printFileNames(): I/O file names:\n";
  m_log << "Data file name                  : " << m_fname             << "\n";
  m_log << "Commmon part of the file name   : " << m_fname_com         << "\n";
  m_log << "Metadata file name              : " << m_fname_med         << "\n";
  m_log << "Image time records file name    : " << m_fname_time        << "\n";
  m_log << "Indexes of tau input file name  : " << m_fname_tau         << "\n";
  m_log << "Indexes of tau output file name : " << m_fname_tau_out     << "\n";
  m_log << "Resulting output file for block : " << m_fname_result      << "\n";
  m_log << "Resulting output file name      : " << m_fname_result      << "\n";
  m_log << "Resulting output file for image : " << m_fname_result_img  << "\n";
  m_log << "Sting with file number          : " << m_file_num_str      << "\n";
  m_log << "sizeof(cor_t)                   : " << sizeof(cor_t)       << "\n";
}

//----------------

void
CorAna::readMetadataFile()
{
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
  else m_log << "CorAna::readMetadataFile(): Unable to open file: " << m_fname_med << "\n";  
}

//----------------

void
CorAna::printMetadata()
{
    m_log   << "\nCorAna::printMetadata(): Metadata from input file: " << m_fname_med 
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
CorAna::readTimeRecordsFile()
{
  m_log << "\nCorAna::readTimeRecordsFile(): Read time records from file: " << m_fname_time << "\n";

  std::string s;
  TimeRecord tr;

  std::ifstream inf(m_fname_time.c_str());
  if (inf.is_open())
  {
    while ( true )
    {
      getline (inf,s);
      if(!inf.good()) break;
      std::stringstream ss(s); 
      ss >> tr.ind >> tr.sec >> tr.dt_sec >> tr.tstamp;
      v_time_records.push_back(tr);
    }
    inf.close();
  }
  else m_log << "CorAna::readTimeRecordsFile(): Unable to open file: " << m_fname_time << "\n";  
}

//----------------

void
CorAna::printTimeRecords()
{
  m_log << "\nCorAna::printTimeRecords(): Time records from file: " << m_fname_time
        << " size=" << v_time_records.size() << "\n";
  unsigned counter=0;
  for(std::vector<TimeRecord>::const_iterator it = v_time_records.begin(); 
                                              it!= v_time_records.end(); it++) {
    counter++;
    if ( counter<10 
      || (counter<100  && counter%10  == 0) 
      || (counter<1000 && counter%100 == 0)
      || counter%1000 == 0 
      || counter==v_time_records.size() ) 
      m_log << " ind:"    << std::setw(4)                                        << it->ind
            << " sec:"    << std::fixed << std::setw(15) << std::setprecision(3) << it->sec
            << " dt_sec:" << std::setw(6)                                        << it->dt_sec
            << " tstamp:"                                                        << it->tstamp
            << "\n";      
  }
  m_log << "\n";
}

//----------------

void
CorAna::defineIndTau()
{
  if( readIndTauFromFile() ) makeIndTau();
}

//----------------

int
CorAna::readIndTauFromFile()
{
  if (m_fname_tau.empty()) {
    m_log << "\nCorAna::readIndTauFromFile(): The file name with a list of tau indexes is empty.\n"; 
    return 1;
  }

  std::ifstream inf(m_fname_tau.c_str());
  if (inf.is_open())
  {
    m_log << "\nCorAna::readIndTauFromFile(): " << m_fname_tau  << "\n";

    unsigned val;
    while ( inf.good() )
    {
      inf >> val;
      v_ind_tau.push_back(val);
    }
    inf.close();
    m_npoints_tau = v_ind_tau.size();
    return 0;
  }
  else 
  {
    m_log << "\nCorAna::readIndTauFromFile(): Unable to open file with a list of tau indexes: " << m_fname_tau << "\n"; 
    return 2;
  }
}

//----------------

void
CorAna::makeIndTau()
{
  m_log << "\nCorAna::makeIndTau(): Make the list of tau indexes using standard algorithm.\n";
  for(unsigned itau=1; itau<m_nimgs; itau++) {
     if( itau<8 
      || (itau<16     && itau%2    == 0) 
      || (itau<32     && itau%4    == 0) 
      || (itau<64     && itau%8    == 0) 
      || (itau<128    && itau%16   == 0) 
      || (itau<256    && itau%32   == 0) 
      || (itau<512    && itau%64   == 0) 
      || (itau<1024   && itau%128  == 0) 
      ||                itau%256  == 0 ) v_ind_tau.push_back(itau);
  }
  m_npoints_tau = v_ind_tau.size();
}

//----------------

void
CorAna::printIndTau()
{
  m_log << "\nCorAna::printIndTau(): Vector of indexes for tau: size =" << m_npoints_tau << "\n";
  unsigned counter=0;
  for(std::vector<unsigned>::const_iterator it = v_ind_tau.begin(); 
                                            it!= v_ind_tau.end(); it++) {
    m_log << " "  << std::setw(5) << std::right << *it;
    counter++; if(counter>19) {counter=0; m_log << "\n";}
  }
  m_log << "\n";
}

//----------------

void
CorAna::saveIndTauInFile()
{
  m_log << "\nCorAna::saveIndTauInFile(): " << m_fname_tau_out  << "\n";

  std::ofstream out(m_fname_tau_out.c_str());
  for(std::vector<unsigned>::const_iterator it = v_ind_tau.begin(); 
                                            it!= v_ind_tau.end(); it++)
  out << " " << *it;
  out << "\n";
  out.close();
}

//----------------
//----------------

} // namespace ImgAlgos
