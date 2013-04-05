//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgMaskEvaluation...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgMaskEvaluation.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
//#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(ImgMaskEvaluation)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ImgMaskEvaluation::ImgMaskEvaluation (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_file_mask_satu()
  , m_file_mask_nois()
  , m_file_mask_comb()
  , m_file_frac_satu()
  , m_file_frac_nois()
  , m_thre_satu()
  , m_thre_nois()
  , m_frac_satu()
  , m_frac_nois()
  , m_dr()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src        = configSrc("source",  "DetInfo(:Cspad)");
  m_key            = configStr("key",     "");
  m_file_mask_satu = configStr("file_mask_satu", "img-mask-satu.dat");
  m_file_mask_nois = configStr("file_mask_nois", "img-mask-nois.dat");
  m_file_mask_comb = configStr("file_mask_comb", "img-mask-comb.dat");
  m_file_frac_satu = configStr("file_frac_satu", "img-frac-satu.dat");
  m_file_frac_nois = configStr("file_frac_nois", "img-frac-nois.dat");
  m_thre_satu      =    config("thre_satu", 1000000.); 
  m_thre_nois      =    config("thre_SoN",        5.);
  m_frac_satu      =    config("frac_satu",       0);
  m_frac_nois      =    config("frac_nois",     0.3);
  m_dr             =    config("dr_SoN_ave",      1);
  m_print_bits     =    config("print_bits",      0);

  m_do_mask_satu   = (m_file_mask_satu.empty()) ? false : true;
  m_do_mask_nois   = (m_file_mask_nois.empty()) ? false : true;
  m_do_mask_comb   = (m_file_mask_comb.empty()) ? false : true;
  m_do_frac_satu   = (m_file_frac_satu.empty()) ? false : true;
  m_do_frac_nois   = (m_file_frac_nois.empty()) ? false : true;
}

//--------------------

// Print input parameters
void 
ImgMaskEvaluation::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source          : " << m_str_src
        << "\n key             : " << m_key      
        << "\n file_mask_satu  : " << m_file_mask_satu
        << "\n file_mask_nois  : " << m_file_mask_nois
        << "\n file_mask_comb  : " << m_file_mask_comb
        << "\n file_frac_satu  : " << m_file_frac_satu
        << "\n file_frac_nois  : " << m_file_frac_nois
        << "\n thre_satu       : " << m_thre_satu     
        << "\n thre_nois       : " << m_thre_nois     
        << "\n frac_satu       : " << m_frac_satu     
        << "\n frac_nois       : " << m_frac_nois     
        << "\n dr              : " << m_dr
        << "\n print_bits      : " << m_print_bits
        << "\n do_mask_satu    : " << m_do_mask_satu
        << "\n do_mask_nois    : " << m_do_mask_nois
        << "\n do_mask_comb    : " << m_do_mask_comb
        << "\n do_frac_satu    : " << m_do_frac_satu
        << "\n do_frac_nois    : " << m_do_frac_nois
        << "\n";     
    log << "\n Image shape parameters:"
        << "\n Columns : "    << m_cols  
        << "\n Rows    : "    << m_rows     
        << "\n Size    : "    << m_size  
        << "\n";
  }
}

//--------------
// Destructor --
//--------------
ImgMaskEvaluation::~ImgMaskEvaluation ()
{
}

/// Method which is called once at the beginning of the job
void 
ImgMaskEvaluation::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
ImgMaskEvaluation::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgMaskEvaluation::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgMaskEvaluation::event(Event& evt, Env& env)
{
  ++ m_count;
  if( m_print_bits & 2 ) printEventRecord(evt);
  if( m_count == 1 )     initImgArrays(evt);
  collectStat(evt);
}
  
/// Method which is called at the end of the calibration cycle
void 
ImgMaskEvaluation::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgMaskEvaluation::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ImgMaskEvaluation::endJob(Event& evt, Env& env)
{
  procStatArrays();

  if(m_do_mask_satu) save2DArrayInFile<int16_t> ( m_file_mask_satu, p_mask_satu, m_rows, m_cols, m_print_bits & 16 );
  if(m_do_mask_nois) save2DArrayInFile<int16_t> ( m_file_mask_nois, p_mask_nois, m_rows, m_cols, m_print_bits & 16 );
  if(m_do_mask_comb) save2DArrayInFile<int16_t> ( m_file_mask_comb, p_mask_comb, m_rows, m_cols, m_print_bits & 16 );
  if(m_do_frac_satu) save2DArrayInFile<double>  ( m_file_frac_satu, p_frac_satu, m_rows, m_cols, m_print_bits & 16 );
  if(m_do_frac_nois) save2DArrayInFile<double>  ( m_file_frac_nois, p_frac_nois, m_rows, m_cols, m_print_bits & 16 );
}

//--------------------

/// Check the event counter and deside what to do next accumulate/change mode/etc.
void 
ImgMaskEvaluation::initImgArrays(Event& evt)
{
    defineImageShape(evt, m_str_src, m_key, m_shape); // shape is not available in beginJob and beginRun
    m_rows = m_shape[0];
    m_cols = m_shape[1];
    m_rows1= m_rows - 1;
    m_cols1= m_cols - 1;
    m_size = m_rows*m_cols;

    p_stat_satu = new unsigned[m_size];
    p_stat_nois = new unsigned[m_size];
    p_mask_satu = new int16_t [m_size]; 
    p_mask_nois = new int16_t [m_size];
    p_mask_comb = new int16_t [m_size];
    p_frac_satu = new double  [m_size];
    p_frac_nois = new double  [m_size]; 

    resetStatArrays();

    evaluateVectorOfIndexesForMedian();

    if( m_print_bits & 1 ) printInputParameters();
    if( m_print_bits & 4 ) printVectorOfIndexesForMedian();
}

//--------------------

/// Reset arrays for statistics accumulation
void
ImgMaskEvaluation::resetStatArrays()
{
  std::fill_n(p_stat_satu,  int(m_size), unsigned(0));
  std::fill_n(p_stat_nois,  int(m_size), unsigned(0));
  std::fill_n(p_mask_satu,  int(m_size), int16_t (1));
  std::fill_n(p_mask_nois,  int(m_size), int16_t (1));
  std::fill_n(p_mask_comb,  int(m_size), int16_t (1));
  std::fill_n(p_frac_satu,  int(m_size), double (0.));
  std::fill_n(p_frac_nois,  int(m_size), double (0.));
}

//--------------------

/// Collect statistics
void 
ImgMaskEvaluation::collectStat(Event& evt)
{
  if ( collectStatForType<uint16_t> (evt) ) return;
  if ( collectStatForType<int>      (evt) ) return;
  if ( collectStatForType<float>    (evt) ) return;
  if ( collectStatForType<uint8_t>  (evt) ) return;
  if ( collectStatForType<double>   (evt) ) return;

  MsgLog(name(), info, "Image is not available in the event(...) for source:" << m_str_src << " key:" << m_key);
}

//--------------------

/// Process accumulated stat arrays and evaluate masks
void 
ImgMaskEvaluation::procStatArrays()
{
    if( m_print_bits & 8 ) MsgLog(name(), info, "Process statistics for collected total " << m_count << " events");
  
    if(m_do_mask_satu || m_do_frac_satu) 
        for (unsigned i=0; i<m_size; ++i) {
          p_frac_satu[i] = double(p_stat_satu[i]) / m_count;
          if( p_frac_satu[i] > m_frac_satu ) p_mask_satu[i] = 0;
	}

    if(m_do_mask_nois || m_do_frac_nois) 
        for (unsigned i=0; i<m_size; ++i) {
          p_frac_nois[i] = double(p_stat_nois[i]) / m_count;
          if( p_frac_nois[i] > m_frac_nois ) p_mask_nois[i] = 0;
	}

    if(m_do_mask_comb && m_do_mask_nois && m_do_mask_satu) 
        for (unsigned i=0; i<m_size; ++i) {
          p_mask_comb[i] = p_mask_satu[i] & p_mask_nois[i];
	}
}

//--------------------

void 
ImgMaskEvaluation::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
  );
}

//--------------------
/// Evaluate vector of indexes for mediane algorithm
/// The area of pixels for the mediane algorithm is defined as a rectangular from -m_dr to +m_dr in 2-d
void 
ImgMaskEvaluation::evaluateVectorOfIndexesForMedian()
{
  v_indForMediane.clear();

  TwoIndexes inds;
  int indmax = int(m_dr);
  int indmin = -indmax;

  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {

      //float r = std::sqrt( float(i*i + j*j) );
      //if ( r < m_rmin || r > m_rmin + m_dr ) continue;
      if ( i==0 && j==0 ) continue;  // exclude self from averaging
      //if ( i==0 || j==0 ) continue;  // exclude central row and column from averaging
      inds.i = i;
      inds.j = j;
      v_indForMediane.push_back(inds);
    }
  }
}

//--------------------
/// Print vector of indexes for mediane algorithm
void 
ImgMaskEvaluation::printVectorOfIndexesForMedian()
{
  std::cout << "ImgMaskEvaluation::printVectorOfIndexesForMedian():" << std::endl;
  int n_pairs_in_line=0;
  for( vector<TwoIndexes>::const_iterator ij  = v_indForMediane.begin();
                                          ij != v_indForMediane.end(); ij++ ) {

    cout << " (" << ij->i << "," << ij->j << ")";
    if ( ++n_pairs_in_line > 9 ) {cout << "\n"; n_pairs_in_line=0;}
  }   
  cout << "\nVector size: " << v_indForMediane.size() << endl;
}

//--------------------

} // namespace ImgAlgos
