//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgVsTimeSplitInFiles...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgVsTimeSplitInFiles.h"

//-----------------
// C/C++ Headers --
//-----------------
// #include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "ImgAlgos/GlobalMethods.h"
#include "ImgAlgos/TimeInterval.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>
#include <iomanip> // for setw, setfill
#include <sstream> // for streamstring

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgVsTimeSplitInFiles)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

ImgVsTimeSplitInFiles::ImgVsTimeSplitInFiles (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_fname_prefix()
  , m_file_type()
  , m_add_tstamp()
  , m_nfiles_out()
  , m_ampl_thr()
  , m_ampl_min()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src       = configStr("source", "DetInfo(:Princeton)");
  m_key           = configStr("key",                 "img");
  m_fname_prefix  = configStr("fname_prefix",     "my-exp");
  m_file_type     = configStr("file_type",           "bin");
  m_add_tstamp    = config   ("add_tstamp",           true);
  m_nfiles_out    = config   ("nfiles_out",              8);
  m_ampl_thr      = config   ("ampl_thr",                1);
  m_ampl_min      = config   ("ampl_min",                1);
  m_print_bits    = config   ("print_bits",              0);

  setFileMode();
}

//--------------------
/// Print input parameters
void 
ImgVsTimeSplitInFiles::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : "     << m_str_src
        << "\nkey          : "     << m_key      
        << "\nfname_prefix : "     << m_fname_prefix
        << "\nfile_type    : "     << m_file_type
        << "\nm_file_mode  : "     << m_file_mode
        << "\nadd_tstamp   : "     << m_add_tstamp
        << "\nnfiles_out   : "     << m_nfiles_out
        << "\nm_ampl_thr   : "     << m_ampl_thr
        << "\nm_ampl_min   : "     << m_ampl_min
        << "\nm_print_bits : "     << m_print_bits;
  }
}

//--------------------

void 
ImgVsTimeSplitInFiles::setFileMode()
{
  m_file_mode = TEXT;
  if (m_file_type == "bin") m_file_mode = BINARY;
  if (m_file_type == "txt") m_file_mode = TEXT;
}

///--------------------


//--------------
// Destructor --
//--------------

ImgVsTimeSplitInFiles::~ImgVsTimeSplitInFiles ()
{
}

//--------------------

//--------------------

/// Method which is called once at the beginning of the job
void 
ImgVsTimeSplitInFiles::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
ImgVsTimeSplitInFiles::beginRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
ImgVsTimeSplitInFiles::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgVsTimeSplitInFiles::event(Event& evt, Env& env)
{
  m_count++;
  if(m_count==1){
    initSplitInFiles(evt, env);
    openOutputFiles(evt);
  }

  procEvent(evt);
  if( m_print_bits & 2 ) printEventRecord(evt, " is split and saved");
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
ImgVsTimeSplitInFiles::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
ImgVsTimeSplitInFiles::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
ImgVsTimeSplitInFiles::endJob(Event& evt, Env& env)
{
  closeOutputFiles();
  saveMetadataInFile();
  if( m_print_bits & 4 ) printSummary(evt, " images are splitted");
}

//--------------------
//--------------------
//--------------------

void 
ImgVsTimeSplitInFiles::initSplitInFiles(Event& evt, Env& env)
{
  shared_ptr< ndarray<uint16_t,2> > img = evt.get(m_str_src, m_key, &m_src);

  if (img.get()) {

    m_img_rows = img->shape()[0];
    m_img_cols = img->shape()[1];
    m_img_size = img->size();
    m_blk_size = m_img_size / m_nfiles_out; 
    m_rst_size = m_img_size % m_nfiles_out;

    m_data = new unsigned [m_img_size];

    MsgLog( name(), info, "Get image parameters:"
                       << "\n Rows             : " << m_img_rows 
                       << "\n Cols             : " << m_img_cols
                       << "\n Total image size : " << m_img_size
                       << "\n m_nfiles_out     : " << m_nfiles_out
                       << "\n m_blk_size       : " << m_blk_size
                       << "\n m_rst_size       : " << m_rst_size
                       << "\n"  
    );  

    if(m_rst_size) {
      std::string msg = "\n  Can not split the image for integer number of files without rest...";
                  msg+= "\n  Try to change the number of output files to split the image for equal parts.\n";
      MsgLogRoot(error, msg);
      throw std::runtime_error(msg);
    }
  } 
  else
  {
    const std::string msg = "Image shape is not defined in the event(...) for source:" + m_str_src + " key:" + m_key;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }
}

//--------------------

void 
ImgVsTimeSplitInFiles::saveMetadataInFile()
{
  std::string fname = m_fname_common+".med";
  std::ofstream out(fname.c_str());

  out <<   "IMAGE_ROWS      " << m_img_rows 
      << "\nIMAGE_COLS      " << m_img_cols
      << "\nIMAGE_SIZE      " << m_img_size
      << "\nNUMBER_OF_FILES " << m_nfiles_out
      << "\nBLOCK_SIZE      " << m_blk_size
      << "\nREST_SIZE       " << m_rst_size
      << "\nNUMBER_OF_IMGS  " << m_count
      << "\nFILE_TYPE       " << m_file_type
      << "\nDATA_TYPE       " << m_data_type
      << "\n";

  out.close();
  if( m_print_bits & 16 ) MsgLog( name(), info, "The file with metadata: " << fname << " is created.");
}

//--------------------

void 
ImgVsTimeSplitInFiles::openOutputFiles(Event& evt)
{
  p_out = new std::ofstream [m_nfiles_out];

  ios_base::openmode mode = ios_base::out | ios_base::binary;
  TimeInterval* time_job_start = new TimeInterval();

  m_fname_common = m_fname_prefix + "-r"  + stringRunNumber(evt);
  if(m_add_tstamp)  m_fname_common += time_job_start -> strStartTime("-%Y%m%d-%H%M%S"); 

  std::string fname;
  for(unsigned b=0; b<m_nfiles_out; b++){

     fname = m_fname_common
           + "-b"  + stringFromUint(b,4)
           + "."   + m_file_type;

     if( m_print_bits & 16 ) MsgLog( name(), info, "Open output file: " << fname );     

     p_out[b].open(fname.c_str(), mode);
     //p_out[b] << "This is a content of the file " << fname;
     //p_out[b].close();
  }

  p_out_time.open((m_fname_common + "-time.txt").c_str(), mode);
}

//--------------------

void 
ImgVsTimeSplitInFiles::closeOutputFiles()
{
  for(unsigned b=0; b<m_nfiles_out; b++) p_out[b].close();
  p_out_time.close();
}

//--------------------

void 
ImgVsTimeSplitInFiles::procEvent(Event& evt)
{
  saveTimeRecord(evt);

  shared_ptr< ndarray<double,2> > img = evt.get(m_str_src, m_key, &m_src);
  if (img.get()) {
    m_data_type = "double"; 
    procSplitAndWriteImgInFiles<double> (img, m_print_bits & 8);
  }

  shared_ptr< ndarray<uint16_t,2> > img_u16 = evt.get(m_str_src, m_key, &m_src);
  if (img_u16.get()) {
    m_data_type = "uint16_t"; 
    procSplitAndWriteImgInFiles<uint16_t> (img_u16, m_print_bits & 8);
  }
}

//--------------------

void 
ImgVsTimeSplitInFiles::saveTimeRecord(Event& evt)
{
  m_tsec = doubleTime(evt);
  if(m_count==1) m_tsec_prev = m_tsec;

  p_out_time << std::setw(6) << m_count 
             << fixed << std::setw(16) << std::setprecision(3) << m_tsec
             << fixed << std::setw(7) << std::setprecision(3) << m_tsec-m_tsec_prev
             << stringTimeStamp(evt,"  %Y%m%d-%H%M%S%f") << "\n";

  m_tsec_prev = m_tsec;  
}

//--------------------

void 
ImgVsTimeSplitInFiles::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
	             << comment.c_str() 
  );
}

//--------------------

void 
ImgVsTimeSplitInFiles::printSummary(Event& evt, std::string comment)
{
  MsgLog( name(), info, "Run=" << stringRunNumber(evt) 
	                << " Number of processed events=" << stringFromUint(m_count)
                        << comment.c_str()
  );
}

//--------------------
//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos
