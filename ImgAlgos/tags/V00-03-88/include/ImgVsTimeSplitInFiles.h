#ifndef IMGALGOS_IMGVSTIMESPLITINFILES_H
#define IMGALGOS_IMGVSTIMESPLITINFILES_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgVsTimeSplitInFiles.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <fstream> // for std::ofstream operator << 
#include <sstream> // for stringstream 
//#include <typeinfo> // for typeid()

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "PSEvt/Source.h"


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief ImgVsTimeSplitInFiles gets image from event, splits it and saves in files. 
 *
 *  ImgVsTimeSplitInFiles psana module class works after any ImageProducer.
 *  * It gets the image as ndarray<T,2>
 *  * splits it for requested number of equal parts
 *  * saves each part in binary or text format in file for all events,
 *  * saves timestamps for selected events in the text file.
 *  * saves metadata in the text file.
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadImageProducer
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class ImgVsTimeSplitInFiles : public Module {
public:

  //typedef unsigned data_split_t;
  //typedef uint16_t data_split_t;
  //typedef double data_split_t;
  typedef float data_split_t;


  enum FILE_MODE {BINARY, TEXT};

  // Default constructor
  ImgVsTimeSplitInFiles (const std::string& name) ;

  // Destructor
  virtual ~ImgVsTimeSplitInFiles () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called at the end of the calibration cycle
  virtual void endCalibCycle(Event& evt, Env& env);

  /// Method which is called at the end of the run
  virtual void endRun(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:

  void setFileMode();
  void initSplitInFiles(Event& evt, Env& env);
  void saveImageInFile(Event& evt);
  void printInputParameters();
  void printEventRecord(Event& evt, std::string comment=std::string());
  void printSummary(Event& evt, std::string comment=std::string());
  void openOutputFiles(Event& evt);
  void closeOutputFiles();
  void saveMetadataInFile();
  void procEvent(Event& evt);
  void saveTimeRecord(Event& evt);
  void evaluateMeanTimeBetweenEvents();
  void saveTimeRecordWithIndexInFile();

private:

  // Data members, this is for example purposes only

  Pds::Src    m_src;
  Source      m_str_src;         // i.e. CxiDs1.0:Cspad.0
  std::string m_key;             // i.e. Image2D
  std::string m_fname_prefix;    // prefix of the file name
  std::string m_file_type;       // file type "txt" or "bin" 
  std::string m_data_type_input; // data type "double", "uint16_t", etc. 
  bool        m_add_tstamp;
  unsigned    m_nfiles_out;
  double      m_ampl_thr;
  double      m_ampl_min;
  unsigned    m_print_bits;
  long        m_count;

  FILE_MODE   m_file_mode;

  unsigned    m_img_rows;
  unsigned    m_img_cols;
  unsigned    m_img_size;
  unsigned    m_blk_size;
  unsigned    m_rst_size;

  data_split_t*   m_data;         // image data in case if processing is necessary

  std::string m_fname_common;
  std::string m_fname_time;
  std::string m_fname_time_ind;

  std::ofstream* p_out;
  std::ofstream  p_out_time;

  double      m_tsec_0;       // time of the 1st event
  double      m_tsec;         // time of current event
  double      m_tsec_prev;    // time of previous event
  double      m_dt;           // delta time between current and previous event
  unsigned    m_nevt;         // event number of current event frim eventID().vector()
  unsigned    m_nevt_prev;    // event number of previous event
  unsigned    m_sumt0;
  double      m_sumt1;
  double      m_sumt2;
  double      m_t_ave;        // average time between consecutive events in eventID().vector()
  double      m_t_rms;        // rms spread of dt 
  unsigned    m_tind_max;     // maximal value of the time index.

protected:

//--------------------
// Initialization of the splitting procedure
    template <typename T>
    bool initSplitInFilesForType(Event& evt, Env& env)
    {
      shared_ptr< ndarray<const T,2> > img = evt.get(m_str_src, m_key, &m_src);

      if (img.get()) {

            m_img_rows = img->shape()[0];
            m_img_cols = img->shape()[1];
            m_img_size = img->size();
            m_blk_size = m_img_size / m_nfiles_out; 
            m_rst_size = m_img_size % m_nfiles_out;
	    
            m_data = new data_split_t [m_img_size];
	    
            MsgLog( name(), info, "Get image parameters:"
                           << "\n Rows             : " << m_img_rows 
                           << "\n Cols             : " << m_img_cols
                           << "\n Total image size : " << m_img_size
                           << "\n m_nfiles_out     : " << m_nfiles_out
                           << "\n m_blk_size       : " << m_blk_size
                           << "\n m_rst_size       : " << m_rst_size
                           << "\n Input data type  : " << typeid(T).name()            << " of size " << sizeof(T)
	                   << "\n Output data type : " << typeid(data_split_t).name() << " of size " << sizeof(data_split_t)
                           << "\n"  
                           );  

            if(m_rst_size) {
              std::string msg = "\n  Can not split the image for integer number of files without rest...";
                          msg+= "\n  Try to change the number of output files to split the image for equal parts.\n";
              MsgLogRoot(error, msg);
              throw std::runtime_error(msg);
            }
	    return true;
      } 
      return false;
    }

//--------------------
// Apply threshold correction to the image. 
// Data is accessible through the smart pointer.
    template <typename T>
    void procImgData(const boost::shared_ptr< ndarray<T,2> >& p_ndarr)
    {
      const T* data = p_ndarr->data();
      procImgData<T>(data);
    }

//--------------------
// Apply threshold correction to the image. 
// Data is accessible through the pointer.
    template <typename T>
    void procImgData(const T* data)
    {
      T            athr = static_cast<T>           (m_ampl_thr);
      data_split_t amin = static_cast<data_split_t>(m_ampl_min);
      for(unsigned i=0; i<m_img_size; i++)
	m_data[i] = (data[i] > athr) ? static_cast<data_split_t>(data[i]) : amin;
    }


//--------------------
// Process event for specified data type
    template <typename T>
    bool procEventForType(Event& evt)
    {
      shared_ptr< ndarray<const T,2> > shp = evt.get(m_str_src, m_key, &m_src);
      if (shp.get()) {
        m_data_type_input = typeid(T).name();
        procSplitAndWriteImgInFiles<T> (shp, m_print_bits & 8);
	return true;
      }
      return false;      
    }

//--------------------
// Splits the image for blocks and saves the blocks in files
    template <typename T>
    void procSplitAndWriteImgInFiles (const boost::shared_ptr< ndarray<const T,2> >& p_ndarr, 
                                      bool print_msg=false) 
    {
      const T* img_data = p_ndarr->data();             // Access to entire image
      if( m_print_bits & 32 ) MsgLog( name(), info, stringOf2DArrayData<T>(*p_ndarr.get(), std::string("data in:")) );

      procImgData<T>(img_data);

      for(unsigned b=0; b<m_nfiles_out; b++){

	const data_split_t* p_block_data = &m_data[b*m_blk_size]; // Access to the block 

	if (m_file_mode == TEXT) {
	  std::stringstream ss; 
	  for(unsigned i=0; i<m_blk_size; i++) ss << *p_block_data++ << " ";
	  std::string s = ss.str(); 
	  p_out[b].write(s.c_str(), s.size());
	  p_out[b] <<  "\n";
	} 

        else if (m_file_mode == BINARY) {
	  p_out[b].write(reinterpret_cast<const char*>(p_block_data), m_blk_size*sizeof(data_split_t));
	  //p_out[b] <<  "\n";
	} 

        else {
          p_out[b] << " UNKNOWN FILE TYPE:" << m_file_type << " AND MODE:" << m_file_mode;
	}

      }
    }

//--------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGVSTIMESPLITINFILES_H
