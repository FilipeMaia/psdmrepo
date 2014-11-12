#ifndef PSANA_EXAMPLES_PNCCDIMAGEPRODUCER_H
#define PSANA_EXAMPLES_PNCCDIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdImageProducer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/pnccd.ddl.h"
#include "ImgAlgos/GlobalMethods.h"
#include "ImgAlgos/PnccdNDArrProducer.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/**
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class PnccdImageProducer : public Module {
public:

  /// Data type for detector image 
  typedef uint16_t data_t;

  const static size_t   Segs   = ImgAlgos::PnccdNDArrProducer::Segs; 
  const static size_t   Rows   = ImgAlgos::PnccdNDArrProducer::Rows; 
  const static size_t   Cols   = ImgAlgos::PnccdNDArrProducer::Cols; 
  const static size_t   FrSize = ImgAlgos::PnccdNDArrProducer::FrSize; 
  const static size_t   Size   = ImgAlgos::PnccdNDArrProducer::Size; 
 
  const static size_t   ImRows = 1024; 
  const static size_t   ImCols = 1024; 



  // Default constructor
  PnccdImageProducer (const std::string& name) ;

  // Destructor
  virtual ~PnccdImageProducer () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);
  
protected:

  void printInputParameters();
 
private:

  Pds::Src    m_src;
  Source      m_str_src;
  std::string m_key_in; 
  std::string m_key_out;
  size_t      m_gap_rows;
  size_t      m_gap_cols;
  uint16_t    m_gap_value;
  unsigned    m_print_bits;

//-------------------
//-------------------
//-------------------

  //for T like Psana::PNCCD::FullFrameV1
  template <typename T>
  bool procEventForFullFrame(Event& evt)
  { 
      shared_ptr<const T> frame = evt.get(m_str_src, m_key_in, &m_src);
      if (frame) {
      
	  const ndarray<const data_t, 2> data = frame->data(); // .copy(); - if no const
          if( m_print_bits & 2 ) {for (int i=0; i<10; ++i) cout << " " << data[0][i]; std::cout << "\n";}
     
          save2DArrayInEvent<data_t> (evt, m_src, m_key_out, data);
          return true;
      }
      return false;
  }

//-------------------
//-------------------
//-------------------

  template <typename T>
  bool procEventFor3DArrType(Event& evt)
  { 
      shared_ptr< ndarray<const T,3> > shp = evt.get(m_str_src, m_key_in, &m_src);
      if (shp.get()) {

          const ndarray<const T,3> inp_ndarr = *shp.get(); //const T* p_data = shp->data();

	  if( m_print_bits & 2 ) std::cout << "Input ndarray<const T,3>:\n" << inp_ndarr << "\n";
	  
          ndarray<T,2> img_ndarr = make_ndarray<T>(ImRows+m_gap_rows, ImCols+m_gap_cols);

          //std::fill_n(&img_ndarr[Rows][0], int(ImCols*m_gap_rows), T(m_gap_value));
          std::fill_n(&img_ndarr[0][0], int(img_ndarr.size()), T(m_gap_value));

	  // Hardwired configuration:
	  size_t rows0    [Segs] = {0,     Rows+m_gap_rows, Rows+m_gap_rows, 0};
	  size_t cols0    [Segs] = {0,     0,               Cols+m_gap_cols, Cols+m_gap_cols};
	  bool  is_rotated[Segs] = {false, true,            true,            false}; // rotated by 180 degree

	  for (size_t s=0; s<Segs; ++s) {

	    const T* it_inp = &inp_ndarr[s][Rows-1][Cols-1]; // pointer to the end of the segment data.

	    for (size_t r=0; r<Rows; ++r) {

	      if ( is_rotated[s] ) {
		for ( T* it=&img_ndarr[rows0[s]+r][cols0[s]]; it!=&img_ndarr[rows0[s]+r][cols0[s]+Cols]; ++it, --it_inp) {*it = *it_inp;}
	      } else {
                  std::memcpy(&img_ndarr[rows0[s]+r][cols0[s]] ,&inp_ndarr[s][r][0], Cols*sizeof(T)); // copy inp_ndarr -> img_ndarr
	          //for (size_t c=0; c<Cols; ++c) img_ndarr[rows0[s]+r][cols0[s]+c] = inp_ndarr[s][r][c];
	      }
	    }
	  }

          save2DArrayInEvent<T> (evt, m_src, m_key_out, img_ndarr);

          return true;
      }

      return false;
  }

//-------------------

}; // End of class

} // namespace ImgAlgos

#endif // PSANA_EXAMPLES_PNCCDIMAGEPRODUCER_H
