#ifndef IMGALGOS_NDARRIMAGEPRODUCER_H
#define IMGALGOS_NDARRIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrImageProducer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"
#include "MsgLogger/MsgLogger.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSCalib/GeometryAccess.h"

//#include "PSCalib/CSPad2x2CalibPars.h"
//#include "CSPadPixCoords/PixCoordsCSPad2x2V2.h"
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
 *  @brief NDArrImageProducer produces the CSPad2x2 image for each event and add it to the event in psana framework.
 *
 *  NDArrImageProducer works in psana framework. It does a few operation as follows:
 *  1) get the pixel coordinates from PixCoords2x1 and PixCoordsCSPad2x2 classes,
 *  2) get data from the event,
 *  3) produce the Image2D object with CSPad image for each event,
 *  4) add the Image2D object in the event for further modules.
 *
 *  The CSPad2x2 image array currently is shaped as [400][400] pixels.
 *
 *  This class should not be used directly in the code of users modules. 
 *  Instead, it should be added as a module in the psana.cfg file with appropriate parameters.
 *  Then, the produced Image2D object can be extracted from event and used in other modules.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see PixCoords2x1, PixCoordsQuad, PixCoordsCSPad, CSPadImageGetTest
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class NDArrImageProducer : public Module {
public:

  // Default constructor
  NDArrImageProducer (const std::string& name) ;

  // Destructor
  virtual ~NDArrImageProducer () ;

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

  void printInputParameters();
  bool getCalibPars(Event& evt, Env& env);
  void cspad_image_init();
  void procEvent(Event& evt, Env& env);
  void checkTypeImplementation();

private:

  // Data members, this is for example purposes only

  std::string m_calibdir;       // i.e. ./calib
  std::string m_calibgroup;     // i.e. CsPad2x2::CalibV1
  std::string m_str_src;        // i.e. MecTargetChamber.0:Cspad2x2.1
  Source      m_source;         // i.e. Detinfo(MecTargetChamber.0:Cspad2x2.1)
  Pds::Src    m_src;
  std::string m_inkey; 
  std::string m_outimgkey;      // i.e. "CSPad:Image"
  std::string m_outtype;
  std::string m_oname;
  unsigned    m_oindex;
  double      m_pix_scale_size_um;
  int         m_x0_off_pix;
  int         m_y0_off_pix;
  int        *m_xy0_off_pix;
  int         m_mode;           // mode of mapping pixels from ndarray to 2-d image
  bool        m_do_tilt;        // on/off tilt angles
  unsigned    m_print_bits;
  unsigned    m_count_evt;
  unsigned    m_count_clb;
  unsigned    m_count_msg;      // number of messages counter
  DATA_TYPE   m_dtype;

  unsigned    m_size;
  const unsigned  *m_coor_x_ind;
  const unsigned  *m_coor_y_ind;
  unsigned    m_x_ind_max;
  unsigned    m_y_ind_max;

  PSCalib::GeometryAccess* m_geometry;

public:

//--------------------

  template <typename TINP, typename TOUT>
    void save2DArrayInEventForType (Event& evt, ndarray<TINP,2>& img_inp ) {

      //If types are equal - just save array in event 
      if (typeid(TOUT) == typeid(TINP)) { // typeid(double).name()
        save2DArrayInEvent<TINP>(evt, m_src, m_outimgkey, img_inp);
        return;
      }
      
      //Copy array with type changing
      ndarray<TOUT,2> img_out (img_inp.shape());

      typename ndarray<TOUT,2>::iterator it_out = img_out.begin(); 
      for (typename ndarray<TINP,2>::iterator it=img_inp.begin(); it!=img_inp.end(); ++it, ++it_out ) {
          *it_out = (TOUT)*it;
      }

      save2DArrayInEvent<TOUT>(evt, m_src, m_outimgkey, img_out);
  }

//--------------------

  template <typename T, unsigned ND>
  void image_fill_and_add_in_event(Event& evt, const ndarray<const T,ND>& data)
  {
    if (data.size() != m_size) { 
      stringstream ss; ss << "ndarray for source:" << m_source 
                          << " and key:" << m_inkey
                          << " has size: " << data.size()
                          << " different from number of pixels in geometry:" << m_size; 
      MsgLog(name(), warning, ss.str());
      throw std::runtime_error(ss.str());
    }

    unsigned shape[2] = {m_x_ind_max+1, m_y_ind_max+1};
    ndarray<T,2> img_nda(shape);

    std::fill_n(img_nda.data(), int(img_nda.size()), T(0));
  
    const T* p_data = data.data();


    if (m_mode == 0) {
      // Pixel intensity is replaced by the latest mapped pixel
      for (unsigned i=0; i<m_size; ++i)
        img_nda[m_coor_x_ind[i]][m_coor_y_ind[i]] = p_data[i];
    }

    else if (m_mode == 1) {
      // Select maximal intensity of two overlapping pixels
      for (unsigned i=0; i<m_size; ++i) {  
	 T* p_tmp = &img_nda[m_coor_x_ind[i]][m_coor_y_ind[i]];
	 if ( *p_tmp==0 || p_data[i] > *p_tmp) *p_tmp = p_data[i];
      }
    }

    else if (m_mode == 2) {
      // Accumulate pixel intensity in the 2-d image
      for (unsigned i=0; i<m_size; ++i) {  
        //unsigned ix = m_coor_x_ind[i];
        //unsigned iy = m_coor_y_ind[i];  
        img_nda[m_coor_x_ind[i]][m_coor_y_ind[i]] += p_data[i]; 
      }
    }

    //else if (m_mode == 3) 
      // Interpolation TBA

    else {
      // The same as mode 0
      for (unsigned i=0; i<m_size; ++i)
        img_nda[m_coor_x_ind[i]][m_coor_y_ind[i]] = p_data[i];
    }

    if      ( m_dtype == ASINP  ) save2DArrayInEvent<T>(evt, m_src, m_outimgkey, img_nda);
    else if ( m_dtype == INT16  ) save2DArrayInEventForType<T, int16_t> (evt, img_nda); 
    else if ( m_dtype == FLOAT  ) save2DArrayInEventForType<T, float>   (evt, img_nda); 
    else if ( m_dtype == DOUBLE ) save2DArrayInEventForType<T, double>  (evt, img_nda); 
    else if ( m_dtype == INT    ) save2DArrayInEventForType<T, int>     (evt, img_nda); 
  }

//--------------------

  template <typename T, unsigned ND>
  bool procNDArrForTypeAndND(Event& evt, Env& env) {

       // CONST ndarray 
       shared_ptr< ndarray<const T,ND> > shp_const = evt.get(m_source, m_inkey, &m_src); // get m_src here
       if (shp_const.get()) {
         if ( ! getCalibPars(evt, env) ) return false;
         const ndarray<const T,ND>& nda = *shp_const.get();
         image_fill_and_add_in_event <T,ND> (evt, nda);
         return true;
       }

       // NON-CONST ndarray 
       shared_ptr< ndarray<T,ND> > shp = evt.get(m_source, m_inkey, &m_src); // get m_src here
       if (shp.get()) {
         if ( ! getCalibPars(evt, env) ) return false;
         ndarray<T,ND>& nda = *shp.get();
         image_fill_and_add_in_event <T,ND> (evt, nda);
         return true;
       }

       return false;
  }
  
//--------------------

  template <typename T>
  bool procNDArrForType (Event& evt, Env& env) {
 
    if( m_print_bits & 8 ) MsgLog(name(), warning, "Produce image from ndarray, source:" << m_source 
                                  << " key:" << m_inkey << " data type:" << typeid(T).name() );
    
    if ( procNDArrForTypeAndND<T,2>(evt, env) ) return true;
    if ( procNDArrForTypeAndND<T,3>(evt, env) ) return true;
    if ( procNDArrForTypeAndND<T,4>(evt, env) ) return true;
    if ( procNDArrForTypeAndND<T,1>(evt, env) ) return true;
    
    return false;
  }
  
//--------------------


}; // class NDArrImageProducer

} // namespace ImgAlgos

#endif // IMGALGOS_NDARRIMAGEPRODUCER_H
