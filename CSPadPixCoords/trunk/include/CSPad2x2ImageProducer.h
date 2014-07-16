#ifndef CSPADPIXCOORDS_CSPAD2X2IMAGEPRODUCER_H
#define CSPADPIXCOORDS_CSPAD2X2IMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2ImageProducer.
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
#include "PSCalib/CSPad2x2CalibPars.h"
#include "CSPadPixCoords/PixCoordsCSPad2x2V2.h"
#include "CSPadPixCoords/GlobalMethods.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "PSEvt/Source.h"


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief CSPad2x2ImageProducer produces the CSPad2x2 image for each event and add it to the event in psana framework.
 *
 *  CSPad2x2ImageProducer works in psana framework. It does a few operation as follows:
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

class CSPad2x2ImageProducer : public Module {
public:

  typedef CSPadPixCoords::PixCoordsCSPad2x2V2 PC2X2;

  const static int NX_CSPAD2X2=400; 
  const static int NY_CSPAD2X2=400;

  // Default constructor
  CSPad2x2ImageProducer (const std::string& name) ;

  // Destructor
  virtual ~CSPad2x2ImageProducer () ;

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
  void getConfigPars(Env& env);
  void getCalibPars(Event& evt, Env& env);
  void cspad_image_init();
  void processEvent(Event& evt, Env& env);

  //void cspad_image_fill(const int16_t* data, CSPadPixCoords::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar);
  //void cspad_image_fill(const ndarray<const int16_t,3>& data);
  void cspad_image_add_in_event(Event& evt);
  void checkTypeImplementation();

private:

  // Data members, this is for example purposes only

  std::string m_calibDir;       // i.e. ./calib
  std::string m_typeGroupName;  // i.e. CsPad2x2::CalibV1
  //std::string m_str_src;        // i.e. MecTargetChamber.0:Cspad2x2.1
  Source      m_source;         // i.e. Detinfo(MecTargetChamber.0:Cspad2x2.1)
  Pds::Src    m_src;
  std::string m_inkey; 
  std::string m_outimgkey;      // i.e. "CSPad:Image"
  std::string m_outtype;
  bool        m_tiltIsApplied;
  bool        m_useWidePixCenter; 
  unsigned    m_print_bits;
  unsigned    m_count;
  unsigned    m_count_cfg;
  long        m_count_msg;        // number of messages counter
  DATA_TYPE   m_dtype;

  uint32_t m_roiMask;
  uint32_t m_numAsicsStored;

  PSCalib::CSPad2x2CalibPars  *m_cspad2x2_calibpars;
  PC2X2                       *m_pix_coords_cspad2x2;

  uint32_t   m_cspad_ind;
  double    *m_coor_x_pix;
  double    *m_coor_y_pix;
  uint32_t  *m_coor_x_int;
  uint32_t  *m_coor_y_int;

  float      m_common_mode[2];

  double m_arr_cspad2x2_image[NX_CSPAD2X2][NY_CSPAD2X2];


public:

  /**
   * @brief Get configuration info from Env, return true if configuration is found, othervise false.
   * 
   */

//--------------------

  template <typename T>
  bool getConfigParsForType(Env& env)
  {
      shared_ptr<T> config = env.configStore().get(m_source, &m_src);
      if (config) {
        m_roiMask        = config->roiMask();
        m_numAsicsStored = config->numAsicsStored();
        ++ m_count_cfg;
        WithMsgLog(name(), info, str) {
          str << "CsPad2x2::ConfigV"    << m_count_cfg << ":";
          str << " roiMask = "          << config->roiMask();
          str << " m_numAsicsStored = " << config->numAsicsStored();
         }  
	return true;
      }
      return false;
  }
//--------------------

  template <typename TOUT>
  void save2DArrayInEventForType (Event& evt) {

      const unsigned shape[] = {NX_CSPAD2X2, NY_CSPAD2X2};
      ndarray<double,2> img_nda (&m_arr_cspad2x2_image[0][0],shape);
      
      if (typeid(TOUT) == typeid(double)) { // typeid(double).name()
        save2DArrayInEvent<double>(evt, m_src, m_outimgkey, img_nda);
        return;
      }
      
      ndarray<TOUT,2> img_out (shape);
      
      //Copy array with type changing
      typename ndarray<TOUT,2>::iterator it_out = img_out.begin(); 
      for ( ndarray<const double,2>::iterator it=img_nda.begin(); it!=img_nda.end(); ++it, ++it_out ) {
        *it_out = (TOUT)*it;
      }

      save2DArrayInEvent<TOUT>(evt, m_src, m_outimgkey, img_out);
  }

//--------------------

  template <typename T>
  void cspad_image_fill(const ndarray<const T,3>& data)
  {
    std::fill_n(&m_arr_cspad2x2_image[0][0], int(NX_CSPAD2X2*NY_CSPAD2X2), double(0));
  
    for(unsigned sect=0; sect < PC2X2::N2X1_IN_DET; ++sect) {
      if ( !(m_roiMask & (1<<sect)) ) continue;
   
        for (unsigned r=0; r<PC2X2::ROWS2X1; ++r) {
        for (unsigned c=0; c<PC2X2::COLS2X1; ++c) {
  
          int ix = int (m_pix_coords_cspad2x2 -> getPixCoor_um (PC2X2::AXIS_X, sect, r, c) * PC2X2::UM_TO_PIX);
          int iy = int (m_pix_coords_cspad2x2 -> getPixCoor_um (PC2X2::AXIS_Y, sect, r, c) * PC2X2::UM_TO_PIX);
  
          if(ix <  0)           continue;
          if(iy <  0)           continue;
          if(ix >= NX_CSPAD2X2) continue;
          if(iy >= NY_CSPAD2X2) continue;

          m_arr_cspad2x2_image[ix][iy] += (double)data[r][c][sect]; 
        }
        }
    }
  }

//--------------------

  template <typename TELEMENT>
  bool procCSPad2x2DataForType (Event& evt) {

    shared_ptr<TELEMENT> elem1 = evt.get(m_source, m_inkey, &m_src); // get m_src here

    if (elem1) {

      for (unsigned i=0; i<PC2X2::N2X1_IN_DET; i++) m_common_mode[i] = elem1->common_mode(i);

      const ndarray<const int16_t, 3>& data_nda = elem1->data();
      //const int16_t* data = &data_nda[0][0][0];

      cspad_image_fill <int16_t> (data_nda);
      cspad_image_add_in_event(evt);

      return true; 
    } // if (elem1)
    return false;
  }

//--------------------

  template <typename T>
    void procCSPad2x2NDArrForTypeAndNDArr(Event& evt, const ndarray<const T,3>& inp_nda) {
       cspad_image_fill <T> (inp_nda);
       cspad_image_add_in_event(evt);
  }
  
//--------------------

  template <typename T>
  bool procCSPad2x2NDArrForType (Event& evt) {
 
    if( m_print_bits & 8 ) MsgLog(name(), warning, "Produce image from CSPAD array, source:" << m_source 
                                  << " key:" << m_inkey << " data type:" << typeid(T).name() );
    
    shared_ptr< ndarray<const T,3> > shp_const = evt.get(m_source, m_inkey, &m_src); // get m_src here
    if (shp_const.get()) { procCSPad2x2NDArrForTypeAndNDArr<T>(evt, *shp_const.get()); return true; }
    
    shared_ptr< ndarray<T,3> > shp = evt.get(m_source, m_inkey, &m_src); // get m_src here
    if (shp.get()) { procCSPad2x2NDArrForTypeAndNDArr<T>(evt, *shp.get()); return true; }
    
    return false;
  }
  
//--------------------


}; // class CSPad2x2ImageProducer

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPAD2X2IMAGEPRODUCER_H
