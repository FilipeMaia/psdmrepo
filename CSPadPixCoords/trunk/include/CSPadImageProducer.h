#ifndef CSPADPIXCOORDS_CSPADIMAGEPRODUCER_H
#define CSPADPIXCOORDS_CSPADIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadImageProducer.
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
#include "PSCalib/CSPadCalibPars.h"

#include "CSPadPixCoords/QuadParameters.h"
#include "CSPadPixCoords/PixCoords2x1.h"
#include "CSPadPixCoords/PixCoordsQuad.h"
#include "CSPadPixCoords/PixCoordsCSPad.h"

#include "CSPadPixCoords/Image2D.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "PSEvt/Source.h"


//		---------------------
// 		-- Class Interface --
//		---------------------

using namespace std;

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief CSPadImageProducer produces the CSPad image for each event and add it to the event in psana framework.
 *
 *  CSPadImageProducer works in psana framework. It does a few operation as follows:
 *  1) get the pixel coordinates from PixCoords2x1, PixCoordsQuad, and PixCoordsCSPad classes,
 *  2) get data from the event,
 *  3) produce the Image2D object with CSPad image for each event,
 *  4) add the Image2D object in the event for further modules.
 *
 *  Time consumed to fill the CSPad image array (currently [1750][1750]) 
 *  is measured to be about 40 msec/event on psana0105. 
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

class CSPadImageProducer : public Module {
public:

  // Default constructor
  CSPadImageProducer (const std::string& name) ;

  // Destructor
  virtual ~CSPadImageProducer () ;

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
  void getQuadConfigPars(Env& env);

  void procEvent(Event& evt, Env& env);
  void getCSPadConfigFromData(Event& evt);

  //void cspad_image_init();
  //void cspad_image_fill(const int16_t* data, CSPadPixCoords::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar);
  //void cspad_image_save_in_file(const std::string &filename = "cspad_image.txt");
  //void cspad_image_add_in_event(Event& evt);

private:

  // Data members, this is for example purposes only

  std::string m_calibDir;       // i.e. ./calib
  std::string m_typeGroupName;  // i.e. CsPad::CalibV1
  std::string m_str_src;        // i.e. CxiDs1.0:Cspad.0
   
  Source      m_source;         // Data source set from config file
  Pds::Src    m_src;
  std::string m_inkey; 
  std::string m_imgkey;   // i.e. "CSPad:Image"
  bool     m_tiltIsApplied;
  unsigned m_print_bits;
  long     m_count;


  // Parameters form Psana::CsPad::ConfigV# object
  uint32_t m_numQuadsInConfig;
  uint32_t m_roiMask        [4];
  uint32_t m_numAsicsStored [4];

  // Parameters form Psana::CsPad::DataV# and Psana::CsPad::ElementV# object
  uint32_t m_numQuads;
  uint32_t m_quadNumber     [4];


  uint32_t m_n2x1;         // 8
  uint32_t m_ncols2x1;     // 185
  uint32_t m_nrows2x1;     // 388
  uint32_t m_sizeOf2x1Img; // 185*388;

  PSCalib::CSPadCalibPars        *m_cspad_calibpar;
  CSPadPixCoords::PixCoords2x1   *m_pix_coords_2x1;
  CSPadPixCoords::PixCoordsQuad  *m_pix_coords_quad;
  CSPadPixCoords::PixCoordsCSPad *m_pix_coords_cspad;

  CSPadPixCoords::PixCoords2x1::COORDINATE XCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE YCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE ZCOOR;
	
  //uint32_t   m_cspad_ind;
  double    *m_coor_x_pix;
  double    *m_coor_y_pix;
  uint32_t  *m_coor_x_int;
  uint32_t  *m_coor_y_int;

  enum{ NX_QUAD=850, 
        NY_QUAD=850 };

  enum{ NX_CSPAD=1750, 
        NY_CSPAD=1750,
        IMG_SIZE=NX_CSPAD*NY_CSPAD };
//  double m_arr_cspad_image[NX_CSPAD][NY_CSPAD];

//-------------------
  /**
   * @brief Gets m_numQuadsInConfig, m_roiMask[q] and m_numAsicsStored[q] from the Psana::CsPad::ConfigV# object.
   * 
   */

  template <typename T>
  bool getQuadConfigParsForType(Env& env) {

        shared_ptr<T> config = env.configStore().get(m_source);
        if (config.get()) {
            m_numQuadsInConfig = config->numQuads();
            for (uint32_t q = 0; q < config->numQuads(); ++ q) {
              m_roiMask[q]         = config->roiMask(q);
              m_numAsicsStored[q]  = config->numAsicsStored(q);
            }
	    return true;
	}
	return false;
  }

//-------------------
  /**
   * @brief Gets m_numQuads and m_quadNumber[q] from the Psana::CsPad::DataV# and ElementV# objects.
   * 
   */

  template <typename TDATA, typename TELEMENT>
  bool getCSPadConfigFromDataForType(Event& evt) {

    shared_ptr<TDATA> data = evt.get(m_source, m_inkey, &m_src);
    if (data.get()) {
      m_numQuads = data->quads_shape()[0];

      cout << "m_numQuads = " << m_numQuads << "   m_quadNumber[q] = ";
      for (uint32_t q = 0; q < m_numQuads; ++ q) {
        const TELEMENT& el = data->quads(q);
        m_quadNumber[q] = el.quad();
        cout << m_quadNumber[q];
      }
      cout << endl;
      return true;
    }
    return false;
  }

//--------------------
  /**
   * @brief Adds image in the event as ndarray<T,2> or Image2D<T>, depending on m_imgkey.
   * 
   * @param[in]  arr2d    pointer to the data array with image of type T.
   */

  template <typename T>
  void addImageInEventForType(Event& evt, T* arr2d)
  {
    if(m_imgkey == "Image2D") {

      shared_ptr< CSPadPixCoords::Image2D<T> > img2d( new CSPadPixCoords::Image2D<T>(arr2d, NY_CSPAD, NX_CSPAD) );
      evt.put(img2d, m_src, m_imgkey);

    } else {

      const unsigned shape[] = {NY_CSPAD,NX_CSPAD};
      shared_ptr< ndarray<T,2> > img2d( new ndarray<T,2>(arr2d, shape) );
      evt.put(img2d, m_src, m_imgkey);
    }
  }

//-------------------
  /**
   * @brief Fills a part of the image (img_nda) for one quad per call.
   * 
   * @param[in]  data            pointer to the beginning of the data array for quad.
   * @param[in]  quadpars        pointer to the object with configuration parameters for quad.
   * @param[in]  cspad_calibpar  pointer to the object with geometry calibration parameters for CSPAD.
   * @param[out] img_nda         reference to the ndarray<T,2> with CSPAD image.
   */
  
  template <typename T>
  void cspadImageFillForType(const T* data, CSPadPixCoords::QuadParameters* quadpars, PSCalib::CSPadCalibPars* cspad_calibpar, ndarray<T,2>& img_nda) 
  {
        int       quad    = quadpars -> getQuadNumber();
        uint32_t  roiMask = quadpars -> getRoiMask();

        int ind_in_arr = 0;

	for(uint32_t sect=0; sect < m_n2x1; sect++)
	{
	     bool bitIsOn = roiMask & (1<<sect);
	     if( !bitIsOn ) continue; 

	     int pix_in_cspad = (quad*m_n2x1 + sect) * m_sizeOf2x1Img;
 
             const T *data2x1 = &data[ind_in_arr * m_sizeOf2x1Img];
             //cout  << "  add section " << sect << endl;	     
 
             for (uint32_t c=0; c<m_ncols2x1; c++) {
             for (uint32_t r=0; r<m_nrows2x1; r++) {

               // This access takes 72ms/cspad
               //int ix = (int) m_pix_coords_cspad -> getPixCoor_pix (XCOOR, quad, sect, r, c);
               //int iy = (int) m_pix_coords_cspad -> getPixCoor_pix (YCOOR, quad, sect, r, c);

               // This access takes 40ms/cspad
               int ix = m_coor_x_int [pix_in_cspad];
               int iy = m_coor_y_int [pix_in_cspad];
	       pix_in_cspad++;

	       if(ix <  0)        continue;
	       if(iy <  0)        continue;
	       if(ix >= NX_CSPAD) continue;
	       if(iy >= NY_CSPAD) continue;

               img_nda[ix][iy] += (T)data2x1[c*m_nrows2x1+r];
               //m_arr_cspad_image[ix][iy] += (double)data2x1[c*m_nrows2x1+r];
             }
             }
             ++ind_in_arr;
 	}
  }

//-------------------
  /**
   * @brief For requested m_source and m_inkey process Psana::CsPad::DataV1, or V2
   * Returns false if data is missing.
   */

  template <typename TDATA, typename TELEMENT>
  bool procCSPadDataForType (Event& evt) {

      typedef int16_t data_cspad_t;

      shared_ptr<TDATA> data_obj = evt.get(m_source, m_inkey, &m_src); // get m_src here
      
      if (data_obj.get()) {
      
        const unsigned shape[] = {NY_CSPAD,NX_CSPAD};
        ndarray<data_cspad_t,2> img_nda(shape);
        data_cspad_t* p_image = img_nda.data();
        std::fill_n(p_image, int(IMG_SIZE), data_cspad_t(0));    
        //std::fill_n(&m_arr_cspad_image[0][0], int(NX_CSPAD*NY_CSPAD), double(0));
      
        int nQuads = data_obj->quads_shape()[0];
        //cout << "nQuads = " << nQuads << endl;
        for (int q = 0; q < nQuads; ++ q) {
            const TELEMENT& el = data_obj->quads(q);
      
            //const data_cspad_t* data = el.data(); // depricated stuff
            int quad = el.quad() ;
            const ndarray<const data_cspad_t,3>& data_nda = el.data();
            const data_cspad_t* data = &data_nda[0][0][0];
      
            CSPadPixCoords::QuadParameters *quadpars = new CSPadPixCoords::QuadParameters(quad, NX_QUAD, NY_QUAD, m_numAsicsStored[q], m_roiMask[q]);
      
            cspadImageFillForType<data_cspad_t>(data, quadpars, m_cspad_calibpar, img_nda);
        }
      
        //addImageInEventForType<double>(evt, &m_arr_cspad_image[0][0]);
        addImageInEventForType<data_cspad_t>(evt, img_nda.data());

        return true;
      } // if (data_obj.get())
      return false;
  }

//-------------------
  /**
   * @brief For requested m_source and m_inkey process CSPAD data ndarray<T,3>
   * Returns false if data is missing.
   */

  template <typename T>
  bool procCSPadNDArrForType (Event& evt) {

      const unsigned shape[] = {NY_CSPAD,NX_CSPAD};
      ndarray<T,2> img_nda(shape);
      T* p_image = img_nda.data();
      std::fill_n(p_image, int(IMG_SIZE), T(0));    

      shared_ptr< ndarray<T,3> > shp = evt.get(m_source, m_inkey, &m_src); // get m_src here
      if (shp.get()) {
      
        const ndarray<T,3> data = *shp.get(); //const T* p_data = shp->data();

	int ind2x1_in_arr = 0;

        for (uint32_t q = 0; q < m_numQuads; ++ q) {
            const T* data_quad = &data[ind2x1_in_arr][0][0]; 
            CSPadPixCoords::QuadParameters *quadpars = new CSPadPixCoords::QuadParameters(m_quadNumber[q], NX_QUAD, NY_QUAD, m_numAsicsStored[q], m_roiMask[q]);      

            cspadImageFillForType<T>(data_quad, quadpars, m_cspad_calibpar, img_nda);

            ind2x1_in_arr += m_numAsicsStored[q];
	}

        addImageInEventForType<T>(evt, img_nda.data());

        return true;
      } // if (shp.get())

    return false;
  }

//-------------------

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPADIMAGEPRODUCER_H
