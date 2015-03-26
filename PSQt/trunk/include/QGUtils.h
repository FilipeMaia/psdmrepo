#ifndef QGUTILS_H
#define QGUTILS_H

#include "ndarray/ndarray.h"
#include "PSQt/Logger.h"

#include "PSQt/GUAxes.h"
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPathItem>

#include <math.h>   // atan2, abs, fmod
#include <string> 
#include <sstream>  // for stringstream
#include <stdint.h> // uint8_t, uint32_t, etc.
#include <iomanip>  // for setw
//#include "PSQt/GeoImage.h"
//#include <QPainter>
//#include <QPen>

#include <QLabel>
#include <QPainter>
#include <QPixmap>
#include <QFont>

namespace PSQt {

//--------------------------

/**
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see 
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

//--------------------------

/**
 * @brief Returns uint32_t color code for R, G, and B parameters.
 * R - red   [0,1] 
 * G - green [0,1] 
 * B - blue  [0,1] 
 */ 
uint32_t fRGBA(const float R, const float G, const float B);

/**
 * @brief Returns uint32_t color code for H, S, and V parameters.
 * H - hue [0,360) 
 * S - black/white satturation [0,1] 
 * V - value of color [0,1] 
 * ??? http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
 * http://en.wikipedia.org/wiki/HSL_and_HSV
 * http://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
 */ 
uint32_t HSV2RGBA(const float H, const float S, const float V);

/**
 * @brief Returns uint32_t* color table for requested parameters.
 * NColors - number of colors
 * H1, H2=[-360,360] - extended to negative hue range
 */ 
uint32_t*
  ColorTable(const unsigned& NColors=0, const float& H1=0, const float& H2=0);

/**
 * @brief Returns 2-d ndarray with colorbar image
 * @param[in] rows - number of rows in the color bar image
 * @param[in] cols - number of columns and colors in the color bar image *
 * @param[in] hue1 - first limit of the hue angle [0,360]
 * @param[in] hue2 - second limit of the hue angle [0,360]
 */ 
ndarray<uint32_t, 2> 
getColorBarImage(const unsigned& rows =   20, 
                 const unsigned& cols = 1024,
                 const float&    hue1 = -120,
                 const float&    hue2 =   60) ;

/**
 * @brief Sets image as a pixmap for label
 */ 
void 
  setPixmapForLabel(const QImage& image, QPixmap*& pixmap, QLabel*& label);

//--------------------------

 int string_to_int(const std::string& str);

//--------------------------

 void print_font_status(const QFont& font);

//--------------------------
//--------------------------

template<typename T>
  std::string val_to_string(const T& v, const unsigned& prec=0)
  {
    std::stringstream ss; ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
  }

//--------------------------

 void set_pen(QPen & pen, const std::string & opt);
 void set_brush(QBrush & brush, const std::string & opt);
 void set_pen_brush(QPen & pen, QBrush & brush, const std::string & opt);

//--------------------------

 QGraphicsPathItem* 
   graph(QGraphicsScene* scene, const float* x, const float* y, const int n=1, const std::string& opt=std::string("-bT"));

 QGraphicsPathItem* 
   graph(QGraphicsView * view,  const float* x, const float* y, const int n=1, const std::string& opt=std::string("-bT"));

//--------------------------

 QGraphicsPathItem* 
   hist(QGraphicsScene* scene, const float* x, const float* y, const int n=1, const std::string& opt=std::string("-bT"));

 QGraphicsPathItem* 
   hist(QGraphicsView * view,  const float* x, const float* y, const int n=1, const std::string& opt=std::string("-bT"));

//--------------------------
 
 bool is_file(const std::string& path);
 bool is_directory(const std::string& path);
 bool is_link(const std::string& path);

//--------------------------
 
/**
 * @brief check if directory exists
 */ 
 bool dir_exists(const std::string& dir);

//--------------------------
 
/**
 * @brief check if file exists
 */ 
 bool file_exists (const std::string& fname);

//--------------------------
 
 bool path_exists (const std::string& path);

//--------------------------
 
 std::string basename(const std::string& path);
 std::string dirname(const std::string& path);

//--------------------------

 std::string split_string_left(const std::string& s, size_t& pos, const char& sep);

//--------------------------
 void splitext(const std::string& path, std::string& root, std::string& ext);

//--------------------------
 std::string stringTimeStamp(const std::string& format=std::string("%Y-%m-%d-%H:%M:%S"));

//--------------------------
 std::string getFileNameWithTStamp(const std::string& fname);

//--------------------------
 std::string getGeometryFileName(const std::string& fname="geometry.txt", const bool& add_tstamp=true);

//--------------------------

template <typename T>
void
  getMinMax(const ndarray<T,2>& nda, float& vmin, float& vmax)
  {
    float v = 0;
    vmin = nda[0][0];
    vmax = nda[0][0];
    typename ndarray<T, 2>::iterator itd = nda.begin();
    for(; itd!=nda.end(); ++itd) { 
      v = float(*itd);
      if( v < vmin ) vmin = v;
      if( v > vmax ) vmax = v;
    }
  }

//--------------------------

template <typename T>
void
  getAveRms(const ndarray<T,2>& nda, double& ave, double& rms)
  {
    double val = 0;
    double s0 = 0;
    double s1 = 0;
    double s2 = 0;

    typename ndarray<T, 2>::iterator itd = nda.begin();
    for(; itd!=nda.end(); ++itd) { 
      val = *itd;
      s0 += 1;
      s1 += val;
      s2 += val*val;
    }
    
    if(s0) {
      s1 /= s0;
      s2 /= s0;
    }

    ave = s1;
    rms = sqrt(s2-s1*s1);
  }

//--------------------------

template <typename T>
ndarray<uint32_t,2> 
  getUint32NormalizedImage (const ndarray<T,2>& dnda, const float& amin=0, const float& amax=0, const unsigned& ncolors=0, const float& hue1=-120, const float& hue2=-360)
{
  typedef uint32_t image_t;

  MsgInLog("QGUtils", DEBUG, "getUint32NormalizedImage(): convert raw ndarray to colored uint32_t");

  ndarray<image_t, 2> inda(dnda.shape());

  float dmin, dmax;

  if (amin || amax) {
    dmin = amin;
    dmax = amax;
  }
  else {
    getMinMax(dnda, dmin, dmax);

    //double dave, drms;
    //getAveRms(dnda, dave, drms);
    //dmin = dave - 1*drms;
    //dmax = dave + 10*drms;
  }

  std::stringstream ssd; ssd << "getUint32NormalizedImage():" 
                             << " dmin:"  << dmin
                             << " dmax:"  << dmax
                             << " h1:"    << hue1
                             << " h2:"    << hue2
                             << " ncols:" << ncolors;

  MsgInLog("QGUtils", DEBUG, ssd.str());

  typename ndarray<T, 2>::iterator itd = dnda.begin();

  //double dmin = dnda[0][0];
  //double dmax = dnda[0][0];

  //for(; itd!=dnda.end(); ++itd) { 
  //  if( *itd < dmin ) dmin = *itd;
  //  if( *itd > dmax ) dmax = *itd;
  //}

  //std::stringstream ssd; ssd << "getUint32NormalizedImage(): dmin: " << dmin << " dmax: " << dmax;
  //MsgInLog("QGUtils", DEBUG, ssd.str());

  // Convert image of type T to uint32_t Format_ARGB32
  if (ncolors) {
    //const unsigned ncolors = 1024;
    float k = (dmax-dmin) ? ncolors/(dmax-dmin) : 1; 
    image_t* ctable = ColorTable(ncolors, hue1, hue2);
    ndarray<image_t, 2>::iterator iti;
    for(itd=dnda.begin(), iti=inda.begin(); itd!=dnda.end(); ++itd, ++iti) { 
      float da = *itd-dmin;
      unsigned cind = (da>0) ? unsigned(da*k) : 0;
      cind = (cind<ncolors) ? cind : ncolors-1;
      *iti = ctable[cind]; // converts to 24bits adds alpha layer
    }
    delete [] ctable;
  }
  else {
    T k = (dmax-dmin) ? 0xFFFFFF/(dmax-dmin) : 1; 
    ndarray<image_t, 2>::iterator iti;
    for(itd=dnda.begin(), iti=inda.begin(); itd!=dnda.end(); ++itd, ++iti) { 
      *iti = image_t( (*itd-dmin)*k ) + 0xFF000000; // converts to 24bits adds alpha layer
    }
  }    

  return inda;
}

//--------------------------

} // namespace PSQt

#endif // QGUTILS_H
