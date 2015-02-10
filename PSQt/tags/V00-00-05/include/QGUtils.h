#ifndef QGUTILS_H
#define QGUTILS_H

#include <string> 
#include <sstream> // for stringstream
#include <stdint.h> // uint8_t, uint32_t, etc.
//#include "PSQt/GeoImage.h"
//#include <QPainter>
//#include <QPen>
#include "PSQt/GUAxes.h"
#include <QGraphicsScene>
#include <QGraphicsView>

#include <QLabel>
#include <QPainter>
#include <QPixmap>
#include <QFont>

namespace PSQt {
//--------------------------

/**
 * R - red   [0,1] 
 * G - green [0,1] 
 * B - blue  [0,1] 
 */ 
uint32_t fRGBA(const float R, const float G, const float B);

/**
 * H - hue [0,360) 
 * S - black/white satturation [0,1] 
 * V - value of color [0,1] 
 * ??? http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
 * http://en.wikipedia.org/wiki/HSL_and_HSV
 * http://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
 */ 
uint32_t HSV2RGBA(const float H, const float S, const float V);

/**
 * NColors - number of colors
 * H1, H2=[-360,360] - extended to negative hue range
 */ 
uint32_t*
  ColorTable(const unsigned& NColors=1024, const float& H1=-120, const float& H2=-360);

/**
 * Sets image as a pixmap for label
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
  std::string val_to_string(const T& v)
  {
    std::stringstream ss; ss << v;
    return ss.str();
  }

//--------------------------

 void set_pen(QPen & pen, const std::string & opt);
 void set_brush(QBrush & brush, const std::string & opt);
 void set_pen_brush(QPen & pen, QBrush & brush, const std::string & opt);

//--------------------------

 void graph(QGraphicsScene* scene, const float* x, const float* y, const int n=1, const std::string& opt=std::string("-bT"));
 void graph(QGraphicsView * view,  const float* x, const float* y, const int n=1, const std::string& opt=std::string("-bT"));
 void graph(PSQt::GUAxes  * axes,  const float* x, const float* y, const int n=1, const std::string& opt=std::string("-bT"));

//--------------------------
 
 bool is_file(const std::string& path);
 bool is_directory(const std::string& path);
 bool is_link(const std::string& path);

//--------------------------
 
/**
 * check if directory exists
 */ 
 bool dir_exists(const std::string& dir);

//--------------------------
 
/**
 * check if file exists
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

} // namespace PSQt

#endif // QGUTILS_H
