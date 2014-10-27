//--------------------------

#include "PSQt/QGUtils.h"
#include <math.h>    // atan2, abs, fmod

//#include "PSQt/WdgImage.h"
//#include "ndarray/ndarray.h" // for img_from_pixel_arrays(...)

#include <iostream>    // for std::cout
//#include <fstream>   // for std::ifstream(fname)

using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

//--------------------------

uint32_t
fRGBA(const float R, const float G, const float B) // R, G, B = [0,1]
{
  unsigned r = unsigned(R*255);
  unsigned g = unsigned(G*255);
  unsigned b = unsigned(B*255);
  return uint32_t(0xFF000000 + (r<<16) + (g<<8) + b);  
}

//--------------------------

uint32_t 
HSV2RGBA(const float H, const float S, const float V) // H=[0,360), S, V = [0,1]
{
  // extend hue from [0,+360] to any angle in cycle
  float HE = fmod(H,360);

  float fhr = (HE<0) ? (HE+360)/60 : HE/60 ; 
  int   ihr = int(fhr); // [0,5]
  float rem = fhr - ihr;

  float p = V * (1 - S);
  float q = V * (1 - (S * rem));
  float t = V * (1 - (S * (1 - rem)));

  switch(ihr) {
    case  0: return fRGBA(V,t,p);
    case  1: return fRGBA(q,V,p);
    case  2: return fRGBA(p,V,t);
    case  3: return fRGBA(p,q,V);
    case  4: return fRGBA(t,p,V);
    case  5:
    default: return fRGBA(V,p,q);
  }
}

//--------------------------

uint32_t*
ColorTable(const unsigned& NColors, const float& H1, const float& H2) // H1, H2=[-360,360]
{
  uint32_t* ctable = new uint32_t[NColors];

  const float S=1; 
  const float V=1;
  const float dH = (H2-H1)/NColors;
  float H=H1;
  for(unsigned i=0; i<NColors; ++i, H+=dH) {
    ctable[i] = HSV2RGBA(H, S, V);
  }
  return ctable;
}

//--------------------------

void 
setPixmapForLabel(const QImage& image, QPixmap*& pixmap, QLabel*& label)
{
  if(pixmap) delete pixmap;
  pixmap = new QPixmap(QPixmap::fromImage(image));
  //else pixmap -> loadFromData ( (const uchar*) &dimg[0], unsigned(rows*cols) );

  label->setPixmap(pixmap->scaled(label->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
  //label->setPixmap(*pixmap);
}

//--------------------------

int string_to_int(const std::string& str)
{
  return atoi( str.c_str() );
}

//--------------------------

} // namespace PSQt

//--------------------------
