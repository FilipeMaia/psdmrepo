//--------------------------
#include "PSQt/GUView.h"
//#include "PSQt/QGUtils.h"

#include <iostream>    // for std::cout
//#include <fstream>     // for std::ifstream(fname)
//#include <math.h>  // atan2
//#include <cstring> // for memcpy

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

QGraphicsView*
GUView::make_view( const float& xmin
                 , const float& xmax
                 , const float& ymin
                 , const float& ymax
                 , const unsigned pribits
                 ) 
{
  if(pribits & 1) std::cout << "  GUView::make_view:"
                            << "  xmin=" << xmin
                            << "  xmax=" << xmax
                            << "  ymin=" << ymin
                            << "  ymax=" << ymax
                            << "  pribits=" << pribits
                            << '\n';

  QGraphicsScene* scene = new QGraphicsScene;
  scene->setSceneRect (xmin, ymin, xmax-xmin, ymax-ymin);
  return new QGraphicsView(scene);
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
} // namespace PSQt
//--------------------------
