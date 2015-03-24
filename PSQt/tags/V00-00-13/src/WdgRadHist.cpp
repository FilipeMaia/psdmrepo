//--------------------------
#include "PSQt/WdgRadHist.h"
#include "PSQt/Logger.h"
#include "ndarray/ndarray.h"

#include "PSQt/QGUtils.h"

#include <sstream>  // for stringstream
#include <iostream>    // for std::cout
//#include <fstream>     // for std::ifstream(fname)
//#include <math.h>  // atan2
//#include <cstring> // for memcpy

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

WdgRadHist::WdgRadHist(QWidget *parent)
  : QWidget(parent)
  , m_axes(0)
  , m_path_item(0)
{
  //  setWdgParams();
  float xmin = 0;
  float xmax = 1000;
  float ymin = 0;
  float ymax = 1000;
  unsigned nxdiv=4;
  unsigned nydiv=4;

  //m_axes = new PSQt::GUAxes(0, xmin, xmax, ymin, ymax, nxdiv, nydiv);
  m_axes = new PSQt::GUAxes(0);
  m_axes->setLimits(xmin, xmax, ymin, ymax, nxdiv, nydiv);

  m_vbox = new QVBoxLayout();
  m_vbox -> addWidget(m_axes);

  this -> setLayout(m_vbox);
  this->setContentsMargins(-9,-9,-9,-9);
}

//--------------------------

WdgRadHist::~WdgRadHist()
{
  //  if (m_frame)      delete m_frame;  
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
WdgRadHist::onRHistIsFilled(ndarray<float, 1>& nda, const unsigned& zirmin, const unsigned& zirmax)
{
  std::stringstream ss; ss <<  "::onRHistIsFilled() size()=" << nda.size()
                           <<  "  zirmin=" << zirmin
                           <<  "  zirmax=" << zirmax;
  MsgInLog(_name_(), INFO, ss.str());   

  unsigned npts = zirmax-zirmin;
  float x1[npts];
  float y1[npts];

  float xmin = zirmin;
  float xmax = zirmax;
  float ymin = nda[0];
  float ymax = nda[0];
  float dx = 1;
  float x = xmin;
  float y = 0;

  for(unsigned i=0; i<npts; x+=dx, i++) {
    x1[i] = x;
    y = nda[i+zirmin];
    y1[i] = y;
    if(y>ymax) ymax=y;
    if(y<ymin) ymin=y;
  }
  m_axes->setLimits(xmin, xmax, ymin, ymax);

  //PSQt::graph(m_axes, x1, y1, npts, "-rT4");
  if(m_path_item) {
    m_axes->pscene()->removeItem(m_path_item);
    delete m_path_item;
  }
  m_path_item = PSQt::hist(m_axes, x1, y1, npts, "-bT");
 }

//--------------------------
} // namespace PSQt
//--------------------------
