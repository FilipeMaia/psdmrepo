//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

//--------------------------
#include "PSQt/WdgSpecHist.h"
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

WdgSpecHist::WdgSpecHist(QWidget *parent)
  : QWidget(parent)
//: Frame(parent)
  , m_axes(0)
  , m_path_item(0)
{
  //  setWdgParams();
  float xmin = 0;
  float xmax = 100;
  float ymin = 0;
  float ymax = 100;
  unsigned nxdiv=4;
  unsigned nydiv=4;

  //m_axes = new PSQt::GUAxes(0, xmin, xmax, ymin, ymax, nxdiv, nydiv);
  m_axes = new PSQt::GUAxes(0);
  m_axes->setLimits(xmin, xmax, ymin, ymax, nxdiv, nydiv);

  //m_cbar = new PSQt::WdgColorBar(this, 0, 0, 1024, PSQt::HR, 0.1);
  m_cbar = new PSQt::WdgColorBar(this);
  m_cbar->setMinimumSize(100, 20);

  m_hbox = new QHBoxLayout();
  m_hbox -> addSpacing(75);
  m_hbox -> addWidget(m_cbar);
  m_hbox -> addSpacing(75);
  //m_hbox -> addStretch(1);

  m_vbox = new QVBoxLayout();
  m_vbox -> addWidget(m_axes);
  m_vbox -> addLayout(m_hbox);

  this -> setLayout(m_vbox);
  this->setContentsMargins(-9,-9,-9,-9);
  this->setWindowTitle(_name_());
}

//--------------------------

WdgSpecHist::~WdgSpecHist()
{
  //  if (m_frame)      delete m_frame;  
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
WdgSpecHist::onSHistIsFilled(float* arr, const float& amin, const float& amax, const unsigned& nbins)
{
  std::stringstream ss; ss <<  "Show spectral histogram with nbins=" << nbins
                           <<  "  amin=" << amin
                           <<  "  amax=" << amax;
  MsgInLog(_name_(), INFO, ss.str());   

  float x1[nbins+1];
  float y1[nbins];

  float xmin = amin;
  float xmax = amax;
  float ymin = arr[0];
  float ymax = arr[0];
  float dx = (amax-amin)/nbins;
  float x = xmin;
  float y = 0;

  for(unsigned i=0; i<nbins; x+=dx, i++) {
    x1[i] = x;
    y = arr[i];
    y1[i] = y;
    if(y>ymax) ymax=y;
    if(y<ymin) ymin=y;
  }
  x1[nbins] = x1[nbins-1] + dx;

  m_axes->setLimits(xmin, xmax, ymin, ymax);

  //PSQt::graph(m_axes, x1, y1, npts, "-rT4");
  if(m_path_item) {
    m_axes->pscene()->removeItem(m_path_item);
    delete m_path_item;
  }
  m_path_item = PSQt::hist(m_axes, x1, y1, nbins, "-bT");
 }

//--------------------------
} // namespace PSQt
//--------------------------
