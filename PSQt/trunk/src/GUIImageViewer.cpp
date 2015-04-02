//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

//--------------------------

#include "PSQt/GUIImageViewer.h"
#include "PSQt/Logger.h"
#include "AppUtils/AppDataPath.h"
#include "PSQt/DragCenter.h"

//#include <string>
//#include <fstream>   // ofstream
//#include <iomanip>   // for setw, setfill
#include <sstream>   // for stringstream
//#include <math.h>
//#include <stdio.h>

#include <cstdlib>     // for rand()
#include <iostream>    // cout
#include <fstream>    // ifstream(fname), ofstream
//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

GUIImageViewer::GUIImageViewer( QWidget *parent )
    : Frame(parent)
    , m_colortab(0)
{
  MsgInLog(_name_(), INFO, "Open image viewer"); 

  AppUtils::AppDataPath adp_fname_def("PSQt/images/2011-08-10-Tiled-XPP.jpg"); //galaxy.jpeg"); 

  m_file      = new PSQt::WdgFile(this, "Image:", adp_fname_def.path());
  m_pointpos  = new PSQt::WdgPointPos(this, "Center x:", " y:", 0, 0, false, 60, 2);
  m_colortab  = new WdgColorTable();
  m_but_add   = new QPushButton("Add circle", this);
  m_but_cols  = new QPushButton("Colors", this);
  m_but_spec  = new QPushButton("Spectrum", this);
  m_but_rhis  = new QPushButton("r-Histo", this);
  m_image     = new PSQt::WdgImageFigs(this, m_file->fileName());
  m_imageproc = new ImageProc();
  m_radhist   = new PSQt::WdgRadHist();
  m_spechist  = new PSQt::WdgSpecHist();
  const DragBase* p_drag_center = m_image->getDragStore()->getDragCenter();

  m_hbox = new QHBoxLayout();
  m_hbox -> addWidget(m_pointpos);
  m_hbox -> addWidget(m_but_add);
  m_hbox -> addStretch(1);
  m_hbox -> addWidget(m_but_cols);
  m_hbox -> addWidget(m_but_spec);
  m_hbox -> addWidget(m_but_rhis);

  m_vbox = new QVBoxLayout();
  m_vbox -> addWidget(m_file);
  m_vbox -> addWidget(m_image);
  m_vbox -> addLayout(m_hbox);

  this -> setLayout(m_vbox);
  this -> setWindowTitle("Image Viewer");
  this -> move(100,50);  

  setStyle();
  setTips();
  
  connect(m_but_add,  SIGNAL(clicked()), this, SLOT(onButAdd()));
  connect(m_but_cols, SIGNAL(clicked()), this, SLOT(onButColorTab()));
  connect(m_but_spec, SIGNAL(clicked()), this, SLOT(onButSpec()));
  connect(m_but_rhis, SIGNAL(clicked()), this, SLOT(onButRHis()));
  connect(m_file, SIGNAL(fileNameIsChanged(const std::string&)), m_image, SLOT(onFileNameChanged(const std::string&))); 

  connect(p_drag_center, SIGNAL(centerIsMoved(const QPointF&)), m_pointpos, SLOT(setPointPos(const QPointF&))); 
  connect(m_pointpos, SIGNAL(posIsChanged(const QPointF&)), p_drag_center, SLOT(moveToRaw(const QPointF&))); 
  connect(m_pointpos, SIGNAL(posIsChanged(const QPointF&)), m_image, SLOT(forceUpdate())); 
  connect(p_drag_center, SIGNAL(centerIsChanged(const QPointF&)), m_imageproc, SLOT(onCenterIsChanged(const QPointF&))); 

  connect(m_image, SIGNAL(zoomIsChanged(int&, int&, int&, int&, float&, float&)), 
          m_imageproc, SLOT(onZoomIsChanged(int&, int&, int&, int&, float&, float&)));

  connect(m_colortab, SIGNAL(hueAnglesUpdated(const float&, const float&)), 
          m_image, SLOT(onHueAnglesUpdated(const float&, const float&)));

  connect(m_imageproc, SIGNAL(rhistIsFilled(ndarray<float, 1>&, const unsigned&, const unsigned&)), 
          m_radhist, SLOT(onRHistIsFilled(ndarray<float, 1>&, const unsigned&, const unsigned&)));

  connect(m_imageproc, SIGNAL(shistIsFilled(float*, const float&, const float&, const unsigned&)), 
          m_spechist, SLOT(onSHistIsFilled(float*, const float&, const float&, const unsigned&)));

  connect(m_colortab, SIGNAL(hueAnglesUpdated(const float&, const float&)), 
          m_spechist->colorBar(), SLOT(onHueAnglesUpdated(const float&, const float&)));

  connect(m_spechist->axes(), SIGNAL(pressOnAxes(QMouseEvent*, QPointF)),
          m_image, SLOT(onPressOnAxes(QMouseEvent*, QPointF)));

  //connect(m_spechist->axes(), SIGNAL(pressOnAxes(QMouseEvent*, QPointF)),
  //        m_imageproc, SLOT(onPressOnAxes(QMouseEvent*, QPointF)));

  // const QPointF center(m_image->getDragStore()->getCenter());
  m_pointpos->setPointPos(m_image->getDragStore()->getCenter());

  // In order to complete initialization by natural signals
  ((DragCenter*)p_drag_center) -> forceToEmitSignal();

  // open spectral and radial histogram windows
  //MsgInLog(_name_(), INFO, "c-tor is done");
}

//--------------------------

void
GUIImageViewer::setTips() 
{
  m_but_add  -> setToolTip("Add circle for current center and random radius.\n"\
                           "Then move circle clicking on it by left mouse button and drag,\n"\
                           "or remove circle clicking on it by middle mouse button.");
  m_but_cols -> setToolTip("Open/close color setting tool");
  m_but_spec -> setToolTip("Open/close spectral window");
  m_but_rhis -> setToolTip("Open/close radial projection (angul-integrated)");
}
//--------------------------

void
GUIImageViewer::setStyle()
{
  m_but_add  -> setCursor(Qt::PointingHandCursor); 
  m_but_cols -> setCursor(Qt::PointingHandCursor); 
  m_but_spec -> setCursor(Qt::PointingHandCursor); 
  m_but_rhis -> setCursor(Qt::PointingHandCursor); 
}
 
//--------------------------

void 
GUIImageViewer::resizeEvent(QResizeEvent *event)
{
  stringstream ss; ss << "Image Viewer, w=" << event->size().width() << " h=" <<  event->size().height();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
GUIImageViewer::closeEvent(QCloseEvent *event)
{
  if (m_colortab) m_colortab->close(); 
  if (m_radhist ) m_radhist ->close(); 
  if (m_spechist) m_spechist->close(); 

  QWidget::closeEvent(event);
  stringstream ss; ss << "Close image viewer: event type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

void
GUIImageViewer::moveEvent(QMoveEvent *e)
{
  stringstream ss; ss << _name_() << "Image Viewer, x=" << e->pos().x() << " y=" << e->pos().y();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
GUIImageViewer::mousePressEvent(QMouseEvent *e)
{
  stringstream ss; ss << _name_() << "Image Viewer, x=" << e->pos().x() << " y=" << e->pos().y();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
GUIImageViewer::showChildWindows()
{
  onButSpec();
  onButRHis();
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
GUIImageViewer::onButExit()
{
  MsgInLog(_name_(), DEBUG, "onButExit");
  this->close(); // will call closeEvent(...)
}

//--------------------------

void 
GUIImageViewer::message(QWidget* wdg, const char* cmt)
{
  stringstream ss; ss << cmt << " window " << ((wdg->isVisible()) ? "is open" : "closed");
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

void 
GUIImageViewer::onButRHis()
{
  static unsigned counter=0; 
  if(!counter++)
    m_radhist->move(this->pos().x() + this->size().width() + 8, this->pos().y());      
  m_radhist->setVisible(! (m_radhist->isVisible()));
  this->message(m_radhist, "Radial histogram");
}

//--------------------------

void 
GUIImageViewer::onButSpec()
{
  static unsigned counter=0; 
  if(!counter++)
    m_spechist -> move(this->pos().x() + this->size().width() + 8, 
                       this->pos().y() + 363);      
  m_spechist->setVisible(! (m_spechist->isVisible()));
  this->message(m_spechist, "Spectral histogram");
}

//--------------------------

void 
GUIImageViewer::onButAdd()
{
  MsgInLog(_name_(), DEBUG, "onButAdd");
  float rad_raw = 110+rand()%180;
  this -> m_image -> addCircle(rad_raw);
}

//--------------------------

void 
GUIImageViewer::onButColorTab()
{
  static unsigned counter=0; 
  if(!counter++)
    m_colortab -> move(this->pos().x() + 8, this->pos().y());      
  //m_colortab -> move(this->pos().x() + this->size().width() + 8, this->pos().y());      
  m_colortab->setVisible(! (m_colortab->isVisible()));
  
  this->message(m_colortab, "Color table");
}

//--------------------------

void 
GUIImageViewer::onImageIsUpdated(ndarray<GeoImage::raw_image_t,2>& nda)
{
  MsgInLog(_name_(), DEBUG, "onImageIsUpdated()");

  this -> wdgImage() -> onImageIsUpdated(nda);
  m_imageproc -> onImageIsUpdated(nda);
}

//--------------------------

} // namespace PSQt

//--------------------------
