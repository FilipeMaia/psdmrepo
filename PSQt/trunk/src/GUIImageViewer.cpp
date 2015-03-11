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
//  : QWidget(parent)
    , m_colortab(0)
{
  //const std::string base_dir = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/";
  //const std::string fname_geo = base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data"; 
  //const std::string fname_img = base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt"; 

  AppUtils::AppDataPath adp_fname_def("PSQt/images/2011-08-10-Tiled-XPP.jpg"); //galaxy.jpeg"); 

  m_but_add  = new QPushButton("Add circle", this);
  m_but_cols = new QPushButton("Colors", this);
  m_file     = new PSQt::WdgFile(this, "Image:", adp_fname_def.path());
  m_pointpos = new PSQt::WdgPointPos(this, "Center x:", " y:", 0, 0, false, 60, 2);
  m_colortab = new WdgColorTable();
  m_but_spec = new QPushButton("Spectrum", this);
  m_but_rhis = new QPushButton("r-Histo", this);

  m_but_add  -> setCursor(Qt::PointingHandCursor); 
  m_but_cols -> setCursor(Qt::PointingHandCursor); 
  m_but_spec -> setCursor(Qt::PointingHandCursor); 
  m_but_rhis -> setCursor(Qt::PointingHandCursor); 

  //m_image = new PSQt::WdgImage(this, m_file->fileName());
  m_image = new PSQt::WdgImageFigs(this, m_file->fileName());
  m_imageproc = new ImageProc();
  
  connect(m_but_add,  SIGNAL( clicked() ), this, SLOT(onButAdd()) );
  connect(m_but_cols, SIGNAL( clicked() ), this, SLOT(onButColorTab()) );
  connect(m_but_spec, SIGNAL( clicked() ), this, SLOT(onButSpec()) );
  connect(m_but_rhis, SIGNAL( clicked() ), this, SLOT(onButRHis()) );
  connect(m_file, SIGNAL(fileNameIsChanged(const std::string&)), m_image, SLOT(onFileNameChanged(const std::string&)) ); 

  const DragBase* p_drag_center = m_image->getDragStore()->getDragCenter();
  connect(p_drag_center, SIGNAL(centerIsMoved(const QPointF&)), m_pointpos, SLOT(setPointPos(const QPointF&))); 
  connect(m_pointpos, SIGNAL(posIsChanged(const QPointF&)), p_drag_center, SLOT(moveToRaw(const QPointF&))); 
  connect(m_pointpos, SIGNAL(posIsChanged(const QPointF&)), m_image, SLOT(forceUpdate())); 
  connect(p_drag_center, SIGNAL(centerIsChanged(const QPointF&)), m_imageproc, SLOT(onCenterIsChanged(const QPointF&))); 
  connect(m_image, SIGNAL(zoomIsChanged(int&, int&, int&, int&)), m_imageproc, SLOT(onZoomIsChanged(int&, int&, int&, int&)));

  connect(m_colortab, SIGNAL(hueAnglesUpdated(const float&, const float&)), 
          m_image, SLOT(onHueAnglesUpdated(const float&, const float&)));

  const QPointF center(m_image->getDragStore()->getCenter());
  m_pointpos->setPointPos(center);
  //m_imageproc->onCenterIsChanged(center);

  // In order to complete initialization by natural signals
  ((DragCenter*)p_drag_center) -> forceToEmitSignal();


  QHBoxLayout *hbox = new QHBoxLayout();
  hbox -> addWidget(m_pointpos);
  hbox -> addWidget(m_but_add);
  hbox -> addStretch(1);
  hbox -> addWidget(m_but_cols);
  hbox -> addWidget(m_but_spec);
  hbox -> addWidget(m_but_rhis);

  QVBoxLayout *vbox = new QVBoxLayout();
  vbox -> addWidget(m_file);
  vbox -> addWidget(m_image);
  vbox -> addLayout(hbox);

  this -> setLayout(vbox);
  this -> setWindowTitle(tr("Basic Drawing"));
  this -> move(100,50);  

  showTips();
  //MsgInLog(_name_(), INFO, "c-tor is done");
}

//--------------------------

void
GUIImageViewer::showTips() 
{
  m_but_spec -> setToolTip("Open/close spectral window");
  m_but_rhis -> setToolTip("Open/close radial projection (angul-integrated)");
  m_but_add  -> setToolTip("Add circle for current center and random radius.\n"\
                           "Then move circle clicking on it by left mouse button and drag,\n"\
                           "or remove circle clicking on it by middle mouse button.");
  m_but_cols -> setToolTip("Open/close color setting tool");
}

//--------------------------

void 
GUIImageViewer::resizeEvent(QResizeEvent *event)
{
  //  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  stringstream ss; ss << "Window is resized, w:" << event->size().width() << " h:" <<  event->size().height();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
GUIImageViewer::closeEvent(QCloseEvent *event)
{
  if (m_colortab) m_colortab->close(); 

  QWidget::closeEvent(event);
  stringstream ss; ss << "closeEvent(...): type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------
void
GUIImageViewer::moveEvent(QMoveEvent *event)
{
  int x = event->pos().x();
  int y = event->pos().y();
  QString text = QString::number(x) + "," + QString::number(y);
  setWindowTitle(text);
}

//--------------------------

void 
GUIImageViewer::mousePressEvent(QMouseEvent *event)
{
  int x = event->pos().x();
  int y = event->pos().y();
  QString text = "mousePressEvent: " + QString::number(x) + "," + QString::number(y);
  //std::cout << text.toStdString()  << std::endl;
  setWindowTitle(text);
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
GUIImageViewer::onButRHis()
{
  MsgInLog(_name_(), INFO, "onButRHis");
}

//--------------------------

void 
GUIImageViewer::onButSpec()
{
  MsgInLog(_name_(), INFO, "onButSpec");
}

//--------------------------

void 
GUIImageViewer::onButAdd()
{
  MsgInLog(_name_(), DEBUG, "onButAdd");
  float rad_raw = 100+rand()%100;
  this -> m_image -> addCircle(rad_raw);
}

//--------------------------

void 
GUIImageViewer::onButColorTab()
{
  static unsigned counter=0; 
  if(!counter++)
    m_colortab -> move(this->pos().x() + this->size().width() + 8, this->pos().y());      
  m_colortab->setVisible(! (m_colortab->isVisible()));
  
  stringstream ss; ss << "Color table selection window " << 
		     ((m_colortab->isVisible()) ? "is open" : "closed");
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

} // namespace PSQt

//--------------------------
