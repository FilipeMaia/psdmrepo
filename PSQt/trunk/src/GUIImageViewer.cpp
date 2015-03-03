//--------------------------

#include "PSQt/GUIImageViewer.h"
#include "PSQt/Logger.h"
#include "AppUtils/AppDataPath.h"

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
{
  //const std::string base_dir = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/";
  //const std::string fname_geo = base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data"; 
  //const std::string fname_img = base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt"; 

  AppUtils::AppDataPath adp_fname_def("PSQt/images/2011-08-10-Tiled-XPP.jpg"); //galaxy.jpeg"); 

  m_but_exit = new QPushButton( "Exit", this );
  m_but_add  = new QPushButton( "Add circle", this );
  m_file     = new PSQt::WdgFile(this, "Image:", adp_fname_def.path());
  m_pointpos = new PSQt::WdgPointPos(this, "Center x:", " y:", 0, 0, false, 60, 2);

  m_but_exit -> setCursor(Qt::PointingHandCursor); 
  m_but_add  -> setCursor(Qt::PointingHandCursor); 

  //m_image = new PSQt::WdgImage(this, m_file->fileName());
  m_image = new PSQt::WdgImageFigs(this, m_file->fileName());
  
  connect(m_but_exit, SIGNAL( clicked() ), this, SLOT(onButExit()) );
  connect(m_but_add,  SIGNAL( clicked() ), this, SLOT(onButAdd()) );
  connect(m_file, SIGNAL(fileNameIsChanged(const std::string&)), m_image, SLOT(onFileNameChanged(const std::string&)) ); 

  const DragBase* p_drag_center = m_image->getDragStore()->getDragCenter();
  connect(p_drag_center, SIGNAL(centerIsMoved(const QPointF&)), m_pointpos, SLOT(setPointPos(const QPointF&))); 
  connect(m_pointpos, SIGNAL(posIsChanged(const QPointF&)), p_drag_center, SLOT(moveToRaw(const QPointF&))); 
  connect(m_pointpos, SIGNAL(posIsChanged(const QPointF&)), m_image, SLOT(forceUpdate())); 
  m_pointpos->setPointPos(m_image->getDragStore()->getCenter());

  QHBoxLayout *hbox = new QHBoxLayout();
  hbox -> addWidget(m_but_exit);
  hbox -> addWidget(m_but_add);
  hbox -> addWidget(m_pointpos);
  hbox -> addStretch(1);

  QVBoxLayout *vbox = new QVBoxLayout();
  vbox -> addWidget(m_file);
  vbox -> addWidget(m_image);
  vbox -> addLayout(hbox);

  this -> setLayout(vbox);
  this -> setWindowTitle(tr("Basic Drawing"));
  this -> move(100,50);  

  showTips();
}

//--------------------------

void
GUIImageViewer::showTips() 
{
  m_but_exit -> setToolTip("Close image viewer window.");
  m_but_add  -> setToolTip("Add circle for current center and random radius.\n"\
                           "Then move circle clicking on it by left mouse button and drag,\n"\
                           "or remove circle clicking on it by middle mouse button.");
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

void 
GUIImageViewer::onButExit()
{
  MsgInLog(_name_(), DEBUG, "onButExit");
  this->close(); // will call closeEvent(...)
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

} // namespace PSQt

//--------------------------
