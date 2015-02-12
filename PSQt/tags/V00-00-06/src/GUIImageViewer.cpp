//--------------------------

#include "PSQt/GUIImageViewer.h"
#include "PSQt/Logger.h"

//#include <string>
//#include <fstream>   // ofstream
//#include <iomanip>   // for setw, setfill
#include <sstream>   // for stringstream
//#include <math.h>
//#include <stdio.h>

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

  m_but_exit = new QPushButton( "Exit", this );
  m_but_test = new QPushButton( "Test", this );
  m_file     = new PSQt::WdgFile(this);

  m_image    = new PSQt::WdgImage(this);
  
  connect(m_but_exit, SIGNAL( clicked() ), this, SLOT(onButExit()) );
  connect(m_but_test, SIGNAL( clicked() ), m_image, SLOT(onTest()) );
  connect(m_file, SIGNAL(fileNameIsChanged(const std::string&)), m_image, SLOT(onFileNameChanged(const std::string&)) ); 

  QHBoxLayout *hbox = new QHBoxLayout();
  hbox -> addWidget(m_but_exit);
  hbox -> addWidget(m_but_test);
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
  m_but_exit  -> setToolTip("Exit application");
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
  //int x = event->pos().x();
  //int y = event->pos().y();
  //QString text = "mousePressEvent: " + QString::number(x) + "," + QString::number(y);
  //std::cout << text.toStdString()  << std::endl;
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

} // namespace PSQt

//--------------------------
