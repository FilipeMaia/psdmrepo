//--------------------------

#include "PSQt/GUIMain.h"

//#include <string>
//#include <fstream>   // ofstream
//#include <iomanip>   // for setw, setfill
//#include <sstream>   // for stringstream
//#include <math.h>
//#include <stdio.h>

#include <iostream>    // cout
#include <fstream>    // ifstream(fname), ofstream
//using namespace std; // for cout without std::

#include "MsgLogger/MsgLogger.h"

namespace PSQt {

//--------------------------

GUIMain::GUIMain( QWidget *parent )
    : QWidget(parent)
{
  const std::string base_dir = "/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/";
  const std::string fname_geo = base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data"; 
  const std::string fname_img = base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt"; 

  this -> setFrame();

  m_but_exit = new QPushButton( "Exit", this );
  m_but_test = new QPushButton( "Test", this );
  m_file     = new PSQt::WdgFile(this);

  m_image    = new PSQt::WdgImage(this, fname_geo, fname_img);
  
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
GUIMain::showTips() 
{
  m_but_exit  -> setToolTip("Exit application");
}

//--------------------------

void
GUIMain::setFrame() 
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  //m_frame -> setFrameStyle ( QFrame::Box );    // NoFrame, Box, Panel, WinPanel, ..., StyledPanel 
  //m_frame -> setFrameShadow( QFrame::Sunken ); // Plain, Sunken, Raised 
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  m_frame -> setCursor(Qt::SizeAllCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  //m_frame -> setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
}

//--------------------------
//--------------------------

void 
GUIMain::resizeEvent(QResizeEvent *event)
{
//m_frame->setGeometry(this->rect());
  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  setWindowTitle("Window is resized");
}

//--------------------------

void 
GUIMain::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  //std::cout << "GUIMain::closeEvent(...): type = " << event -> type() << std::endl;
  MsgLog("GUIMain", info, "closeEvent(...): type = " << event -> type());
}

//--------------------------
void
GUIMain::moveEvent(QMoveEvent *event)
{
  int x = event->pos().x();
  int y = event->pos().y();
  QString text = QString::number(x) + "," + QString::number(y);
  setWindowTitle(text);
}

//--------------------------

void 
GUIMain::mousePressEvent(QMouseEvent *event)
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
GUIMain::onButExit()
{
  std::cout << "GUIMain::onButExit()\n";
  this->close(); // will call closeEvent(...)
}

//--------------------------

} // namespace PSQt

//--------------------------
