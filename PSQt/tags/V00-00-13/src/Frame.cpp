//--------------------------

#include "PSQt/Frame.h"
//#include "PSQt/Logger.h"

//#include <iostream>    // cout
//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

Frame::Frame( QWidget *parent, Qt::WindowFlags flags)
  : QFrame(parent, flags)
{
  setFrame();
  //showTips();
}

//--------------------------

void
Frame::showTips() 
{
  setToolTip("This is a Frame object");
}

//--------------------------

void
Frame::setFrame() 
{
  setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  //setFrameStyle ( QFrame::Box );    // NoFrame, Box, Panel, WinPanel, ..., StyledPanel 
  //setFrameShadow( QFrame::Sunken ); // Plain, Sunken, Raised 
  setLineWidth(0);
  setMidLineWidth(1);
  setCursor(Qt::SizeAllCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  //setStyleSheet("color: rgb(255, 255, 100)");
  //setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
  //setVisible(false);
}
//--------------------------

void
Frame::setBoarderVisible(const bool isVisible) 
{
  if(isVisible) QFrame::setFrameShape(QFrame::Box);
  else          QFrame::setFrameShape(QFrame::NoFrame);
}

//--------------------------

//void 
//Frame::resizeEvent(QResizeEvent *event)
//{
//  setGeometry(0, 0, event->size().width(), event->size().height());
//  setWindowTitle("Window is resized");
//}

//--------------------------

//void 
//Frame::closeEvent(QCloseEvent *event)
//{
//  QFrame::closeEvent(event);
//  stringstream ss; ss << "closeEvent(...): type = " << event -> type();
//  MsgInLog(_name_(), INFO, ss.str());
//}

//--------------------------
//void
//Frame::moveEvent(QMoveEvent *event)
//{
//  int x = event->pos().x();
//  int y = event->pos().y();
//  QString text = QString::number(x) + "," + QString::number(y);
//  setWindowTitle(text);
//}

//--------------------------

//void 
//Frame::mousePressEvent(QMouseEvent *event)
//{
//  int x = event->pos().x();
//  int y = event->pos().y();
//  QString text = "mousePressEvent: " + QString::number(x) + "," + QString::number(y);
//  std::cout << text.toStdString()  << std::endl;
//}

//--------------------------

} // namespace PSQt

//--------------------------
