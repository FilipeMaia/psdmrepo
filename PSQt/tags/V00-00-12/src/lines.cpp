#include "PSQt/lines.h"
#include <QApplication>
#include <QPainter>

namespace PSQt {

Lines::Lines(QWidget *parent)
    : QWidget(parent)
{
}

void Lines::paintEvent(QPaintEvent *event)
{
  this -> setFrame();

  //QPalette palette(Qt::white); //palette.setColor(QPalette::Base,Qt::yellow);
  //QPalette palette(QColor(255, 255, 255, 255)); // QColor(R,G,B,Alpha) palette.setColor(QPalette::Base,Qt::yellow);
  //QPalette palette("#FFAAFF");
  setPalette ( QPalette(QColor(255, 255, 255, 255)) );
  setAutoFillBackground (true); // MUST BE TRUE TO DRAW THE BACKGROUND COLOR SET IN Palette

  QPen pen(Qt::black, 2, Qt::SolidLine);
  QPainter painter(this);

  painter.setPen(pen);
  painter.drawLine(20, 20, 250, 20);

  pen.setStyle(Qt::DashLine);
  painter.setPen(pen);
  painter.drawLine(20, 40, 250, 40);

  pen.setStyle(Qt::DashDotLine);
  painter.setPen(pen);
  painter.drawLine(20, 60, 250, 60);

  pen.setStyle(Qt::DotLine);
  painter.setPen(pen);
  painter.drawLine(20, 80, 250, 80);

  pen.setStyle(Qt::DashDotDotLine);
  painter.setPen(pen);
  painter.drawLine(20, 100, 250, 100);


  QVector<qreal> dashes;
  qreal space = 4;

  dashes << 1 << space << 5 << space;

  pen.setStyle(Qt::CustomDashLine);
  pen.setDashPattern(dashes);
  painter.setPen(pen);
  painter.drawLine(20, 120, 250, 120);
}

//--------------------------

void Lines::setFrame()
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  //m_frame -> setFrameStyle ( QFrame::Box );    // NoFrame, Box, Panel, WinPanel, ..., StyledPanel 
  //m_frame -> setFrameShadow( QFrame::Sunken ); // Plain, Sunken, Raised 
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  m_frame -> setCursor(Qt::SizeAllCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  m_frame -> setGeometry(this->rect());
  m_frame -> setVisible(true);
}

//--------------------------
} // namespace PSQt
