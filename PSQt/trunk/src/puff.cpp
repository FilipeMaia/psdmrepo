//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

#include "PSQt/puff.h"
#include <QApplication>
#include <QPainter>
#include <QPainterPath>
#include <QTimer>
#include <iostream>

namespace PSQt {

Puff::Puff(QWidget *parent)
    : QWidget(parent)
{
  x = 1;
  opacity = 1.0;
  timerId = startTimer(15);
}

void Puff::paintEvent(QPaintEvent *event)
{
  QPainter painter(this);

  QString text = "ZetCode";

  painter.setPen(QPen(QBrush("#575555"), 1));

  QFont font("Courier", x, QFont::DemiBold);
  QFontMetrics fm(font);
  int textWidth = fm.width(text);

  painter.setFont(font);


  if (x > 10) {
    opacity -= 0.01;
    painter.setOpacity(opacity);
  }

  if (opacity <= 0) {
    killTimer(timerId);
    std::cout << "timer stopped" << std::endl;
  }

  int h = height();
  int w = width();

  painter.translate(QPoint(w/2, h/2));
  painter.drawText(-textWidth/2, 0, text);

}


void Puff::timerEvent(QTimerEvent *event)
{
  x += 1;
  repaint();
}

} // namespace PSQt
