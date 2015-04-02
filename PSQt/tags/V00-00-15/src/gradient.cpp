//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------


#include "PSQt/gradient.h"
#include <QApplication>
#include <QPainter>

namespace PSQt {

Gradient::Gradient(QWidget *parent)
    : QWidget(parent)
{
}

void Gradient::paintEvent(QPaintEvent *event)
{
  QPainter painter(this);

  QLinearGradient grad1(0, 20, 0, 110);
  grad1.setColorAt(0.1, Qt::black);
  grad1.setColorAt(0.5, Qt::yellow);
  grad1.setColorAt(0.9, Qt::black);
  painter.fillRect(20, 20, 300, 90, grad1);

  QLinearGradient grad2(0, 55, 250, 0);
  grad2.setColorAt(0.2, Qt::black);
  grad2.setColorAt(0.5, Qt::red);
  grad2.setColorAt(0.8, Qt::black);
  painter.fillRect(20, 140, 300, 100, grad2);
}

} // namespace PSQt
