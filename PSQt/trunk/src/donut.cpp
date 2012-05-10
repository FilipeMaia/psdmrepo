#include "PSQt/donut.h"
#include <QApplication>
#include <QPainter>

namespace PSQt {

Donut::Donut(QWidget *parent)
    : QWidget(parent)
{
}

void Donut::paintEvent(QPaintEvent *event)
{
  this -> setMinimumHeight(280);  

  //QPalette palette(Qt::white); //palette.setColor(QPalette::Base,Qt::yellow);
  //QPalette palette(QColor(255, 255, 255, 255)); // QColor(R,G,B,Alpha) palette.setColor(QPalette::Base,Qt::yellow);
  setPalette (QPalette(QColor(255, 255, 255, 255))); // Set QPalette for QWidget
  setAutoFillBackground (true); // MUST BE TRUE TO DRAW THE BACKGROUND COLOR SET IN Palette

  QPainter painter(this);

  painter.setPen(QPen(QBrush("#535353"), 0.5));

  painter.setRenderHint(QPainter::Antialiasing);

  int h = height();
  int w = width();

  painter.translate(QPoint(w/2, h/2));

  for (qreal rot=0; rot < 360.0; rot+=5.0 ) {
      painter.drawEllipse(-125, -40, 250, 80);
      painter.rotate(5.0);
  }
}

} // namespace PSQt
