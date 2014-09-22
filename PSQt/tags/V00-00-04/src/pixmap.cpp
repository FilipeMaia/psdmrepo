#include "PSQt/pixmap.h"
#include <QApplication>
#include <QPainter>

#include <QtCore>

namespace PSQt {

Pixmap::Pixmap(QWidget *parent)
    : QWidget(parent)
{
}

void Pixmap::paintEvent(QPaintEvent *event)
{
   QRectF  target(10.0, 20.0, 180.0, 160.0);
   QRectF  source(0.0, 0.0, 170.0, 140.0);
   //QPixmap image(":myPixmap.png");
   //QPixmap image( QSize(200,100) );
   QPixmap image( 200,100 );
   //image.fill( QColor("#d4d4d4") );
   image.fill( Qt::blue );
   QPainter painter(this);
   painter.drawPixmap(target, image, source);
}

} // namespace PSQt
