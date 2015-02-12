
#include "PSQt/custombutton.h"
#include <QApplication>
#include <QPainter>

namespace PSQt {

CustomButton::CustomButton(QWidget *parent)
    : QPushButton(parent)
{  
}

CustomButton::~CustomButton()
{
}

//Paint event of button
void CustomButton::paintEvent(QPaintEvent *paint)
{
    QPushButton::paintEvent(paint);
    QPainter p(this);
    p.save();

    p.setPen(Qt::blue);                  
    p.setFont(QFont("Arial", 20));       
    p.drawText(QPoint( 10, 22),"Custom");
    p.drawText(QPoint(110, 22),"Button");
    p.drawText(QPoint(210, 22),"Test"); 

    QPen pen(Qt::red, 4, Qt::SolidLine); // Qt::DashLine, Qt::DotLine, Qt::DashDotLine, Qt::DashDotDotLine
    p.setPen(pen);                  
    p.drawLine(260, 5, 400, 15);

    //p.drawImage(QPoint(300,300),SimpleIcon);
    p.restore();
}

} // namespace PSQt
