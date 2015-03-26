//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

#include "PSQt/text.h"
#include <QApplication>
#include <QPainter>

namespace PSQt {

Text::Text(QWidget *parent)
    : QWidget(parent)
{
}

void Text::paintEvent(QPaintEvent *event)
{
     QPainter painter(this);
     painter.setPen(Qt::blue);
     painter.setFont(QFont("Arial", 30));
     painter.drawText(rect(), Qt::AlignCenter, "Qt");
}

} // namespace PSQt
