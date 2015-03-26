//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------


#include "PSQt/center.h"
#include <QDesktopWidget>
#include <QApplication>

namespace PSQt {

void center(QWidget &widget)
{
  int x, y;
  int screenWidth;
  int screenHeight;

  int WIDTH = 350;
  int HEIGHT = 280;

  QDesktopWidget *desktop = QApplication::desktop();

  screenWidth = desktop->width();
  screenHeight = desktop->height();

  x = (screenWidth - WIDTH) / 10;
  y = (screenHeight - HEIGHT) / 10;

  widget.setGeometry(x, y, WIDTH, HEIGHT);
}

} // namespace PSQt
