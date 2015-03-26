//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------


#include "PSQt/set_window.h"
#include <QDesktopWidget>
#include <QApplication>
#include <iostream>

namespace PSQt {

void set_window(QWidget &widget, int x, int y)
{
  QDesktopWidget *desktop = QApplication::desktop();

  int WIDTH = 350;
  int HEIGHT = 280;
  int screenWidth  = desktop->width();
  int screenHeight = desktop->height();

  std::cout << "screen Width and Height:" << screenWidth << " " << screenHeight << std::endl;

  widget.setGeometry(x, y, WIDTH, HEIGHT);
}

} // namespace PSQt
