
// http://zetcode.com/tutorials/qt4tutorial/painting/

#include "PSQt/lines.h"
#include "PSQt/brushes.h"
#include "PSQt/colors.h"
#include "PSQt/shapes.h"
#include "PSQt/donut.h"
#include "PSQt/gradient.h"
#include "PSQt/puff.h"
#include "PSQt/text.h"
#include "PSQt/pixmap.h"

#include "PSQt/set_window.h"
#include <QDesktopWidget>
#include <QApplication>

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);  

  PSQt::Lines     window1;
  PSQt::Brushes   window2;
  PSQt::Colors    window3;
  PSQt::Shapes    window4;
  PSQt::Donut     window5;
  PSQt::Gradient  window6;
  PSQt::Puff      window7;
  PSQt::Text      window8;
  PSQt::Pixmap    window9;

  window1.setWindowTitle("Lines");
  window2.setWindowTitle("Brushes");
  window3.setWindowTitle("Colors");
  window4.setWindowTitle("Shapes");
  window5.setWindowTitle("Donut");
  window6.setWindowTitle("Gradient");
  window7.setWindowTitle("Puff");
  window8.setWindowTitle("Text");
  window9.setWindowTitle("Pixmap");

  window1.show();
  window2.show();
  /*
  window3.show();
  window4.show();
  window5.show();
  window6.show();
  window7.show();
  window8.show();
  */
  window9.show();

  PSQt::set_window(window1, 100,  0);
  PSQt::set_window(window2, 200,  0);
  PSQt::set_window(window3, 300,  0);
  PSQt::set_window(window4, 400,  0);
  PSQt::set_window(window5, 500,  0);
  PSQt::set_window(window6, 600,  0);
  PSQt::set_window(window7, 700,  0);
  PSQt::set_window(window8, 800,  0);
  PSQt::set_window(window9, 900,  0);

  return app.exec();
}
