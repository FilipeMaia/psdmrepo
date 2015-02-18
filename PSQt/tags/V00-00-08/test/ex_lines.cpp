
// http://zetcode.com/tutorials/qt4tutorial/painting/

#include "PSQt/lines.h"
#include "PSQt/center.h"
#include <QDesktopWidget>
#include <QApplication>

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);  

  PSQt::Lines window;

  window.setWindowTitle("Lines");
  window.show();
  PSQt::center(window);

  return app.exec();
}
