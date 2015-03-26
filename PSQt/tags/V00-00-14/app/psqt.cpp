//
// This application launches the main Qt GUI which starts with psana
//
#include <QApplication>
#include "PSQt/PSQtGUIMain.h"
#include "PSQt/TestThread1.h"
#include "PSQt/allinone.h"

int main( int argc, char **argv )
{
  QApplication app( argc, argv ); // SHOULD BE created before QThread object
  
  PSQt::TestThread1* t1 = new PSQt::TestThread1();
  PSQt::TestThread1* t2 = new PSQt::TestThread1();

  t1 -> start();
  t2 -> start();

  PSQt::PSQtGUIMain* w1 = new PSQt::PSQtGUIMain();
  PSQt::MyWidget*    w2 = new PSQt::MyWidget();

  w1->show();
  w2->show();

  return app.exec(); // Begin to display qt4 GUI
}
