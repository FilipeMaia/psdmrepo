//
// This application launches the work threads and main GUI 
//
#include <QApplication>
#include "PSQt/GUIMain.h"
#include "PSQt/WdgFile.h"
#include "PSQt/WdgImage.h"
#include "PSQt/ThreadTimer.h"
#include "PSQt/allinone.h"

int main( int argc, char **argv )
{
  QApplication app( argc, argv ); // SHOULD BE created before QThread object
  
  PSQt::ThreadTimer* t1 = new PSQt::ThreadTimer(0, 1);      t1 -> start();
  PSQt::ThreadTimer* t2 = new PSQt::ThreadTimer(0, 60, 1);  t2 -> start();

  PSQt::GUIMain*     w1 = new PSQt::GUIMain();
  //PSQt::MyWidget*  w2 = new PSQt::MyWidget();
  //PSQt::WdgFile*   w3 = new PSQt::WdgFile();
  //PSQt::WdgImage*  w4 = new PSQt::WdgImage();

  w1->show();
  //w2->show();
  //w3->show();
  //w4->show();

  return app.exec(); // Begin to display qt4 GUI
}
