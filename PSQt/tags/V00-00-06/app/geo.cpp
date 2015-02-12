//
// This application launches the work threads and main GUI 
//
#include <QApplication>
#include "PSQt/GUIMain.h"
#include "PSQt/GUIImageViewer.h"
#include "PSQt/WdgDirTree.h"
#include "PSQt/WdgFile.h"
#include "PSQt/WdgImage.h"
#include "PSQt/WdgColorTable.h"
#include "PSQt/WdgGeoTree.h"
#include "PSQt/WdgGeo.h"
#include "PSQt/Frame.h"
#include "PSQt/ThreadTimer.h"
#include "PSQt/allinone.h"
#include "PSQt/GUILogger.h"

int main( int argc, char **argv )
{
  QApplication app( argc, argv ); // SHOULD BE created before QThread object

  const bool do_print = false;
  PSQt::ThreadTimer* t1 = new PSQt::ThreadTimer(0,  5, do_print);  t1 -> start();
  PSQt::ThreadTimer* t2 = new PSQt::ThreadTimer(0, 60, do_print);  t2 -> start();

  //cout << "argc = " << argc << endl;
  cout << "Command line:";
  for(int i = 0; i < argc; i++) cout << " " << argv[i]; cout << '\n';

  if(argc==1 || atoi(argv[1])==1) { PSQt::GUIMain*  w = new PSQt::GUIMain();        w->show(); }
  else if(atoi(argv[1])==2) { PSQt::MyWidget*       w = new PSQt::MyWidget();       w->show(); }
  else if(atoi(argv[1])==3) { PSQt::WdgFile*        w = new PSQt::WdgFile();        w->show(); }
  else if(atoi(argv[1])==4) { PSQt::WdgImage*       w = new PSQt::WdgImage(0);      w->show(); }
  else if(atoi(argv[1])==5) { PSQt::WdgColorTable*  w = new PSQt::WdgColorTable();  w->show(); }
  else if(atoi(argv[1])==6) { PSQt::GUIImageViewer* w = new PSQt::GUIImageViewer(); w->show(); }
  else if(atoi(argv[1])==7) { PSQt::WdgDirTree*     w = new PSQt::WdgDirTree();     w->show(); }
  else if(atoi(argv[1])==8) { PSQt::WdgGeoTree*     w = new PSQt::WdgGeoTree();     w->show(); }
  else if(atoi(argv[1])==9) { PSQt::WdgGeo*         w = new PSQt::WdgGeo();         w->show(); }
  else if(atoi(argv[1])==10){ PSQt::GUILogger*      w = new PSQt::GUILogger();      w->show(); }
  else {cout << "Input argument: " << argv[1] << " is outside allowed range..."; return 0;}

  return app.exec(); // Begin to display qt4 GUI
}
