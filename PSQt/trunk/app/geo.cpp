//
// This application launches the work threads and main GUI 
//
#include <QApplication>
#include "PSQt/GUIMain.h"
#include "PSQt/WdgFile.h"
#include "PSQt/WdgImage.h"
#include "PSQt/WdgColorTable.h"
#include "PSQt/ThreadTimer.h"
#include "PSQt/allinone.h"

int main( int argc, char **argv )
{
  QApplication app( argc, argv ); // SHOULD BE created before QThread object
  
  PSQt::ThreadTimer* t1 = new PSQt::ThreadTimer(0, 1);      t1 -> start();
  PSQt::ThreadTimer* t2 = new PSQt::ThreadTimer(0, 60, 1);  t2 -> start();

  cout << "argc = " << argc << endl;
  for(int i = 0; i < argc; i++)
      cout << "argv[" << i << "] = " << argv[i] << endl;

  const std::string base_dir = "/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/";
  const std::string fname_geo = base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data"; 
  const std::string fname_img = base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt"; 

  //PSQt::GUIMain*       w1 = new PSQt::GUIMain();
  //PSQt::MyWidget*      w2 = new PSQt::MyWidget();      
  //PSQt::WdgFile*       w3 = new PSQt::WdgFile();       
  //PSQt::WdgImage*      w4 = new PSQt::WdgImage(0, fname_geo, fname_img);      
  //PSQt::WdgColorTable* w5 = new PSQt::WdgColorTable();                   

  if(argc==1 || atoi(argv[1])==1) { PSQt::GUIMain* w1 = new PSQt::GUIMain();                          w1->show(); }
  else if(atoi(argv[1])==2) { PSQt::MyWidget*      w2 = new PSQt::MyWidget();                         w2->show(); }
  else if(atoi(argv[1])==3) { PSQt::WdgFile*       w3 = new PSQt::WdgFile();                          w3->show(); }
  else if(atoi(argv[1])==4) { PSQt::WdgImage*      w4 = new PSQt::WdgImage(0, fname_geo, fname_img);  w4->show(); }
  else if(atoi(argv[1])==5) { PSQt::WdgColorTable* w5 = new PSQt::WdgColorTable();                    w5->show(); }

  return app.exec(); // Begin to display qt4 GUI
}
