//===================
// This application launches the work threads and main GUI 
//
#include <getopt.h>
#include <stdio.h>
#include <iostream>

#include <QApplication>
#include "PSQt/GUIMain.h"
#include "PSQt/GUIImageViewer.h"
#include "PSQt/WdgDirTree.h"
#include "PSQt/WdgFile.h"
#include "PSQt/WdgImage.h"
#include "PSQt/WdgColorTable.h"
#include "PSQt/WdgGeoTree.h"
#include "PSQt/WdgGeo.h"
#include "PSQt/WdgPointPos.h"
#include "PSQt/Frame.h"
#include "PSQt/ThreadTimer.h"
#include "PSQt/allinone.h"
#include "PSQt/Logger.h" // for DEBUG, INFO, etc.
#include "PSQt/GUILogger.h"

//===================

void usage(char* name) {
  std::cout << "Usage: " << name << " [-g <geo-fname>] [-i <image-array-fname>] [-L <logger-level>] [-h] [<test-number>] \n";
}

//===================
int main( int argc, char **argv )
{
  //cout << "argc = " << argc << endl;
  cout << "Command line:";
  for(int i = 0; i < argc; i++) cout << " " << argv[i]; cout << '\n';

  // ------------------
  char *ivalue = NULL;
  char *gvalue = NULL;
  char *Lvalue = NULL;
  int c;
  extern char *optarg;
  extern int optind, optopt; //, opterr;

  while ((c = getopt(argc, argv, ":g:i:L:h")) != -1)
      switch (c)
      {
        case 'g':
          gvalue = optarg;
          printf ("-g: gvalue = %s\n",gvalue);
          break;
        case 'i':
          ivalue = optarg;
          printf ("-i: ivalue = %s\n",ivalue);
          break;
        case 'L':
          Lvalue = optarg;
          printf ("-L: Lvalue = %s\n",Lvalue);
          break;
        case 'h':
          //printf ("-h: ");
          usage(argv[0]);
          return EXIT_SUCCESS;
        case ':':
          printf ("(:) Option -%c requires an argument.\n", optopt);
          usage(argv[0]);
          return EXIT_FAILURE;
        case '?':
          printf ("(?): Option -%c is not recognized.\n", optopt);
          usage(argv[0]);
          return EXIT_FAILURE;
        default:
          printf ("default: You should not reach this point... Option \"-%c\" is not recognized.\n", optopt);
          abort ();
      }
 
  //printf ("End of options: gvalue = %s, ivalue = %s, Lvalue = %s\n",
  //        gvalue, ivalue, Lvalue);
 
  for (int index = optind; index < argc; index++)
      printf ("Non-option argument \"%s\"\n", argv[index]);

  //return EXIT_SUCCESS;

  // ------------------

  std::string gfname = (gvalue) ? gvalue : std::string();
  std::string ifname = (ivalue) ? ivalue : std::string();
  PSQt::LEVEL level  = (Lvalue) ? PSQt::levelFromString(std::string(Lvalue)) : PSQt::INFO;

  cout << "\nStart app with input parameters\n" << std::setw(31) << setfill('=') << '='
       << "\n  geometry     : " << gfname
       << "\n  image array  : " << ifname
       << "\n  logger level : " << PSQt::strLevel(level)
       << '\n';

  // ------------------

  QApplication app( argc, argv ); // SHOULD BE created before QThread object

  const bool do_print = false;
  PSQt::ThreadTimer* t1 = new PSQt::ThreadTimer(0,  5, do_print);  t1 -> start();
  PSQt::ThreadTimer* t2 = new PSQt::ThreadTimer(0, 60, do_print);  t2 -> start();

  if(argc==2) {
         if(atoi(argv[1])==1) { PSQt::GUIMain*  w = new PSQt::GUIMain(0,PSQt::INFO);  w->show(); }
    else if(atoi(argv[1])==2) { PSQt::MyWidget*       w = new PSQt::MyWidget();       w->show(); }
    else if(atoi(argv[1])==3) { PSQt::WdgFile*        w = new PSQt::WdgFile();        w->show(); }
    else if(atoi(argv[1])==4) { PSQt::WdgImage*       w = new PSQt::WdgImage(0);      w->show(); }
    else if(atoi(argv[1])==5) { PSQt::WdgColorTable*  w = new PSQt::WdgColorTable();  w->show(); }
    else if(atoi(argv[1])==6) { PSQt::GUIImageViewer* w = new PSQt::GUIImageViewer(); w->show(); }
    else if(atoi(argv[1])==7) { PSQt::WdgDirTree*     w = new PSQt::WdgDirTree();     w->show(); }
    else if(atoi(argv[1])==8) { PSQt::WdgGeoTree*     w = new PSQt::WdgGeoTree();     w->show(); }
    else if(atoi(argv[1])==9) { PSQt::WdgGeo*         w = new PSQt::WdgGeo();         w->show(); }
    else if(atoi(argv[1])==10){ PSQt::GUILogger*      w = new PSQt::GUILogger();      w->show(); }
    else if(atoi(argv[1])==11){ PSQt::WdgPointPos*    w = new PSQt::WdgPointPos();    w->show(); }
    else {cout << "Input argument \"" << argv[1] << "\" is not recognized...\n"; return 0;}
  }
  else {
    PSQt::GUIMain* w = new PSQt::GUIMain(0, level, gfname, ifname);  w->show(); 
  }

  return app.exec(); // Begin to display qt4 GUI
}
