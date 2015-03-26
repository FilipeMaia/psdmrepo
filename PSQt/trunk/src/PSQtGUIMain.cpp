//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

//--------------------------

#include <iostream>    // for std::cout
#include <fstream>    // for std::ifstream(fname)
//using namespace std; // for cout without std::

#include "PSQt/PSQtGUIMain.h"

namespace PSQt {

//--------------------------

PSQtGUIMain::PSQtGUIMain( QWidget *parent )
{
  this -> setFrame();

  //setPalette ( QPalette(QColor(255, 255, 255, 255)) );
  //setAutoFillBackground (true);  

  //m_lab_fname = new QLabel("Xtc file:");
  m_but_fnxtc = new QPushButton( "xtc file:", this );
  m_but_fncfg = new QPushButton( "cfg file:", this );
  m_but_start = new QPushButton( "Start", this );
  m_but_stop  = new QPushButton( "Stop",  this );
  m_but_save  = new QPushButton( "Save",  this );
  m_but_exit  = new QPushButton( "Exit",  this );
  m_edi_fnxtc = new QLineEdit  ("/reg/d/psdm/CXI/cxi49012/xtc/e158-r0001-s02-c00.xtc");
  m_edi_fncfg = new QLineEdit  ("./psana.cfg");

  connect( m_but_start,      SIGNAL( clicked() ),         this, SLOT( onButStart()) ); 
  connect( m_but_stop,       SIGNAL( clicked() ),         this, SLOT( onButStop()) ); 
  connect( m_but_exit,       SIGNAL( clicked() ),         this, SLOT( onButExit()) ); 
  connect( m_but_save,       SIGNAL( clicked() ),         this, SLOT( onButSave()) ); 
  connect( m_edi_fnxtc,      SIGNAL( editingFinished() ), this, SLOT( onEditXtcFileName()) ); 
  connect( m_but_fnxtc,      SIGNAL( clicked() ),         this, SLOT( onButSelectXtcFile()) ); 
  connect( m_edi_fncfg,      SIGNAL( editingFinished() ), this, SLOT( onEditCfgFileName()) ); 
  connect( m_but_fncfg,      SIGNAL( clicked() ),         this, SLOT( onButSelectCfgFile()) ); 

//connect( m_edi_fname,      SIGNAL( textChanged() ),     this, SLOT( onTextXtcFileName()) ); 
 
  QGridLayout *grid = new QGridLayout();
  //grid -> setRowMinimumHeight(1, 6);
  //grid -> setColumnStretch(0, 2);
  grid -> addWidget(m_but_fnxtc, 0, 0); 
  grid -> addWidget(m_edi_fnxtc, 0, 1, 1, 5); 

  grid -> addWidget(m_but_fncfg, 1, 0); 
  grid -> addWidget(m_edi_fncfg, 1, 1, 1, 5); 

  grid -> addWidget(m_but_start, 2, 0, Qt::AlignRight);
  grid -> addWidget(m_but_stop,  2, 1);
  grid -> addWidget(m_but_save,  2, 2);
  grid -> addWidget(m_but_exit,  2, 5);
 
  this -> setLayout(grid);

  this -> setWindowTitle(tr("PSQtGUIMain"));
  this -> setMinimumHeight(40);
  this -> setMinimumWidth(500);
  this -> move(100,50); // open qt window in specified position

  this -> showTips();
}

//--------------------------

void
PSQtGUIMain::showTips() 
{
  m_but_fnxtc -> setToolTip("Find and select the XTC file");
  m_but_fncfg -> setToolTip("Find and select the psana configuration file");
  m_but_start -> setToolTip("Start psana");
  m_but_stop  -> setToolTip("Stop psana");
  m_but_save  -> setToolTip("Save control parameters");
  m_but_exit  -> setToolTip("Exit program");
}

//--------------------------

void
PSQtGUIMain::setFrame() 
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  //m_frame -> setFrameStyle ( QFrame::Box );    // NoFrame, Box, Panel, WinPanel, ..., StyledPanel 
  //m_frame -> setFrameShadow( QFrame::Sunken ); // Plain, Sunken, Raised 
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  m_frame -> setCursor(Qt::SizeAllCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  //m_frame -> setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
}

//--------------------------

//--------------------------

void 
PSQtGUIMain::resizeEvent(QResizeEvent *event)
{
//m_frame->setGeometry(this->rect());
  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  setWindowTitle("Window is resized");
}

//--------------------------

void 
PSQtGUIMain::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  std::cout << "PSQtGUIMain::closeEvent(...): type = " << event -> type() << std::endl;
}

//--------------------------
//--------------------------
//--------------------------

void 
PSQtGUIMain::onButExit()
{
  std::cout << "PSQtGUIMain::onButExit()\n";
  this->close(); // will call closeEvent(...)
}

//--------------------------

void 
PSQtGUIMain::onButStart()
{
  std::cout << "PSQtGUIMain::onButStart()\n";
}

//--------------------------

void 
PSQtGUIMain::onButStop()
{
  std::cout << "PSQtGUIMain::onButStop()\n";
}

//--------------------------

void 
PSQtGUIMain::onButSave()
{
  std::cout << "PSQtGUIMain::onButSave()\n";
}

//--------------------------

void 
PSQtGUIMain::onEditCfgFileName()
{
  std::cout << "PSQtGUIMain::onEditCfgFileName() : ";
  std::string str_fname = (m_edi_fncfg -> displayText()).toStdString();
  std::cout << "Edited file name: " << str_fname << std::endl;
  //bool inputIsOK = 
  fileExists(str_fname);
}

//--------------------------

void 
PSQtGUIMain::onEditXtcFileName()
{
  std::cout << "PSQtGUIMain::onEditXtcFileName() : ";
  std::string str_fname = (m_edi_fnxtc -> displayText()).toStdString();
  std::cout << "Edited file name: " << str_fname << std::endl;
  //bool inputIsOK = 
  fileExists(str_fname);
}

//--------------------------

void 
PSQtGUIMain::onButSelectCfgFile()
{
  std::cout << "PSQtGUIMain::onButSelectCfgFile()\n";
}

//--------------------------

void 
PSQtGUIMain::onButSelectXtcFile()
{
  std::cout << "PSQtGUIMain::onButSelectXtcFile()\n";
  //QString dirname = "./";
  //QString path_file; // = "/reg/d/psdm/CXI/cxi49012/xtc/e158-r0001-s02-c00.xtc";

  const char* sep = "/";
  std::string str_path_file_edi = (m_edi_fnxtc -> displayText()).toStdString();

  size_t pos = str_path_file_edi.rfind(sep);

  pos = (pos != std::string::npos) ? pos : str_path_file_edi.size();

  std::string dname(str_path_file_edi, 0, pos );
  std::string fname(str_path_file_edi, pos+1 );

  std::cout << "dname: " << dname << std::endl;
  std::cout << "fname: " << fname << std::endl;
  
  dname =  "/reg/d/psdm/CXI/cxi49012/xtc/";

  QString path_file = QFileDialog::getOpenFileName(this, tr("Select input XTC file"), 
                                                   QString(dname.c_str()), tr("Xtc file (*.xtc)"));

  std::string str_path_file = path_file.toStdString();
  std::cout << "Selected file name: " << str_path_file << std::endl;

  if( fileExists(str_path_file) )
     m_edi_fnxtc -> setText(path_file);
}

//--------------------------

bool
PSQtGUIMain::fileExists(std::string fname)
{
  std::ifstream xtc_file(fname.c_str());
  if(xtc_file.good()) {
    std::cout << "Selected file exists" << std::endl;
    return true;
  } 

  std::cout << "WARNING: Selected file DOES NOT exists, try to select other file.";  
  return false;
}

//--------------------------


 
//--------------------------

} // namespace PSQt

//--------------------------
