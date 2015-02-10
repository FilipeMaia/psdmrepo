//--------------------------

#include "PSQt/WdgFile.h"

#include <iostream>    // for std::cout
#include <fstream>    // for std::ifstream(fname)
//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

WdgFile::WdgFile( QWidget *parent, 
                  const std::string& but_title, 
                  const std::string& path,
                  const std::string& search_fmt,
                  const bool& show_frame,
                  const unsigned& but_width )
    : QWidget(parent)
    , m_path(path)
    , m_search_fmt(search_fmt)
    , m_show_frame(show_frame)
{
  this -> setFrame();

  //setPalette ( QPalette(QColor(255, 255, 255, 255)) );
  //setAutoFillBackground (true);  

  m_but_file = new QPushButton(but_title.c_str(), this);
  m_edi_file = new QLineEdit  (m_path.c_str());

  m_but_file -> setFixedWidth(but_width);

  connect( m_but_file, SIGNAL( clicked() ),          this, SLOT( onButFile()) ); 
  connect( m_edi_file, SIGNAL( editingFinished() ),  this, SLOT( onEdiFile()) ); 
  //connect( m_edi_file, SIGNAL( textChanged() ),  this, SLOT( onEdiFile()) ); 
  //connect( this, SIGNAL(fileNameIsChanged(const std::string&)), this, SLOT(testSignalString(const std::string&)) ); 
 
  QHBoxLayout *hbox = new QHBoxLayout;
  hbox -> addWidget(m_but_file);
  hbox -> addWidget(m_edi_file);
  this -> setLayout(hbox);

  this -> setWindowTitle(tr("WdgFile"));
  this -> setMinimumWidth(200);
  this -> setFixedHeight( (m_show_frame)? 50 : 34);

  if (! m_show_frame) this -> setContentsMargins(-9,-9,-9,-9);

  //this -> move(300,50); // open qt window in specified position

  this -> showTips();
}

//--------------------------

void
WdgFile::showTips() 
{
  m_but_file -> setToolTip("Click and select the file");
  m_edi_file -> setToolTip("File name field");
}

//--------------------------

void
WdgFile::setFrame() 
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  //m_frame -> setFrameStyle ( QFrame::Box );    // NoFrame, Box, Panel, WinPanel, ..., StyledPanel 
  //m_frame -> setFrameShadow( QFrame::Sunken ); // Plain, Sunken, Raised 
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  m_frame -> setCursor(Qt::SizeAllCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  //m_frame -> setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
  m_frame -> setVisible(m_show_frame);
}

//--------------------------

void 
WdgFile::resizeEvent(QResizeEvent *event)
{
  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  //std::cout << "WdgFile::resizeEvent(...): w=" << event->size().width() 
  //          << "  h=" << event->size().height() << '\n';
  setWindowTitle("Window is resized");
}

//--------------------------

void 
WdgFile::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  std::cout << "WdgFile::closeEvent(...): type = " << event -> type() << '\n';
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
WdgFile::onEdiFile()
{
  std::cout << "WdgFile::onButFile()\n";
  std::string str_fname = (m_edi_file -> displayText()).toStdString();
  std::cout << "Edited file name: " << str_fname << std::endl;

  setNewFileName(str_fname);
}

//--------------------------

void 
WdgFile::onButFile()
{
  std::cout << "WdgFile::onButFile()\n";
  //QString dirname = "./";
  //QString path_file; // = "/reg/d/psdm/CXI/cxi49012/xtc/e158-r0001-s02-c00.xtc";

  const char* sep = "/";
  std::string str_path_file_edi = (m_edi_file -> displayText()).toStdString();

  size_t pos = str_path_file_edi.rfind(sep);

  pos = (pos != std::string::npos) ? pos : str_path_file_edi.size();

  std::string dname(str_path_file_edi, 0, pos );
  std::string fname(str_path_file_edi, pos+1 );

  std::cout << "dname: " << dname << std::endl;
  std::cout << "fname: " << fname << std::endl;
  
  //dname =  "/reg/d/psdm/CXI/cxi49012/xtc/";

  QString path_file = QFileDialog::getOpenFileName(this, tr("Select input file"), 
                                                   QString(dname.c_str()), tr(m_search_fmt.c_str()));

  std::string str_path_file = path_file.toStdString();
  std::cout << "Selected file name: " << str_path_file << std::endl;

  if(setNewFileName(str_path_file)) m_edi_file -> setText(path_file);
}

//--------------------------

bool 
WdgFile::setNewFileName(const std::string& fname)
{
  if(fname == m_path) return false; // if the file name has not been changed
  if( fileExists(fname) ) {
    std::cout << "emit signal fileNameIsChanged(fname)\n";
    emit fileNameIsChanged(fname);
    m_path = fname;
    return true;
  }
  return false;
}

//--------------------------

bool
WdgFile::fileExists(const std::string& fname)
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

void 
WdgFile::testSignalString(const std::string& fname)
{
  std::cout << "WdgFile::testSignalSlot(string): fname = " << fname << '\n';  
}

//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
