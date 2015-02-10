//--------------------------

#include "PSQt/GUILogger.h"
#include "PSQt/Logger.h"

#include <iostream>  // std::cout
#include <sstream>   // std::stringstream

#include <QIntValidator>

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

GUILogger::GUILogger( QWidget *parent, const bool& showbuts)
  : Frame(parent) //  , Logger()
  , m_showbuts(showbuts)
{
  m_txt_edi  = new QTextEdit();
  m_but_save = new QPushButton("Save");
  m_combo    = new QComboBox(this);

  m_list << strLevel(DEBUG).c_str()
         << strLevel(INFO).c_str()
         << strLevel(WARNING).c_str()
         << strLevel(ERROR).c_str()
         << strLevel(CRITICAL).c_str();

  m_combo -> addItems(m_list);
  m_combo -> setCurrentIndex(1); 

  m_cbox = new QHBoxLayout;
  m_cbox -> addStretch(1);
  m_cbox -> addWidget(m_combo);
  m_cbox -> addWidget(m_but_save);

  m_vbox = new QVBoxLayout;
  m_vbox -> addWidget(m_txt_edi);
  m_vbox -> addLayout(m_cbox);

  this -> setLayout(m_vbox);

  this -> showTips();
  this -> setStyle();
  this -> addStartRecords();

  //connect(m_rad_x0,  SIGNAL( clicked() ), this, SLOT( onRadioX()) );
  //connect(m_but_add, SIGNAL( clicked() ), this, SLOT(onButAddSub()) );
  connect(m_combo, SIGNAL(currentIndexChanged(int)), this, SLOT(onCombo(int)) );
  connect(m_but_save, SIGNAL(clicked()), this, SLOT(onSave()) );

  //this -> move(300,50);
}

//--------------------------

void
GUILogger::showTips() 
{
  //m_rad_x0     -> setToolTip((std::string("x")+orig).c_str());
}

//--------------------------

void
GUILogger::setStyle() 
{
  setGeometry(100, 100, 500, 300);

  m_but_save->setVisible(m_showbuts);
  m_combo   ->setVisible(m_showbuts);

  m_txt_edi -> setContentsMargins(-9,-9,-9,-9); 
  this -> setContentsMargins(-9,-9,-9,-9);

  //this -> setWindowTitle(tr("GUILogger"));
  //this -> setMinimumWidth(700);
  //this -> setFixedHeight(50);
  //this -> setFixedHeight( (m_show_frame)? 50 : 34);

  //if (! m_show_frame) this -> setContentsMargins(-9,-9,-9,-9);

  //this -> setContentsMargins(-9,-9,-9,-9);
  //this -> setContentsMargins(-5,-5,-5,-5);

  //this -> setFixedWidth(220);
  //m_edi_x0    -> setFixedWidth(width);
  //m_edi_x0    -> setReadOnly(true);
}

//--------------------------

void 
GUILogger::resizeEvent(QResizeEvent *event)
{
  stringstream ss; ss << "w:" << event->size().width() << " h:" <<  event->size().height();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void
GUILogger::moveEvent(QMoveEvent *event)
{
  stringstream ss; ss << "x:" << event->pos().x() << " y:" << event->pos().y();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
GUILogger::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  MsgInLog(_name_(), INFO, "GUILogger::closeEvent(...)"); 
}

//--------------------------

void  
GUILogger::onCombo(int i)
{
  std::cout << " selected: " << m_list.at(i).toStdString() << '\n'; 
}

//--------------------------

void  
GUILogger::onSave()
{
  std::cout << " onSave\n"; 
  SaveLog();
}

//--------------------------
void 
GUILogger::addNewRecord(Record& rec)
{
  //std::cout << _name_() << "=================  >>>> ::addNewRecord";
  //std::cout << rec.strRecordTotal() << '\n';  
  m_txt_edi->append(rec.strRecordTotal().c_str());

  // scroll down
  m_txt_edi->moveCursor(QTextCursor::End);
  m_txt_edi->repaint();
}


//--------------------------
void 
GUILogger::addStartRecords()
{
  std::string start_recs =  
  Logger::getLogger()->strRecordsForLevel(INFO);
  m_txt_edi->append(start_recs.c_str());

  // scroll down
  m_txt_edi->moveCursor(QTextCursor::End);
  m_txt_edi->repaint();
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
