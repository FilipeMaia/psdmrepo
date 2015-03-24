//--------------------------

#include "PSQt/GUILogger.h"
#include "PSQt/Logger.h"

#include <iostream>  // std::cout
#include <sstream>   // std::stringstream

#include <QIntValidator>

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

GUILogger::GUILogger( QWidget *parent, const bool& showbuts, const bool& showframe)
  : Frame(parent) //  , Logger()
  , m_showbuts(showbuts)
  , m_showframe(showframe)
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

  connect(Logger::getLogger(), SIGNAL(signal_new_record(Record&)), this, SLOT(addNewRecord(Record&)));
  connect(m_combo, SIGNAL(currentIndexChanged(int)), this, SLOT(onCombo(int)) );
  connect(m_but_save, SIGNAL(clicked()), this, SLOT(onSave()) );
}

//--------------------------
void
GUILogger::showTips() 
{
  m_combo->setToolTip("Select the level of messages to show");
  m_txt_edi->setToolTip("Log messages");
  m_but_save->setToolTip("Save log in file");
}

//--------------------------
void
GUILogger::setStyle() 
{
  setGeometry(100, 100, 500, 300);

  m_but_save->setVisible(m_showbuts);
  m_combo   ->setVisible(m_showbuts);

  Frame::setBoarderVisible(m_showframe);
  if (! m_showframe) this -> setContentsMargins(-9,-9,-9,-9);

  //this -> setWindowTitle(tr("GUILogger"));
  //this -> setMinimumWidth(700);
  //this -> setFixedWidth(220);
  //this -> setFixedHeight(50);
  //this -> setFixedHeight( (m_show_frame)? 50 : 34);

  //this -> move(300,50);
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
  stringstream ss; ss << "closeEvent(...): type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------
void  
GUILogger::onCombo(int i)
{
  std::string str_level = m_list.at(i).toStdString();
  MsgInLog(_name_(), INFO, "Selected level: " + str_level);

  LEVEL level = levelFromString(str_level);
  std::string recs = Logger::getLogger()->strRecordsForLevel(level);
  m_txt_edi->clear();
  m_txt_edi->append(recs.c_str());

  SetMsgLevel(level);

  scrollDown();
}

//--------------------------
void  
GUILogger::onSave()
{
  MsgInLog(_name_(), INFO, "\"Save\" button is clicked");
  SaveLog();
}

//--------------------------
void 
GUILogger::scrollDown()
{
  m_txt_edi->moveCursor(QTextCursor::End);
  m_txt_edi->repaint();
}

//--------------------------
void 
GUILogger::addNewRecord(Record& rec)
{
  //std::cout << rec.strRecordTotal() << '\n';  
  //m_txt_edi->append(rec.strRecordTotal().c_str());
  m_txt_edi->append(rec.strRecord().c_str());

  scrollDown();
}


//--------------------------
void 
GUILogger::addStartRecords()
{
  std::string start_recs = Logger::getLogger()->strRecordsForLevel(Logger::getLogger()->getLevel());
  m_txt_edi->append(start_recs.c_str());

  //cout << _name_() << "addStartRecords():\n" << start_recs << '\n';

  scrollDown();
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
