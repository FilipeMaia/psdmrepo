//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

//--------------------------

#include "PSQt/WdgPointPos.h"
#include "PSQt/Logger.h"

#include <iostream>    // for std::cout
#include <sstream>   // std::stringstream
//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

WdgPointPos::WdgPointPos( QWidget *parent
			, const std::string& label1
			, const std::string& label2 
	                , const float& val1
	                , const float& val2
			, const bool& show_boarder
			, const unsigned& fld_width
			, const unsigned& precision
			  )
//    : QWidget(parent)
  : Frame(parent)
  , m_show_boarder(show_boarder)
  , m_fld_width(fld_width)
  , m_precision(precision)
{
  Frame::setBoarderVisible(show_boarder);

  m_lab1    = new QLabel(label1.c_str());
  m_lab2    = new QLabel(label2.c_str());

  m_edi1 = new QLineEdit("");
  m_edi2 = new QLineEdit("");
  m_edi1 -> setFixedWidth(m_fld_width);
  m_edi2 -> setFixedWidth(m_fld_width);
  m_edi1 -> setValidator(new QDoubleValidator(-100000, 100000, m_precision, this));
  m_edi2 -> setValidator(new QDoubleValidator(-100000, 100000, m_precision, this));
  //m_edi1 -> setCursor(Qt::PointingHandCursor); 
  //m_edi2 -> setCursor(Qt::PointingHandCursor); 

  setPointPos(QPointF(val1,val2));

  connect(m_edi1, SIGNAL(editingFinished()), this, SLOT(onEdi())); 
  connect(m_edi2, SIGNAL(editingFinished()), this, SLOT(onEdi())); 
  connect(this, SIGNAL(posIsChanged(const QPointF&)), this, SLOT(testPosIsChanged(const QPointF&)) ); 
 
  QHBoxLayout *hbox = new QHBoxLayout;
  hbox -> addWidget(m_lab1);
  hbox -> addWidget(m_edi1);
  hbox -> addWidget(m_lab2);
  hbox -> addWidget(m_edi2);
  this -> setLayout(hbox);

  this -> setWindowTitle(tr("WdgPointPos"));
  this -> setFixedHeight( (m_show_boarder)? 50 : 36);
  if (! m_show_boarder) this -> setContentsMargins(-9,-9,-9,-9);
  //if (! m_show_boarder) this -> setContentsMargins(-5,-5,-5,-5);
  //this -> setMinimumWidth(200);

  this -> showTips();
}

//--------------------------

void
WdgPointPos::showTips() 
{
  m_edi1 -> setToolTip("Type value to change");
  m_edi2 -> setToolTip("Type value to change");
}

//--------------------------

void 
WdgPointPos::resizeEvent(QResizeEvent *event)
{
  //setWindowTitle("Window is resized");
}

//--------------------------

void 
WdgPointPos::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  stringstream ss; ss << "closeEvent(...): type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
WdgPointPos::setPointPos(const QPointF& pos)
{
  //m_but_file -> setFixedWidth(but_width);
  stringstream ss1, ss2; 
  ss1 << fixed << std::setprecision(m_precision) << pos.x();
  ss2 << fixed << std::setprecision(m_precision) << pos.y();

  m_edi1->setText(ss1.str().c_str());
  m_edi2->setText(ss2.str().c_str());

  //stringstream ss; ss << "Center position is set to x: " << pos.x() << "  y: " << pos.y(); 
  //MsgInLog(_name_(), DEBUG, ss.str());
  //std::cout << ss.str() << '\n'; 
}

//--------------------------

void 
WdgPointPos::onEdi()
{
  std::string str1 = (m_edi1 -> displayText()).toStdString();
  std::string str2 = (m_edi2 -> displayText()).toStdString();

  float val1, val2;
  stringstream ss; ss << str1 << ' ' << str2; ss >> val1 >> val2;
  QPointF pos(val1,val2);

  emit posIsChanged(pos);
}

//--------------------------

void 
WdgPointPos::testPosIsChanged(const QPointF& pos)
{
  stringstream ss; ss << "Center position is changed to x: " << pos.x() << " y: " << pos.y(); 
  MsgInLog(_name_(), INFO, ss.str());
  //std::cout << ss.str() << '\n'; 
}

//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
