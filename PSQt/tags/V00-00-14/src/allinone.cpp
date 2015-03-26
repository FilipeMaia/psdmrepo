//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------



#include <iostream>  // for std::cout
//using namespace std; // for      cout without std::
//#include <stdio.h> // for printf

#include "PSQt/allinone.h"
#include "PSQt/custombutton.h"
#include "PSQt/lines.h"
#include "PSQt/donut.h"


namespace PSQt {

//--------------------------

MyWidget::MyWidget( QWidget *parent )
{
  //setupUi(this); // this sets up GUI

  this -> setFrame();

  QPalette* palette1 = new QPalette(); palette1->setColor(QPalette::Base,Qt::darkCyan);
  QPalette* palette2 = new QPalette(); palette2->setColor(QPalette::Base,Qt::yellow);
  QPalette* palette3 = new QPalette(); palette3->setColor(QPalette::Base,Qt::red);
	    
  QLabel*      label     = new QLabel("Test of QLabel");
             m_line_edit = new QLineEdit("Test of QLineEdit");
  QTextEdit*   text_edit = new QTextEdit("Test of QTextEdit");
  QPushButton* button    = new QPushButton( "Test of QPushButton", this );
  QLCDNumber*  lcd       = new QLCDNumber( this );
  QSlider*     slider    = new QSlider( Qt::Horizontal, this );

  CustomButton* custom   =  new CustomButton(this);
  custom->setMinimumHeight(40);

  QRadioButton* radio1     = new QRadioButton("Test of QRadioButton 1");
  QRadioButton* radio2     = new QRadioButton("Test of QRadioButton 2");
  QButtonGroup* radioGroup = new QButtonGroup();

  QComboBox* comboBox = new QComboBox();
  comboBox->addItem(tr("Solid"), Qt::SolidLine);
  comboBox->addItem(tr("Dash"),  Qt::DashLine);
  comboBox->addItem(tr("Dot"),   Qt::DotLine);
  comboBox->addItem(tr("None"),  Qt::NoPen);

  QCheckBox* checkBox = new QCheckBox(tr("QCh&eckBox"));
  checkBox->setChecked(true);

  QSpinBox* spinBox = new QSpinBox();
  spinBox->setRange(0, 20);
  spinBox->setSpecialValueText(tr("0 (cosmetic pen)"));

  radioGroup->addButton(radio1);
  radioGroup->addButton(radio2);
  radio1->setChecked(true);

  text_edit->setFixedSize(500,50);
  text_edit->setPalette(*palette1);
  m_line_edit->setPalette(*palette2);
  //button   ->setPalette(*palette3); // does not work
  button   ->setStyleSheet("background-color: rgb(255, 0, 0); color: rgb(255, 255, 255)");
  text_edit->setStyleSheet("background-color: rgb(0, 255, 0); color: rgb(100, 100, 100)");

  PSQt::Lines* lines = new PSQt::Lines();
  lines->setMinimumHeight(140);
  //lines->setMinimumWidth(350);

  PSQt::Donut* donut = new PSQt::Donut();
  donut->setMinimumHeight(240);

  connect( slider,      SIGNAL( valueChanged(int)), lcd, SLOT(display(int)) );
  connect( button,      SIGNAL( clicked() ), this, SLOT( onButton()) ); 
  connect( text_edit,   SIGNAL( textChanged() ), this, SLOT( onTextEdit()) ); 
  connect( m_line_edit, SIGNAL( editingFinished() ), this, SLOT( onLineEdit()) ); 
  connect( radio1,      SIGNAL( clicked() ), this, SLOT( onRadio1()) ); 
  connect( radio2,      SIGNAL( clicked() ), this, SLOT( onRadio2()) ); 
  connect( comboBox,    SIGNAL( activated(int)), this, SLOT(onComboBox()));
  connect( checkBox,    SIGNAL( toggled(bool)), this, SLOT(onCheckBox()));
  connect( spinBox,     SIGNAL( valueChanged(int)), this, SLOT(onSpinBox()));
  connect( custom,      SIGNAL( clicked() ), this, SLOT( onCustomButton()) ); 
 
  QVBoxLayout *vbox = new QVBoxLayout();

  vbox -> addWidget(label);
  vbox -> addWidget(button);
  vbox -> addWidget(m_line_edit);
  vbox -> addWidget(text_edit);
  vbox -> addWidget(radio1);
  vbox -> addWidget(radio2);
  vbox -> addWidget(lcd);
  vbox -> addWidget(slider);
  vbox -> addWidget(comboBox);
  vbox -> addWidget(checkBox);
  vbox -> addWidget(spinBox);
  vbox -> addWidget(custom);
  vbox -> addWidget(lines);
  vbox -> addWidget(donut);

  this -> setLayout(vbox);

  setWindowTitle(tr("Basic Drawing"));
  this -> move(100,50); // open qt window in specified position
}

//--------------------------

void MyWidget::setFrame() 
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  //m_frame -> setFrameStyle ( QFrame::Box );    // NoFrame, Box, Panel, WinPanel, ..., StyledPanel 
  //m_frame -> setFrameShadow( QFrame::Sunken ); // Plain, Sunken, Raised 
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  m_frame -> setCursor(Qt::SizeAllCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  m_frame -> setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
}

//--------------------------

void MyWidget::resizeEvent(QResizeEvent *event)
{
//m_frame->setGeometry(this->rect());
  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  setWindowTitle("Window is resized");
}

//--------------------------

void MyWidget::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  std::cout << "MyWidget::closeEvent type = " << event -> type() << std::endl;
}

//--------------------------
// window move event
void MyWidget::moveEvent(QMoveEvent *event)
{
  int x = event->pos().x();
  int y = event->pos().y();
  QString text = QString::number(x) + "," + QString::number(y);
  setWindowTitle(text);
}

//--------------------------

void MyWidget::mousePressEvent (QMouseEvent *event)
{
  int x = event->pos().x();
  int y = event->pos().y();
  QString text = "mousePressEvent: " + QString::number(x) + "," + QString::number(y);
  std::cout << text.toStdString()  << std::endl;
}


//--------------------------
//--------------------------
//--------------------------
//--------------------------

void MyWidget::onButton()
{
  std::cout << "Button is clicked\n";
}

//--------------------------

void MyWidget::onCustomButton()
{
  std::cout << "Custom button is clicked\n";
}

//--------------------------

void MyWidget::onLineWindow()
{
  std::cout << "Line window is clicked\n";
}

//--------------------------

void MyWidget::onDonutWindow()
{
  std::cout << "Donut window is clicked\n";
}

//--------------------------

void MyWidget::onRadio1()
{
  std::cout << "Radio1 is clicked\n";
}

//--------------------------

void MyWidget::onRadio2()
{
  std::cout << "Radio2 is clicked\n";
}

//--------------------------

void MyWidget::onComboBox()
{
  std::cout << "Combo box is changed\n";
}

//--------------------------

void MyWidget::onSpinBox()
{
  std::cout << "Spin box is changed\n";
}

//--------------------------

void MyWidget::onCheckBox()
{
  std::cout << "Check box is changed\n";
}

//--------------------------

void MyWidget::onTextEdit()
{
  std::cout << "Text is changed\n";
}

//--------------------------

void MyWidget::onLineEdit()
{
  QString qstring = m_line_edit -> text();
  std::cout << "Line is changed to:" << qstring.toStdString() << std::endl;
}

//--------------------------

} // namespace PSQt

//--------------------------
