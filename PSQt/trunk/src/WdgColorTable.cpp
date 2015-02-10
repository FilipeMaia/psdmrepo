//--------------------------

#include "PSQt/WdgColorTable.h"

#include <QString>
#include <string>

#include <iostream>    // cout
#include <cstring>  // for memcpy, placed in the std namespace

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

WdgColorTable::WdgColorTable( QWidget *parent, const float& h1, const float& h2, const unsigned& colors )
    : QWidget(parent)
    , m_h1(h1)
    , m_h2(h2)
    , m_colors(colors)
{

  m_figsize = 256;
  m_cbar_width = 40;

  this -> setFrame();

  m_edi_h1 = new QLineEdit( val_to_string<float>(m_h1).c_str(), this );
  m_edi_h2 = new QLineEdit( val_to_string<float>(m_h2).c_str(), this );

  m_lab_cring = new PSQt::LabColorRing(this, m_figsize, m_h1, m_h2);

  QChar chars1[] = {0x03B1, '1', ':'};
  QChar chars2[] = {0x03B1, '2', ':'};
  m_lab_h1    = new QLabel(QString(chars1));
  m_lab_h2    = new QLabel(QString(chars2));
  m_lab_cbar  = new QLabel("L2", this);

  connect( m_edi_h1, SIGNAL( editingFinished() ), this, SLOT(onEdiH1()) );
  connect( m_edi_h2, SIGNAL( editingFinished() ), this, SLOT(onEdiH2()) );
  //connect( m_lab_cring, SIGNAL( mouseReleaseEvent(QMouseEvent*) ), this, SLOT(onSetH()) );
  connect( m_lab_cring, SIGNAL( hueAngleIsChanged(const unsigned&) ), 
           this, SLOT(onSetH(const unsigned&)) );
  connect( this, SIGNAL( hueAngleIsEdited(const unsigned&) ), 
           m_lab_cring, SLOT(onSetShifter(const unsigned&)) );
  
  QHBoxLayout *hbox1 = new QHBoxLayout();
  hbox1 -> addWidget(m_lab_h1);
  hbox1 -> addWidget(m_edi_h1);
  hbox1 -> addStretch(1);
  hbox1 -> addWidget(m_lab_h2);
  hbox1 -> addWidget(m_edi_h2);

  QVBoxLayout *vbox = new QVBoxLayout();
  vbox -> addWidget(m_lab_cring);
  vbox -> addLayout(hbox1);
  vbox -> addWidget(m_lab_cbar);
  vbox -> addStretch(1);

  //QHBoxLayout *hbox = new QHBoxLayout();
  //hbox -> addWidget(m_lab_cring);
  //hbox -> addLayout(vbox);

  this -> setLayout(vbox);
  this -> setWindowTitle(tr("Color Table"));
  this -> move(100,50);  

  showTips();
  setStyle();

  m_pixmap_cbar=0;
  setColorBar(m_h1, m_h2, m_cbar_width, m_figsize);

  //this        -> setMouseTracking(true);
  //m_lab_cring -> setMouseTracking(true);
  //m_lab_cring->installEventFilter(this);
  //this->viewport()->setMouseTracking(true);
}

//--------------------------

void
WdgColorTable::showTips() 
{
  m_edi_h1 -> setToolTip("Hue angle 1");
  m_edi_h2 -> setToolTip("Hue angle 2");
}

//--------------------------

void
WdgColorTable::setFrame() 
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  //m_frame -> setFrameStyle ( QFrame::Box );    // NoFrame, Box, Panel, WinPanel, ..., StyledPanel 
  //m_frame -> setFrameShadow( QFrame::Sunken ); // Plain, Sunken, Raised 
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  //m_frame -> setCursor(Qt::SizeAllCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  //m_frame -> setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
}

//--------------------------

void
WdgColorTable::setStyle() 
{
  m_edi_h1   -> setFixedSize(80,30);
  m_edi_h2   -> setFixedSize(80,30);
  m_lab_cbar -> setMargin(0);
  m_lab_cbar -> setFixedSize(m_figsize, m_cbar_width);
  this       -> setFixedWidth(m_figsize+22);

  //m_edi_h1    -> setMinimumSize(80,30);
  //m_edi_h2    -> setMinimumSize(80,30);
  //m_lab_cring -> setFixedSize(100,100);
}

//--------------------------

void 
WdgColorTable::resizeEvent(QResizeEvent *event)
{
//m_frame->setGeometry(this->rect());
  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  //setWindowTitle("Window resized");
  //std::cout << "WdgColorTable::resizeEvent(...): w=" << event->size().width() 
  //                                          << " h=" << event->size().height() << '\n';

  m_lab_cbar ->setPixmap(m_pixmap_cbar ->scaled(m_lab_cbar->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
  //  m_lab_cring->setPixmap(m_pixmap_cring->scaled(m_lab_cring->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
}

//--------------------------

void 
WdgColorTable::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  //std::cout << "WdgColorTable::closeEvent(...): type = " << event -> type() << std::endl;
  //MsgLog("WdgColorTable", info, "closeEvent(...): type = " << event -> type());
}

//--------------------------
void
WdgColorTable::moveEvent(QMoveEvent *event)
{
  //int x = event->pos().x();
  //int y = event->pos().y();
  //QString text = QString::number(x) + "," + QString::number(y);
  //setWindowTitle(text);
}

//--------------------------

void 
WdgColorTable::mousePressEvent(QMouseEvent *event)
{
  //int x = event->pos().x();
  //int y = event->pos().y();
  //QString text = "mousePressEvent: " + QString::number(x) + "," + QString::number(y);
  //std::cout << text.toStdString()  << std::endl;
}
//--------------------------

void 
WdgColorTable::mouseMoveEvent(QMouseEvent *e)
{
  //    std::cout << "WdgColorTable::mouseMoveEvent: "
  //            << "  x(),y() = "  << e->x() << ", " << e->y()
  //            << '\n';
}

//--------------------------

//  bool 
//  WdgColorTable::eventFilter(QObject *obj, QEvent *e)
//  {
//    if(obj==this && (e->type()==QEvent::Enter || e->type()==QEvent::Leave)) std::cout << "WdgColorTable::eventFilter(...):\n";
//    return false;
//  }

//--------------------------

void 
WdgColorTable::onEdiH1()
{
  std::string str = (m_edi_h1 -> displayText()).toStdString();
  //std::cout << "WdgColorTable::onEdiH1()  H1: " << str << " i=" << string_to_int(str) << std::endl;
  
  m_h1 = (float)string_to_int(str);
  emit hueAngleIsEdited(1);
  setColorBar(m_h1, m_h2, m_cbar_width, m_figsize);
}

//--------------------------

void 
WdgColorTable::onEdiH2()
{
  std::string str = (m_edi_h2 -> displayText()).toStdString();
  //std::cout << "WdgColorTable::onEdiH2()  H2: " << str << " i=" << string_to_int(str) << std::endl;

  m_h2 = (float)string_to_int(str);
  emit hueAngleIsEdited(2);
  setColorBar(m_h1, m_h2, m_cbar_width, m_figsize);
}

//--------------------------

void 
WdgColorTable::onSetH(const unsigned& selected)
{
  //std::cout << "WdgColorTable::onSetH() h1:" << m_h1 << " h2:" << m_h2 
  //          << " selected:" << selected << '\n';

  if(selected == 1) m_edi_h1 -> setText(QString(val_to_string<int>(int(m_h1)).c_str()));
  if(selected == 2) m_edi_h2 -> setText(QString(val_to_string<int>(int(m_h2)).c_str()));

  setColorBar(m_h1, m_h2, m_cbar_width, m_figsize);
}

//--------------------------

void 
WdgColorTable::onButExit()
{
  //std::cout << "WdgColorTable::onButExit()\n";
  this->close(); // will call closeEvent(...)
}

//--------------------------

void
WdgColorTable::setColorBar( const float&    hue1,
                            const float&    hue2,
                            const unsigned& rows,
                            const unsigned& cols
                           )
{
  //std::cout << "WdgImage::setColorBar()\n";
  uint32_t* ctable = ColorTable(cols, hue1, hue2);
  uint32_t dimg[rows][cols];

  for(unsigned r=0; r<rows; ++r) {
    std::memcpy(&dimg[r][0], &ctable[0], cols*sizeof(uint32_t));
  }

  QImage image((const uchar*) &dimg[0], cols, rows, QImage::Format_ARGB32);
  setPixmapForLabel(image, m_pixmap_cbar, m_lab_cbar);
}

//--------------------------

ndarray<uint32_t,1> 
WdgColorTable::getColorTableAsNDArray(const unsigned& colors)
{
  uint32_t* ctable = ColorTable(colors, m_h1, m_h2);
  unsigned shape[] = {colors}; ndarray<uint32_t,1> nda(shape);
  //ndarray<uint32_t,1> nda = make_ndarray<uint32_t>(colors);
  std::memcpy(nda.data(), ctable, colors*sizeof(uint32_t)); 
  //for(unsigned i=0; i<colors; ++i) nda[i] = ctable[i];
  return nda;
}

//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
