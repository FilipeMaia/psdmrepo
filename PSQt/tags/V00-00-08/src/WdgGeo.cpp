//--------------------------

#include "PSQt/WdgGeo.h"
#include "PSQt/Logger.h"

#include <iostream>  // std::cout
#include <sstream>   // std::stringstream

#include <QIntValidator>

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

WdgGeo::WdgGeo( QWidget *parent, shpGO geo, const unsigned& pbits )
    : Frame(parent)
    , m_geo(geo)
    , m_pbits(pbits)
{
  m_lab_geo    = new QLabel();
  m_lab_par    = new QLabel();

  m_but_add    = new QPushButton("Add");
  m_but_sub    = new QPushButton("Sub");
  m_but_gr     = new QButtonGroup();
  m_but_gr -> addButton(m_but_add);
  m_but_gr -> addButton(m_but_sub);

  m_edi_step   = new QLineEdit("100");

  m_edi_x0     = new QLineEdit();
  m_edi_y0     = new QLineEdit();
  m_edi_z0     = new QLineEdit();
  m_edi_rot_x  = new QLineEdit();
  m_edi_rot_y  = new QLineEdit();
  m_edi_rot_z  = new QLineEdit();
  m_edi_tilt_x = new QLineEdit();
  m_edi_tilt_y = new QLineEdit();
  m_edi_tilt_z = new QLineEdit();

  m_rad_x0     = new QRadioButton("x0:");
  m_rad_y0     = new QRadioButton("y0:");
  m_rad_z0     = new QRadioButton("z0:");
  m_rad_rot_x  = new QRadioButton("rot x:");
  m_rad_rot_y  = new QRadioButton("rot y:");
  m_rad_rot_z  = new QRadioButton("rot z:");
  m_rad_tilt_x = new QRadioButton("tilt x:");
  m_rad_tilt_y = new QRadioButton("tilt y:");
  m_rad_tilt_z = new QRadioButton("tilt z:");

  map_radio_to_edit[m_rad_x0    ] = m_edi_x0    ;
  map_radio_to_edit[m_rad_y0    ] = m_edi_y0    ;
  map_radio_to_edit[m_rad_z0    ] = m_edi_z0    ;
  map_radio_to_edit[m_rad_rot_x ] = m_edi_rot_x ;
  map_radio_to_edit[m_rad_rot_y ] = m_edi_rot_y ;
  map_radio_to_edit[m_rad_rot_z ] = m_edi_rot_z ;
  map_radio_to_edit[m_rad_tilt_x] = m_edi_tilt_x;
  map_radio_to_edit[m_rad_tilt_y] = m_edi_tilt_y;
  map_radio_to_edit[m_rad_tilt_z] = m_edi_tilt_z;

  m_rad_gr = new QButtonGroup();

  m_rad_gr->addButton(m_rad_x0);
  m_rad_gr->addButton(m_rad_y0);
  m_rad_gr->addButton(m_rad_z0);		                  
  m_rad_gr->addButton(m_rad_rot_x);
  m_rad_gr->addButton(m_rad_rot_y);
  m_rad_gr->addButton(m_rad_rot_z);		                  
  m_rad_gr->addButton(m_rad_tilt_x);
  m_rad_gr->addButton(m_rad_tilt_y);
  m_rad_gr->addButton(m_rad_tilt_z);

  m_grid = new QGridLayout();
  m_grid -> addWidget(m_rad_x0,     0, 0);
  m_grid -> addWidget(m_rad_y0,     1, 0);
  m_grid -> addWidget(m_rad_z0,     2, 0);
  m_grid -> addWidget(m_rad_rot_x,  3, 0);
  m_grid -> addWidget(m_rad_rot_y,  4, 0);
  m_grid -> addWidget(m_rad_rot_z,  5, 0);		                  
  m_grid -> addWidget(m_rad_tilt_x, 6, 0);
  m_grid -> addWidget(m_rad_tilt_y, 7, 0);
  m_grid -> addWidget(m_rad_tilt_z, 8, 0);
  
  m_grid -> addWidget(m_edi_x0,     0, 2);
  m_grid -> addWidget(m_edi_y0,     1, 2);
  m_grid -> addWidget(m_edi_z0,     2, 2);
  m_grid -> addWidget(m_edi_rot_x,  3, 2);
  m_grid -> addWidget(m_edi_rot_y,  4, 2);
  m_grid -> addWidget(m_edi_rot_z,  5, 2);		                  
  m_grid -> addWidget(m_edi_tilt_x, 6, 2);
  m_grid -> addWidget(m_edi_tilt_y, 7, 2);
  m_grid -> addWidget(m_edi_tilt_z, 8, 2);
  
  m_cbox = new QHBoxLayout;
  m_cbox -> addStretch(1);
  m_cbox -> addWidget(m_but_sub);
  m_cbox -> addWidget(m_edi_step);
  m_cbox -> addWidget(m_but_add);
  m_cbox -> addStretch(1);

  m_box = new QVBoxLayout;
  m_box -> addWidget(m_lab_geo);
  m_box -> addWidget(m_lab_par);
  m_box -> addLayout(m_grid);
  m_box -> addLayout(m_cbox);

  this -> setLayout(m_box);

  this -> showTips();
  this -> setStyle();
  this -> updateGeoPars();

  //connect(m_rad_x0,  SIGNAL( clicked() ), this, SLOT( onRadioX()) );
  //connect(m_but_add, SIGNAL( clicked() ), this, SLOT(onButAddSub()) );
  connect(m_rad_gr, SIGNAL(buttonClicked(int)), this, SLOT(onRadioGroup()));
  connect(m_but_gr, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(onButAddSub(QAbstractButton*)));
  connect(this, SIGNAL(geoIsChanged(shpGO&)), this, SLOT(testSignalGeoIsChanged(shpGO&)));
  connect(m_edi_step, SIGNAL(editingFinished()),  this, SLOT(onEdiStep())); 

  m_rad_x0->setChecked(true);
  this -> onRadioGroup();
  //this -> move(300,50);
}

//--------------------------

void
WdgGeo::showTips() 
{
  std::string orig(" coordinate [um] of the object origin in parent frame");
  std::string rot (" axis rotation [degree] of the object in parent frame");
  std::string tilt(" axis tilt angle [degree] of the object in parent frame");

  m_rad_x0     -> setToolTip((std::string("x")+orig).c_str());
  m_rad_y0     -> setToolTip((std::string("y")+orig).c_str());     
  m_rad_z0     -> setToolTip((std::string("z")+orig).c_str());     
  m_rad_rot_x  -> setToolTip((std::string("x")+rot) .c_str());  
  m_rad_rot_y  -> setToolTip((std::string("y")+rot) .c_str()); 
  m_rad_rot_z  -> setToolTip((std::string("z")+rot) .c_str()); 
  m_rad_tilt_x -> setToolTip((std::string("x")+tilt).c_str()); 
  m_rad_tilt_y -> setToolTip((std::string("y")+tilt).c_str());  
  m_rad_tilt_z -> setToolTip((std::string("z")+tilt).c_str());  
}

//--------------------------

void
WdgGeo::setStyle() 
{
  //this -> setWindowTitle(tr("WdgGeo"));
  //this -> setMinimumWidth(700);
  //this -> setFixedHeight(50);
  //this -> setFixedHeight( (m_show_frame)? 50 : 34);

  //if (! m_show_frame) this -> setContentsMargins(-9,-9,-9,-9);

  //this -> setContentsMargins(-9,-9,-9,-9);
  //this -> setContentsMargins(-5,-5,-5,-5);

  this -> setFixedWidth(220);

  unsigned width = 70;
  m_edi_x0    -> setFixedWidth(width);
  m_edi_y0    -> setFixedWidth(width);
  m_edi_z0    -> setFixedWidth(width);
  m_edi_rot_x -> setFixedWidth(width);
  m_edi_rot_y -> setFixedWidth(width);
  m_edi_rot_z -> setFixedWidth(width);
  m_edi_tilt_x-> setFixedWidth(width);
  m_edi_tilt_y-> setFixedWidth(width);
  m_edi_tilt_z-> setFixedWidth(width);

  m_edi_step  -> setFixedWidth(width);
  m_but_add   -> setFixedWidth(50);
  m_but_sub   -> setFixedWidth(50);

  m_edi_x0    -> setReadOnly(true);
  m_edi_y0    -> setReadOnly(true);
  m_edi_z0    -> setReadOnly(true);
  m_edi_rot_x -> setReadOnly(true);
  m_edi_rot_y -> setReadOnly(true);
  m_edi_rot_z -> setReadOnly(true);
  m_edi_tilt_x-> setReadOnly(true);
  m_edi_tilt_y-> setReadOnly(true);
  m_edi_tilt_z-> setReadOnly(true);
}

//--------------------------

void 
WdgGeo::resizeEvent(QResizeEvent *event)
{
  stringstream ss; ss << "w:" << event->size().width() << " h:" <<  event->size().height();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void
WdgGeo::moveEvent(QMoveEvent *event)
{
  stringstream ss; ss << "x:" << event->pos().x() << " y:" << event->pos().y();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
WdgGeo::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  stringstream ss; ss << "WdgGeo::closeEvent(...): type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str()); 
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------
void 
WdgGeo::onEdiStep()
{
  MsgInLog(_name_(), INFO, "Step value is changed to " + (m_edi_step->displayText()).toStdString());
}

//--------------------------

void
WdgGeo::onRadioGroup()
{
  QRadioButton* but = (QRadioButton*) m_rad_gr->checkedButton();

  //std::cout << "onRadioGroup:  checked radio button:" << (but->text()).toStdString();
  //std::cout << " mapped edit field:" << (map_radio_to_edit[but]->text()).toStdString() << '\n';  

  stringstream ss; ss << "Selected button: " << (but->text()).toStdString() 
                      << " to edit field: " << (map_radio_to_edit[but]->text()).toStdString();
  MsgInLog(_name_(), INFO, ss.str()); 

  if(but == m_rad_x0
  || but == m_rad_y0
  || but == m_rad_z0) {
      m_edi_step -> setText("100");
      m_edi_step -> setValidator(new QIntValidator(0,1000000,this));
  }

  if(but == m_rad_rot_x
  || but == m_rad_rot_y
  || but == m_rad_rot_z) {
      m_edi_step -> setText("90");
      m_edi_step -> setValidator(new QIntValidator(0,360,this));
  }

  if(but == m_rad_tilt_x
  || but == m_rad_tilt_y
  || but == m_rad_tilt_z) {
      m_edi_step -> setText("1");
      m_edi_step -> setValidator(new QDoubleValidator(0,10000,3,this));
  }
}

//--------------------------

void
WdgGeo::onButAddSub(QAbstractButton* button)
{
  QPushButton*  but       = (QPushButton*) button;
  QRadioButton* but_radio = (QRadioButton*) m_rad_gr->checkedButton();
  QLineEdit* edi = map_radio_to_edit[but_radio];

  double dval = (m_edi_step->text()).toDouble();
  if (but == m_but_sub) dval = -dval;

  double val = (edi->text()).toDouble();
  
  //std::cout << "onButAddSub add/sub " << dval << " to the value " << val << '\n';

  val += dval;

  stringstream ss;
  if(but_radio == m_rad_tilt_x
  || but_radio == m_rad_tilt_y
  || but_radio == m_rad_tilt_z) ss << fixed << std::setprecision(3) << val;
  else                          ss << fixed << std::setprecision(0) << val;

  edi->setText(ss.str().c_str());

  stringstream smsg; smsg << "Value of \"" << (but_radio->text()).toStdString() << "\" is changed to " << val << " -> set geo, emit signal: geoIsChanged(m_geo)";
  MsgInLog(_name_(), INFO, smsg.str()); 

  this -> setGeoPars();
  //std::cout << "Do something here\n";
}

//--------------------------

void
WdgGeo::setGeoPars()
{
  const double x0     = (m_edi_x0    ->text()).toDouble();
  const double y0     = (m_edi_y0    ->text()).toDouble();
  const double z0     = (m_edi_z0    ->text()).toDouble();
  const double rot_x  = (m_edi_rot_x ->text()).toDouble();
  const double rot_y  = (m_edi_rot_y ->text()).toDouble();
  const double rot_z  = (m_edi_rot_z ->text()).toDouble();
  const double tilt_x = (m_edi_tilt_x->text()).toDouble();
  const double tilt_y = (m_edi_tilt_y->text()).toDouble();
  const double tilt_z = (m_edi_tilt_z->text()).toDouble();

  m_geo->set_geo_pars(x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x);
  //m_geo->print_geo();
  
  MsgInLog(_name_(), DEBUG, "Emit signal with changed geo"); 
  emit geoIsChanged(m_geo);
}

//--------------------------

void
WdgGeo::updateGeoPars()
{
  double x, y, z, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x;     
  m_geo->get_geo_pars(x, y, z, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x);

  stringstream ss_geo, ss_par; 
  ss_geo << "Object: " << m_geo->get_geo_name()    << "." << m_geo->get_geo_index();
  ss_par << "Parent: " << m_geo->get_parent_name() << "." << m_geo->get_parent_index();

  //m_geo->print_geo();
  //std::string str = m_geo->string_geo();
  //std::cout << "string_geo(): " << str << '\n';

  stringstream ss_x0    ; ss_x0     << fixed << std::setprecision(0) << x;
  stringstream ss_y0    ; ss_y0     << fixed << std::setprecision(0) << y;
  stringstream ss_z0    ; ss_z0     << fixed << std::setprecision(0) << z;
  stringstream ss_rot_x ; ss_rot_x  << fixed << std::setprecision(0) << rot_x;
  stringstream ss_rot_y ; ss_rot_y  << fixed << std::setprecision(0) << rot_y;
  stringstream ss_rot_z ; ss_rot_z  << fixed << std::setprecision(0) << rot_z;
  stringstream ss_tilt_x; ss_tilt_x << fixed << std::setprecision(3) << tilt_x;
  stringstream ss_tilt_y; ss_tilt_y << fixed << std::setprecision(3) << tilt_y;
  stringstream ss_tilt_z; ss_tilt_z << fixed << std::setprecision(3) << tilt_z;

  m_lab_geo   ->setText(ss_geo.str().c_str());
  m_lab_par   ->setText(ss_par.str().c_str());

  m_edi_x0    ->setText(ss_x0    .str().c_str());
  m_edi_y0    ->setText(ss_y0    .str().c_str());
  m_edi_z0    ->setText(ss_z0    .str().c_str());
  m_edi_rot_x ->setText(ss_rot_x .str().c_str());
  m_edi_rot_y ->setText(ss_rot_y .str().c_str());
  m_edi_rot_z ->setText(ss_rot_z .str().c_str());
  m_edi_tilt_x->setText(ss_tilt_x.str().c_str());
  m_edi_tilt_y->setText(ss_tilt_y.str().c_str());
  m_edi_tilt_z->setText(ss_tilt_z.str().c_str());
}

//--------------------------

void
WdgGeo::setNewGO(shpGO& geo)
{
  m_geo = geo;
  this -> updateGeoPars();
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
WdgGeo::testSignalGeoIsChanged(shpGO& geo)
{
  //std::cout << "testSignalGeoIsChanged():\n";
  //geo->print_geo();
  //m_geo->print_geo();
  MsgInLog(_name_(), DEBUG, "testSignalGeoIsChanged(): " + geo->str_data()); // string_geo()); 
}

//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
