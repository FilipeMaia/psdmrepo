//--------------------------

#include "PSQt/WdgGeoTree.h"
#include "PSQt/QGUtils.h"
#include "PSQt/Logger.h"

//#include <string>
//#include <fstream>   // ofstream
//#include <iomanip>   // for setw, setfill
//#include <math.h>
//#include <stdio.h>

#include <sstream>   // for stringstream
#include <iostream>    // cout
#include <fstream>    // ifstream(fname), ofstream

//#include <dirent.h> // for DIR, dirent

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

WdgGeoTree::WdgGeoTree(QWidget *parent, const std::string& gfname, const unsigned& pbits) //, const std::string& gfname, const unsigned& pbits)
    : Frame(parent)
    , m_pbits(pbits)
{
  m_geotree = new GeoTree(parent,gfname,m_pbits); 
  m_view = (QTreeView*) m_geotree;

  QVBoxLayout *vbox = new QVBoxLayout();
  vbox -> addWidget(m_view);
  setLayout(vbox);
  //move(100,50);  
  this -> setContentsMargins(-9,-9,-9,-9);
  showTips();
}
//--------------------------

PSCalib::GeometryAccess* 
WdgGeoTree::geoacc()
{
  return m_geotree->geoacc();
}

//--------------------------

void
WdgGeoTree::showTips() 
{
  //m_file_geo  -> setToolTip("Select \"geometry\" file");
  //m_file_geo  -> setToolTip("Select ndarray with image file");
  //m_but_exit  -> setToolTip("Exit application");
}

//--------------------------

void 
WdgGeoTree::resizeEvent(QResizeEvent *event)
{
  stringstream ss; ss << "w:" << event->size().width() << " h:" <<  event->size().height();
  setWindowTitle(ss.str().c_str());
}

//--------------------------
void
WdgGeoTree::moveEvent(QMoveEvent *event)
{
  stringstream ss; ss << "x:" << event->pos().x() << " y:" << event->pos().y();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
WdgGeoTree::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  std::stringstream ss; ss << "closeEvent(...): type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

void 
WdgGeoTree::mousePressEvent(QMouseEvent *event)
{
  //int x = event->pos().x();
  //int y = event->pos().y();
  //QString text = "mousePressEvent: " + QString::number(x) + "," + QString::number(y);
  //std::cout << text.toStdString()  << std::endl;
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

  GeoTree::GeoTree(QWidget *parent, const std::string& gfname, const unsigned& pbits)
    : QTreeView(parent)
    , m_gfname(gfname)
    , m_pbits(pbits)
    , m_model(0)
    , m_geoacc(0)
//  : Frame(parent)
//  : QWidget(parent)
{
  if (m_pbits & 1) MsgInLog(_name_(), INFO, "In c-tor");

  //setGeometry(100, 100, 200, 300);
  setWindowTitle("Geo selection tree");
 
  makeTreeModel();

  m_view = this;
  //m_view = new QTreeView();  

  m_view->setModel(m_model);
  m_view->setAnimated(true);
  m_view->setHeaderHidden(true);
  m_view->expandAll();
  //m_view->setMinimumWidth(200);

  //QVBoxLayout *vbox = new QVBoxLayout();
  //vbox -> addWidget(m_view);
  //setLayout(vbox);
  //move(100,50);  


  //if(m_pbits & 4) 

  //connect(this, SIGNAL(selectedText(const std::string&)), this, SLOT(testSignalString(const std::string&))); 
  connect(this, SIGNAL(geometryIsLoaded(PSCalib::GeometryAccess*)), this, SLOT(testSignalGeometryIsLoaded(PSCalib::GeometryAccess*))); 
  connect(this, SIGNAL(selectedGO(shpGO&)), this, SLOT(testSignalGO(shpGO&))); 
  connect(this, SIGNAL(collapsed(const QModelIndex&)), this, SLOT(testSignalCollapsed(const QModelIndex&)));
  connect(this, SIGNAL(expanded(const QModelIndex&)), this, SLOT(testSignalExpanded(const QModelIndex&)));
}

//--------------------------

void 
GeoTree::makeTreeModel()
{
  if ( m_model == 0 ) {
    m_model = new QStandardItemModel();
    //m_model->setHorizontalHeaderLabels(QStringList(QString('o')));
    updateTreeModel(m_gfname);
  }
}

//--------------------------

bool
GeoTree::loadGeometry(const std::string& gfname)
{
  if (!file_exists(gfname)) {
    stringstream ss; ss << "Geometry file \"" << gfname << "\" does not exist";
    MsgInLog(_name_(), WARNING, ss.str());
    return false;
  }

  if (! m_geoacc) delete m_geoacc;
  m_geoacc = new PSCalib::GeometryAccess(gfname);
  if (m_pbits & 2) m_geoacc->print_list_of_geos();

  emit geometryIsLoaded(m_geoacc);
  
  return true;
}

//--------------------------

void 
GeoTree::updateTreeModel(const std::string& gfname)
{
  //if(m_pbits & 4) 
  //std::cout << "updateTreeModel for file: " << gfname << '\n';
  MsgInLog(_name_(), INFO, std::string("Update geometry-tree model for file: ") + gfname); 

  if(! loadGeometry(gfname)) return;

  m_model->clear();
  map_item_to_geo.clear();
  map_geo_to_item.clear();

  fillTreeModel(shpGO(),0,0,m_pbits);
  //fillTreeModelTest();

  //m_view->expandAll();
  expandAll();

}

//--------------------------

void 
GeoTree::fillTreeModel( shpGO geo_add
                      , QStandardItem* parent
                      , const unsigned& level
                      , const unsigned& pbits )
{
   shpGO geo = (geo_add != shpGO()) ? geo_add : m_geoacc->get_top_geo();

   QStandardItem* item_parent = (parent) ? parent : m_model->invisibleRootItem();

   stringstream ss; ss << geo->get_geo_name() << "." << geo->get_geo_index();
   std::string iname = ss.str();
   QStandardItem* item_add = new QStandardItem(iname.c_str());

   item_parent->appendRow(item_add);

   map_item_to_geo[item_add] = geo;
   map_geo_to_item[geo_add] = item_add;

   std::vector<shpGO> list = geo->get_list_of_children();

   if(pbits & 1) {stringstream ss; ss << "==== Add item: \"" << iname << "\" number of children:" << list.size();}
      
   if (list.size()) {
       if(pbits & 1) ss << "    add as a group\n";

       //if (parent) item_add->setCheckable(true);

       for(std::vector<shpGO>::iterator it=list.begin(); it!= list.end(); ++ it) { 
	 //if(pbits & 2) cout << " add child from the list: " << (*it)->get_geo_name() << '/n';
         fillTreeModel(*it, item_add, level+1, pbits);
       }
   }
   else {
       if(pbits & 1) {ss << "    add as a set\n";
       //item_add->setIcon(icon);
       //item_add->setCheckable(true);
         MsgInLog(_name_(), INFO, ss.str()); 
       }
       return;
   }
   if(pbits & 1) MsgInLog(_name_(), INFO, ss.str()); 
}

//--------------------------

void 
GeoTree::fillTreeModelTest()
{
   QStandardItem* item_parent = m_model->invisibleRootItem();

   QStandardItem* item1 = new QStandardItem("item1");
   QStandardItem* item2 = new QStandardItem("item2");
   QStandardItem* item3 = new QStandardItem("item3");

   item_parent->appendRow(item1);
   item_parent->appendRow(item2);
   item_parent->appendRow(item3);

   QStandardItem* item31 = new QStandardItem("item31");
   QStandardItem* item32 = new QStandardItem("item32");

   item3->appendRow(item31);
   item3->appendRow(item32);
}

//--------------------------

void 
GeoTree::currentChanged(const QModelIndex & index, const QModelIndex & index_old)
{
  //std::cout << "currentChanged:: new r:" << index.row() << " c:" << index.column() 
  //                      << "     old r:" << index_old.row() << " c:" << index_old.column() << '\n';

  QStandardItem *item = m_model->itemFromIndex(index);

  std::string str(item->text().toStdString());

  //if(m_pbits & 4) {
  //  stringstream ss; ss << "currentChanged::item"
  //          << " r:" << item->row() 
  //          << " c:" << item->column() 
  //	    << " a:" << item->accessibleText().toStdString()
  //	    << " t:" << str; }

  MsgInLog(_name_(), INFO, "Selected item: " + str + " emit signal"); 

  //emit selectedText(str);
  emit selectedGO(map_item_to_geo[item]);
}

//--------------------------

void 
GeoTree::setItemSelected(const QModelIndex& index)
{
  this->selectionModel()->select(index, QItemSelectionModel::Select);
  currentChanged(index,index);
}

//--------------------------

void 
GeoTree::setItemSelected(const QStandardItem* item)
{
  std::map<shpGO,QStandardItem*>::iterator it=map_geo_to_item.begin();
  const QStandardItem* item_sel = (item) ? item : it->second;
  this->setItemSelected(m_model->indexFromItem(item_sel));
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
void 
GeoTree::testSignalGeometryIsLoaded(PSCalib::GeometryAccess*)
{
  MsgInLog(_name_(), DEBUG, "testSignalGeometryIsLoaded"); 
}

//--------------------------

void 
GeoTree::testSignalString(const std::string& str)
{
  MsgInLog(_name_(), DEBUG, "GeoTree::testSignalString(string): str = " + str); 
}

//--------------------------

void 
GeoTree::testSignalGO(shpGO& geo)
{
  stringstream ss; ss << "GeoTree::testSignalGO(shpGO): "; // << geo.get()->get_geo_name() << ' ' << geo.get()->get_geo_index() << '\n';   // geo.get()->print_geo();
  ss << geo->string_geo();
  MsgInLog(_name_(), DEBUG, ss.str()); 
}

//--------------------------

void 
GeoTree::testSignalCollapsed(const QModelIndex& index)
{
  stringstream ss; ss << "GeoTree::testSignalCollapsed(shpGO): row:" << index.row() << " col:" << index.column();
  MsgInLog(_name_(), DEBUG, ss.str()); 
}

//--------------------------

void 
GeoTree::testSignalExpanded(const QModelIndex& index)
{
  stringstream ss; ss << "GeoTree::testSignalExpanded(shpGO): row:" << index.row() << " col:" << index.column();
  MsgInLog(_name_(), DEBUG, ss.str()); 
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
