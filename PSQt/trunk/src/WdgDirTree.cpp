//--------------------------

#include "PSQt/WdgDirTree.h"
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

#include <dirent.h> // for DIR, dirent

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

  WdgDirTree::WdgDirTree( QWidget *parent, const std::string& dir_top, const unsigned& pbits)
    : Frame(parent)
    , m_dir_top(dir_top)
    , m_pbits(pbits)
//  : QWidget(parent)
{
  m_model = 0;

  setGeometry(100, 100, 400, 300);
  setWindowTitle("Item selection tree");
 
  std::cout << "point 1\n";

  makeTreeModel();

  m_view = new QTreeView();  
  m_view->setModel(m_model);
  m_view->setAnimated(true);

  QVBoxLayout *vbox = new QVBoxLayout();
  vbox -> addWidget(m_view);
  //vbox -> addStretch(1);
  //vbox -> addLayout(hbox);

  this -> setLayout(vbox);
  this -> move(100,50);  

  //connect(m_but_exit, SIGNAL( clicked() ), this, SLOT(onButExit()) );
  //connect(m_but_test, SIGNAL( clicked() ), m_image, SLOT(onTest()) );
  //connect(m_file, SIGNAL(fileNameIsChanged(const std::string&)), m_image, SLOT(onFileNameChanged(const std::string&)) ); 
  m_view->expandAll();
  showTips();
}

//--------------------------

void
WdgDirTree::showTips() 
{
  //m_file_geo  -> setToolTip("Select \"geometry\" file");
  //m_file_geo  -> setToolTip("Select ndarray with image file");
  //m_but_exit  -> setToolTip("Exit application");
}

//--------------------------

void 
WdgDirTree::makeTreeModel()
{
  if ( m_model == 0 ) {
    m_model = new QStandardItemModel();
    //m_model->setHorizontalHeaderLabels(QStringList(QString('o')));
    updateTreeModel(m_dir_top);
  }
}

//--------------------------

void 
WdgDirTree::updateTreeModel(const std::string& dir_top)
{
    if (dir_top.empty()) return;

    m_model->clear();

    fillTreeModel(dir_top,0,0,m_pbits);
    //fillTreeModelTest();
}

//--------------------------

void 
WdgDirTree::fillTreeModel(const std::string& path0, QStandardItem* pitem, const unsigned& level, const unsigned& pbits)
{
   std::string bname = basename(path0); 
   std::string dname = dirname(path0); 
   std::string path = (dname.empty()) ? "." : dname;
   if (!bname.empty()) path += '/' + bname;

   if (bname.empty()) bname = ".";
   else if (bname==std::string("./")) bname = ".";

   QStandardItem* item_add = new QStandardItem(QString(bname.c_str()));
   QStandardItem* item_parent = (pitem) ? pitem : m_model->invisibleRootItem();

   item_parent->appendRow(item_add);

   if(pbits & 1) std::cout << "==== Add item: \"" << path << "\"\n";
      
   if (is_link(path)) {
     if(pbits & 2) std::cout << "    add as a link\n";
     return;
   }

   if (is_file(path)) {
       if(pbits & 2) std::cout << "    add as a file\n";
       //item_add->setIcon(icon);
       item_add->setCheckable(true);
       return;
   }

   else if (is_directory(path)) {
       if(pbits & 2) std::cout << "    add as a directory\n";

       DIR*     dir;
       dirent*  pdir; 
       dir = opendir(path.c_str());      // open current directory
       while ((pdir = readdir(dir))) {
	  std::string fname(pdir->d_name);

          if(pbits & 2) cout << "    " << fname << '\n';

          if(fname==std::string(".")) continue;
          if(fname==std::string("..")) continue;
          if(fname.empty()) continue;

	  std::string path_to_child = path; path_to_child += '/' + fname;
          if(pbits & 2) cout << "    path_to_child: " << path_to_child << '\n'; 


          fillTreeModel(path_to_child, item_add, level+1, pbits);
       }
       closedir(dir);
   }

   //std::cout << "WdgDirTree::fillTreeModel(...) !!!!!!!!!!! TBD\n";
}


//--------------------------

void 
WdgDirTree::fillTreeModelTest()
{
   QStandardItem* item_parent = m_model->invisibleRootItem();

   QStandardItem* item1 = new QStandardItem(QString("item1"));
   QStandardItem* item2 = new QStandardItem(QString("item2"));
   QStandardItem* item3 = new QStandardItem(QString("item3"));

   item_parent->appendRow(item1);
   item_parent->appendRow(item2);
   item_parent->appendRow(item3);

   QStandardItem* item31 = new QStandardItem(QString("item31"));
   QStandardItem* item32 = new QStandardItem(QString("item32"));

   item3->appendRow(item31);
   item3->appendRow(item32);
}

//--------------------------

void 
WdgDirTree::resizeEvent(QResizeEvent *event)
{
  //  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  stringstream ss; ss << "Window is resized, w:" << event->size().width() << " h:" <<  event->size().height();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
WdgDirTree::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  std::stringstream ss; ss << "closeEvent(...): type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------
void
WdgDirTree::moveEvent(QMoveEvent *event)
{
  int x = event->pos().x();
  int y = event->pos().y();
  QString text = QString::number(x) + "," + QString::number(y);
  setWindowTitle(text);
}

//--------------------------

void 
WdgDirTree::mousePressEvent(QMouseEvent *event)
{
  //int x = event->pos().x();
  //int y = event->pos().y();
  //QString text = "mousePressEvent: " + QString::number(x) + "," + QString::number(y);
  //std::cout << text.toStdString()  << std::endl;
}

//--------------------------
//--------------------------
//--------------------------

void 
WdgDirTree::onButExit()
{
  std::cout << "WdgDirTree::onButExit()\n";
  this->close(); // will call closeEvent(...)
}

//--------------------------

} // namespace PSQt

//--------------------------
