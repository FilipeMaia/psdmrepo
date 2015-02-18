#ifndef WDGGEOTREE_H
#define WDGGEOTREE_H

//#include "PSQt/WdgImage.h"
//#include "PSQt/WdgFile.h"

#include "PSQt/Frame.h"
#include "PSCalib/GeometryAccess.h"
#include "PSCalib/GeometryObject.h"
#include <boost/shared_ptr.hpp>

#include <Qt>
#include <QtGui>
#include <QtCore>

namespace PSQt {

/**
 *  @ingroup PSQt
 *
 *  @brief Qt Widget for geometry tree
 * 
 *  WdgGeoTree is a sub-class of Frame (QWidget)
 *  GeoTree is a sub-class of QTreeView
 *
 *  Geometry tree for input file is displayed in the window.
 *  GeoTree emits signal "selectedGO(shpGO&)" when the geometry object is selected.
 *
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIMain
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

typedef boost::shared_ptr<PSCalib::GeometryObject> shpGO;

class GeoTree;

class WdgGeoTree : public Frame
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:

  WdgGeoTree( QWidget *parent = 0
            , const std::string& gfname = "/reg/g/psdm/detector/alignment/cspad/geo-cspad-test-2-end.data"
            , const unsigned& pbits=255);

    void resizeEvent     (QResizeEvent *event = 0) ;
    void moveEvent       (QMoveEvent   *event = 0) ;
    void mousePressEvent (QMouseEvent  *event = 0) ;
    void closeEvent      (QCloseEvent  *event = 0) ;

    QTreeView* get_view() { return m_view; };
    GeoTree* get_geotree() { return m_geotree; };
    PSCalib::GeometryAccess* geoacc();
    void showTips() ;

 private:
    unsigned    m_pbits;
    GeoTree*    m_geotree;
    QTreeView*  m_view;
    inline const char* _name_(){return "WdgGeoTree";}
};


class GeoTree : public QTreeView
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:

    GeoTree( QWidget *parent = 0, 
             const std::string& gfname = "/reg/g/psdm/detector/alignment/cspad/geo-cspad-test-2-end.data", 
             const unsigned& pbits=255);

    PSCalib::GeometryAccess* geoacc(){return m_geoacc;}

    // Select item programatically
    void setItemSelected(const QModelIndex& index);

    // Select item programatically
    void setItemSelected(const QStandardItem* item=0);

 public slots:
    void updateTreeModel(const std::string& gfname = "/reg/g/psdm/detector/alignment/cspad/geo-cspad-test-2-end.data");
    void testSignalGeometryIsLoaded(PSCalib::GeometryAccess*);
    void testSignalString(const std::string& str);
    void testSignalGO(shpGO&);
    void testSignalCollapsed(const QModelIndex& index);
    void testSignalExpanded(const QModelIndex& index);
    void currentChanged(const QModelIndex & current, const QModelIndex & previous);

 signals:
    void geometryIsLoaded(PSCalib::GeometryAccess*);
    void selectedText(const std::string& str);
    void selectedGO(shpGO&);

 private:

    std::string              m_gfname;
    QTreeView*               m_view;
    unsigned                 m_pbits;
    QStandardItemModel*      m_model;
    PSCalib::GeometryAccess* m_geoacc;

    std::map<QStandardItem*,shpGO> map_item_to_geo;
    std::map<shpGO,QStandardItem*> map_geo_to_item;

    inline const char* _name_(){return "GeoTree";}
    bool loadGeometry(const std::string& gfname);
    void makeTreeModel();
    void fillTreeModel(shpGO geo=shpGO(), QStandardItem* item=0, const unsigned& level=0, const unsigned& pbits=0);
    void fillTreeModelTest();
};

} // namespace PSQt

#endif // WDGGEOTREE_H
