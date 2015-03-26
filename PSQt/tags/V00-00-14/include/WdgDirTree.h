#ifndef WDGDIRTREE_H
#define WDGDIRTREE_H

#include "PSQt/Frame.h"
//#include <QVBox>
//#include <QWidget>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>

#include <Qt>
#include <QtGui>
#include <QtCore>

//#include "PSQt/WdgImage.h"
#include "PSQt/WdgFile.h"

namespace PSQt {

/**
 *  @ingroup PSQt
 * 
 *  @brief Wiget works with file directory tree (Is not used in current project).
 * 
 *  @code
 *  @endcode
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see 
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

//class WdgDirTree : public QWidget
class WdgDirTree : public Frame
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    WdgDirTree( QWidget *parent = 0, 
                const std::string& dir_top = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15", 
                const unsigned& pbits=0);

    void showTips() ;

    void resizeEvent     (QResizeEvent *event = 0) ;
    void closeEvent      (QCloseEvent  *event = 0) ;
    void moveEvent       (QMoveEvent   *event = 0) ;
    void mousePressEvent (QMouseEvent  *event = 0) ;

 public slots:

    void onButExit() ;

 private:

    std::string         m_dir_top;
    QTreeView*          m_view;
    QStandardItemModel* m_model;
    unsigned            m_pbits;

    inline const char* _name_(){return "WdgDirTree";}
    void makeTreeModel();
    void updateTreeModel(const std::string& dir_top = "./");
    void fillTreeModel(const std::string& dir_top = "./", QStandardItem* item=0, const unsigned& level=0, const unsigned& pbits=0);
    void fillTreeModelTest();
};

} // namespace PSQt

#endif // WDGDIRTREE_H
