#ifndef FRAME_H
#define FRAME_H

#include <QtGui>
//#include <QtCore>
//#include <Qt>

#include "PSQt/WdgImage.h"
#include "PSQt/WdgFile.h"

namespace PSQt {

//--------------------------

/**
 *  @ingroup PSQt
 * 
 *  @brief Inherits from QFrame and sets its basic parameters. Is used to display widget frame.
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

//--------------------------

class Frame : public QFrame
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    Frame( QWidget *parent = 0, Qt::WindowFlags f = 0 );

    void setFrame() ;
    void showTips() ;
    void setBoarderVisible(const bool isVisible) ;

    //void resizeEvent     (QResizeEvent *event = 0) ;
    //void closeEvent      (QCloseEvent  *event = 0) ;
    //void moveEvent       (QMoveEvent   *event = 0) ;
    //void mousePressEvent (QMouseEvent  *event = 0) ;

 private:

};

} // namespace PSQt

#endif // FRAME_H
