#ifndef WDGPOINTPOSITION_H
#define WDGPOINTPOSITION_H

#include <Qt>
#include <QtGui>
#include <QtCore>

#include "PSQt/Frame.h"

namespace PSQt {

/**
 *  @ingroup PSQt
 * 
 *  @brief Widget to display two values for 2-d point position. 
 * 
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIImageViewer
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class WdgPointPos : public Frame // QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

public:
    WdgPointPos( QWidget *parent = 0
	    , const std::string& label1=std::string("X:")
	    , const std::string& label2=std::string("Y:") 
	    , const float& val1=0
	    , const float& val2=0
	    , const bool& show_boarder=true
	    , const unsigned& fld_width=100
	    , const unsigned& precision=2);

    ~WdgPointPos(){}

    void resizeEvent(QResizeEvent *event = 0) ;
    void closeEvent (QCloseEvent  *event = 0) ;

 public slots:
    void onEdi() ;
    void testPosIsChanged(const QPointF& pos);
    void setPointPos(const QPointF& pos);

 signals :
    void posIsChanged(const QPointF&);

 private :
    bool         m_show_boarder;
    unsigned     m_fld_width;
    unsigned     m_precision;

    QLabel*      m_lab1;
    QLabel*      m_lab2;
    QLineEdit*   m_edi1;
    QLineEdit*   m_edi2;

    inline const char* _name_(){return "WdgPointPos";}
    void showTips() ;
};

} // namespace PSQt

#endif // WDGPOINTPOSITION_H
