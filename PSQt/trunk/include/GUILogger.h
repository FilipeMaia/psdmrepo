#ifndef GUILOGGER_H
#define GUILOGGER_H

#include "PSQt/Frame.h"
#include "PSQt/Logger.h"

#include <Qt>
#include <QtGui>
#include <QtCore>


namespace PSQt {

/**
 *  @ingroup PSQt
 *
 *  @brief GUI (Frame) for Logger.
 * 
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIMain
 *
 *  @version $Id:$
 *
 *  @author Mikhail Dubrovin
 */


//class GUILogger : public QWidget
class GUILogger : public Frame // , Logger
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
  GUILogger(QWidget *parent=0, const bool& showbuts=true);
    //    ~GUILogger(){}
    void resizeEvent(QResizeEvent *event = 0) ;
    void moveEvent  (QMoveEvent   *event = 0) ;
    void closeEvent (QCloseEvent  *event = 0) ;

 public slots:
    void addNewRecord(Record&);
    void onCombo(int);
    void onSave();

    //signals :
    //void geoIsChanged(shpGO&);

 private :

    bool          m_showbuts;

    QTextEdit*    m_txt_edi;
    QPushButton*  m_but_save;
    QHBoxLayout*  m_cbox;
    QVBoxLayout*  m_vbox;
    QComboBox*    m_combo;
    QStringList   m_list; 

    inline const std::string _name_(){return "GUILogger";}
    void showTips();
    void setStyle();
    void addStartRecords();
};

} // namespace PSQt

#endif // GUILOGGER_H
