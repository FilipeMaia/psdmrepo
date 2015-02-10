#ifndef PSQTGUIMAIN_H
#define PSQTGUIMAIN_H

//#include <QVBox>
//#include <QWidget>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>

#include <Qt>
#include <QtGui>
#include <QtCore>

namespace PSQt {

class PSQtGUIMain : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    PSQtGUIMain( QWidget *parent = 0 );

    void setFrame() ;
    void showTips() ;

    void resizeEvent(QResizeEvent *event = 0) ;
    void closeEvent (QCloseEvent  *event = 0) ;

    bool fileExists(std::string fname) ;


 public slots:

    void onButStart() ;
    void onButStop() ;
    void onButSave() ;
    void onButExit() ;
    void onButSelectXtcFile() ;
    void onButSelectCfgFile() ;
    void onEditXtcFileName() ;
    void onEditCfgFileName() ;

 private:

    QFrame*      m_frame;

    //QLabel*      m_lab_fname;
    QLineEdit*   m_edi_fncfg;
    QLineEdit*   m_edi_fnxtc;
    QPushButton* m_but_fncfg;
    QPushButton* m_but_fnxtc;
    QPushButton* m_but_start;
    QPushButton* m_but_stop;
    QPushButton* m_but_save;
    QPushButton* m_but_exit;

};

} // namespace PSQt

#endif // PSQTGUIMAIN_H
