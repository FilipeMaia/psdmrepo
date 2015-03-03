#ifndef WDGFILE_H
#define WDGFILE_H

//#include <QVBox>
//#include <QWidget>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>
#include <QString>

#include <Qt>
#include <QtGui>
#include <QtCore>

#include "PSQt/Frame.h"

namespace PSQt {

class WdgFile : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    WdgFile( QWidget *parent = 0, 
             const std::string& but_title=std::string("File:"), 
             const std::string& path=std::string(),
             const std::string& search_fmt=std::string("*.data *.png *.gif *.jpg *.jpeg\n *"),
             const bool& show_frame=true,
             const unsigned& but_width=100);
    //    ~WdgFile(){}

    void resizeEvent(QResizeEvent *event = 0) ;
    void closeEvent (QCloseEvent  *event = 0) ;
    std::string& fileName() { return m_path; };

 public slots:
    void onButFile() ;
    void onEdiFile() ;
    void testSignalString(const std::string& fname) ;

 signals :
    void fileNameIsChanged(const std::string& fname) ;
    void valueChanged(int value) ;

 private :
    std::string  m_path;
    std::string  m_search_fmt;
    bool         m_show_frame;
    //QLabel*      m_lab_fname;
    QFrame*      m_frame;
    QLineEdit*   m_edi_file;
    QPushButton* m_but_file;

    inline const char* _name_(){return "WdgFile";}
    void showTips() ;
    void setFrame() ;
    bool fileExists(const std::string& fname) ;
    bool setNewFileName(const std::string& fname) ;
};

} // namespace PSQt

#endif // WDGFILE_H
