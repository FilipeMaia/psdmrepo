#-----------------------------
#------ GlobalUtils.py -------
#-----------------------------

def stringOrNone(value):
    if value == None : return 'None'
    else             : return str(value)

def intOrNone(value):
    if value == None : return None
    else             : return int(value)

#-----------------------------

def get_save_fname_through_dialog_box(parent, path0, dial_title, filter='*.txt'):       

    path = str( QtGui.QFileDialog.getSaveFileName(parent,
                                                  caption   = dial_title,
                                                  directory = path0,
                                                  filter    = filter
                                                  ) )
    if path == '' :
        logger.debug('Saving is cancelled.', 'get_save_fname_through_dialog_box')
        #print 'Saving is cancelled.'
        return None
    logger.info('Output file: ' + path, 'get_save_fname_through_dialog_box')
    #print 'Output file: ' + path
    return path

#-----------------------------

def get_open_fname_through_dialog_box(parent, path0, dial_title, filter='*.txt'):       

    path = str( QtGui.QFileDialog.getOpenFileName(parent, dial_title, path0, filter=filter) )
    dname, fname = os.path.split(path)
    if dname == '' or fname == '' :
        logger.info('Input directiry name or file name is empty... keep file path unchanged...')
        #print 'Input directiry name or file name is empty... keep file path unchanged...'
        return None
    logger.info('Input file: ' + path, 'get_open_fname_through_dialog_box') 
    #print 'Input file: ' + path
    return path

#-----------------------------

def confirm_dialog_box(parent=None, text='Please confirm that you aware!', title='Please acknowledge') :
        """Pop-up MODAL box for confirmation"""

        mesbox = QtGui.QMessageBox(parent, windowTitle=title,
                                           text=text,
                                           standardButtons=QtGui.QMessageBox.Ok)
               #standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        #mesbox.setDefaultButton(QtGui.QMessageBox.Ok)
        #mesbox.setMinimumSize(400, 200)
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 100);" # Pinkish
        style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        mesbox.setStyleSheet (style)

        clicked = mesbox.exec_() # DISPLAYS THE QMessageBox HERE

        #if   clicked == QtGui.QMessageBox.Save :
        #    logger.info('Saving is requested', __name__)
        #elif clicked == QtGui.QMessageBox.Discard :
        #    logger.info('Discard is requested', __name__)
        #else :
        #    logger.info('Cancel is requested', __name__)
        #return clicked

        logger.info('You acknowkeged that saw the message:\n' + text, 'confirm_dialog_box')
        return

#-----------------------------

def help_dialog_box(parent=None, text='Help message goes here', title='Help') :
        """Pop-up NON-MODAL box for help etc."""

        messbox = QtGui.QMessageBox(parent, windowTitle=title,
                                           text=text,
                                           standardButtons=QtGui.QMessageBox.Close)
        messbox.setStyleSheet (cp.styleBkgd)
        messbox.setWindowModality (QtCore.Qt.NonModal)
        messbox.setModal (False)
        #clicked = messbox.exec_() # For MODAL dialog
        clicked = messbox.show()  # For NON-MODAL dialog
        logger.info('Help window is open' + text, 'help_dialog_box')
        return messbox

#-----------------------------
