#!@PYTHON@
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Interface Controller.
#
#------------------------------------------------------------------------

"""Interface Controller for Photon Science Data Management.

This is the Interface Controller that monitors filesets created by the online system.
It creates a translator process to translate the fileset into HDF5 and enters the 
translated file into iRODS. rt

This software was developed for the LUSI project.  If you use all or
part of it, please give an appropriate acknowledgement.

@see RelatedModule

@version $Id$

@author Robert C. Sass
"""


#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import pwd
import subprocess
import time
import resource
import signal

#---------------------------------
#  Imports of base class module --
#---------------------------------
import threading

#-----------------------------
# Imports for other modules --
#-----------------------------
from InterfaceCtlr.FileMgrIrods import FileMgrIrods
from InterfaceCtlr.InterfaceDb import InterfaceDb
from LusiTime.Time import Time

#---------------------
# Local definitions --
#---------------------

#-------------------
# Local functions --
#-------------------

#--------------------------
# Specialized exceptions --
#--------------------------

#--------------------------------
# Application class definition --
#--------------------------------

class TranslatorThread ( threading.Thread ) :

    # ===========
    # constructor
    # ===========

    def __init__ ( self, fs, name, db, config, logger ) :

        threading.Thread.__init__ ( self, name=name )

        # other instance variables
        self._fs = fs
        self._name = name
        self._db = db
        self._translate_uri = config["_translate_uri"]
        self._log_uri = config["_log_uri"]
        self._controller_id = config["_controller_id"]
        self._config = config
        self._log = logger

        # setup file manager
        self._file_mgr = None
        irods_cmd = self._get_config('filemanager-irods-command')
        if irods_cmd :
            self._file_mgr = FileMgrIrods( irods_cmd, self._config, self._log )

    # ======================
    # Process single fileset
    # ======================

    def run ( self ) :

        fs = self._fs
        fs_id = fs['id']

        # tell everybody we are taking care of this fileset
        self._db.change_fileset_status (fs_id, 'Being_Translated')
        
        # store XTC files in file manager
        fm_xtc_dir = self._get_config('filemanager-xtc-dir')
        if self._file_mgr and fm_xtc_dir :
            for xtc in fs['xtc_files'] :
                basename = xtc.split('/')[-1]
                dst_coll = fm_xtc_dir % fs
                returncode = self._file_mgr.storeFile( xtc, dst_coll + '/' + basename )
                if returncode == 0 :
                    self._db.archive_file ( fs_id, xtc, dst_coll )

        # build all directory and file names
        fname_dict = self.__build_output_fnames(fs)

        # output directory must be empty or non-existent
        try:
            self.__make_hdf5_dir( fname_dict['tmpdirname'] )
        except Exception, exc:
            self.error("[%s] Failed to make temporary directory %s: %s", self._name, fname_dict['tmpdirname'], exc )
            self._db.change_fileset_status (fs_id, 'H5Dir_Error')
            return

        # build command line for running translator
        cmd = self.__build_translate_cmd(fs_id, fname_dict, fs)

        prev_stats = resource.getrusage(resource.RUSAGE_CHILDREN)

        # start translator
        logname = os.path.join(self._log_uri,fname_dict['logname'])
        pid = self._start ( cmd, logname )
        translator_id = self._db.new_translator(self._controller_id, fs_id, logname)
        
        self.info ("[%s] Started translator #%d (PID=%d) with cmd %s", self._name, translator_id, pid, ' '.join(cmd) )
        self.info ("[%s] output directory %s", self._name, fname_dict['tmpdirname'] )
        self.info ("[%s] Log file is in %s", self._name, logname )
        
        # loop and wait until translation done             
        xlate_done = False
        while not xlate_done:

            # check kill flag
            if self._db.test_exit_translator(translator_id):
                # kill it if asked
                self.info ("[%s] Request to kill translator_process #%d (PID=%d)", self._name, translator_id, pid)
                os.kill(pid,signal.SIGTERM)
                # it may need some time to stop
                time.sleep( 3 )
                
            wpid, wstat = os.waitpid( pid, os.WNOHANG )
            if wpid != 0:
                
                xlate_done = True

                self.info ("[%s] translator #%d finished (PID=%d) retcode=%s", self._name, translator_id, wpid, wstat)

                # get the size of resulting files
                output_size = self.__dir_size( fname_dict['tmpdirname'] )
                self.info ("[%s] translator #%d produced %d bytes of data", self._name, translator_id, output_size)

                # store statistics
                self._db.update_translator(translator_id, wstat, prev_stats, output_size )

                if wstat != 0 :
                    self.warning ("[%s] translator #%d failed", self._name, translator_id )
                    self._db.change_fileset_status (fs_id, 'Translation_Error')
                else:
                    self._db.change_fileset_status (fs_id, 'Translation_Complete')
                    
                    # store result in file manager
                    returncode = self.__store_hdf5 ( fs, fname_dict['tmpdirname'], fname_dict['h5dirname'] )
                    self._db.update_irods_status (translator_id, returncode)
                    if returncode != 0:
                        self._db.change_fileset_status (fs_id, 'Archive_Error')
                    else:
                        self._db.change_fileset_status (fs_id, 'Complete')
                            
            else:
                time.sleep(5.0)
        #end while not xlate_done
         



    # =====================
    # Build output filename
    # =====================

    def __build_output_fnames ( self, fsdbinfo ) :

        """Build the output filenames and return as a dict"""

        # current time string as '20090915T120145'
        curtime = Time.now().toString("%Y%m%dT%H%M%S")

        # re-format directory URI
        dir_name = self._translate_uri % fsdbinfo

        # temporary directory name
        tmpdirname = "%04d-%s" % ( fsdbinfo['run_number'], curtime )
        tmpdirname = os.path.join(dir_name,tmpdirname)
        
        # Now construct the output file name
        h5name = self._get_config('hdf5-file-name')
        if not h5name : h5name = "%(experiment)s-r%(run_number)04d-c{seq2}.h5"
        h5name = h5name % fsdbinfo

        logname = "o2o-translate-%(experiment)s/o2o-translate-%(experiment)s-r%(run_number)04d" % fsdbinfo

        return dict ( h5name = h5name,
                      h5dirname = dir_name,
                      tmpdirname = tmpdirname,
                      logname = logname + '-' + curtime + '.log' )

    # ===========================================================
    # Build a list that has the command to execute the translator
    # ===========================================================

    def __build_translate_cmd ( self, fileset_id, fname_dict, fs ) :

        """Build the arg list to pass to the translator from the files in fileset
        and the translate_uri destination for the translator output"""

        cmd_list = []
        cmd_list.append("o2o-translate")

        #
        # Destination dir for translated file
        cmd_list.append("--output-dir")
        cmd_list.append(fname_dict['tmpdirname'])

        #
        # experiment, run number, filename
        cmd_list.append("--instrument")
        cmd_list.append(fs['instrument'])
        cmd_list.append("--experiment")
        cmd_list.append(fs['experiment'])
        cmd_list.append("--run-number")
        cmd_list.append(str(fs['run_number']))
        cmd_list.append("--output-name")
        cmd_list.append(fname_dict['h5name'])

        # add options files
        for f in self._get_config('list:o2o-options-file',[]) :
            cmd_list.append("--options-file")
            cmd_list.append(f)

        # any extra options
        for opt in self._get_config('list:o2o-extra-options',[]) :
            cmd_list.extend( opt.split() )

        for xtc in fs['xtc_files']:
            cmd_list.append(xtc)

        return cmd_list


    # ============================================
    # Calculate the size of all files in directory
    # ============================================

    def __dir_size( self, dirname ) :

        # generator for all file paths under given directory
        def _all_files( dirname ) :
            for root, dirs, files in os.walk( dirname ) :
                for f in files :
                    yield os.path.join( root, f )

        return sum( [ os.stat(f).st_size for f in _all_files( dirname ) ] )


    # ===============================
    # make directory for output files
    # ===============================

    def __make_hdf5_dir(self, dirname) :

        # output directory must be empty or non-existent
        if os.path.exists(dirname) :
            if not os.path.isdir(dirname) :
                msg = '[%s] output directory exist but is not a directory: %s' % ( self._name, dirname )
                self.warning ( msg )
                raise IOError( msg )
            elif os.listdir(dirname) :
                msg = '[%s] output directory exist but is not empty: %s' % ( self._name, dirname )
                self.warning ( msg )
                raise IOError( msg )
        else :
            # create output directory
            self.trace ( '[%s] create directory for output files: %s', self._name, dirname )
            os.makedirs(dirname)

    # =================================================
    # Store HDF5 files in both dataset and file manager
    # =================================================

    def __store_hdf5 (self, fs, tmpdirname, dirname):

        # generator for all file paths under given directory, 
        # returns list of tuples (subdir,filename)
        def _all_files( root, subdir = "" ) :
            for e in os.listdir( os.path.join(root,subdir) ) :
                path = os.path.join(root,subdir,e)
                if os.path.isdir(path) :
                    for x in _all_files ( root, os.path.join(subdir,e) ) :
                        yield x
                else :
                    yield ( subdir, e )

        # build a list of files to store
        files = [ x for x in _all_files( tmpdirname ) ]

        # check that final destination does not have these files
        for f in files :
            dst = os.path.join( dirname, f[0], f[1] )
            if os.path.exists(dst) :
                self.error("store_hdf5: destination files already exists: %s",dst)
                return 2
            
        # move all files to final destination
        for f in files :
            src = os.path.join( tmpdirname, f[0], f[1] )
            dst = os.path.join( dirname, f[0], f[1] )
            try:
                self.debug("moving file %s ->%s", src,dst)
                os.rename(src,dst)
            except Exception, e :
                self.error("store_hdf5: failed to move file: %s -> %s", src, dst)
                self.error("store_hdf5: exception raised: %s", str(e) )
                return 2
        
        # remove temporary directory
        try :
            self.debug("removing temp dir %s", tmpdirname)
            os.rmdir( tmpdirname )
        except Exception, e :
            # non-fatal, means it may have some subdirectories
            self.error("store_hdf5: failed to remove directory %s: %s", tmpdirname, str(e) )

        # add all files to fileset
        self._db.add_files( fs['id'], 'HDF5', [os.path.join(dirname,f[0],f[1]) for f in files ] )

        # archive them
        result = 0
        fm_hdf_dir = self._get_config('filemanager-hdf5-dir')
        if fm_hdf_dir :
            
            fm_hdf_dir = fm_hdf_dir % fs
            
            for dir, basename in files :
                
                path = os.path.join ( dirname, dir, basename )
                coll = fm_hdf_dir + '/' + dir
                fres = self._file_mgr.storeFile( path, coll + '/' + basename )
                if fres != 0 : result = fres
                self._db.archive_file ( fs['id'], path, coll )

        return result
                
    # =====================================
    # Start the translator, redirect output
    # =====================================
    def _start ( self, cmd, logname ) :

        # create log directory if needed
        logdir = os.path.dirname(logname)
        if logdir and not os.path.isdir(logdir):
            try:
                os.makedirs(logdir)
            except OSError, e:
                # complain but proceed, we may have race condition here
                self.warning("Failed to create log directory: %s", str(e))

        pid = os.fork()
        if pid == 0 :
            
            # child
            
            # redirect output
            fd = os.open( logname, os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0666 )
            os.dup2 ( fd, 1 )
            os.dup2 ( fd, 2 )
            
            # execute subprocess, should throw in case of errors
            os.execvp(cmd[0], cmd)
            
        else :

            #parent
            
            return pid

    def _get_config(self, option, default = None):
        val = self._config.get(option, self._fs['instrument'], self._fs['experiment'])
        if val is None : return default
        return val
            
    #                                                                                                                                                 
    #  Logging methods                                                                                                                                
    #                                                                                                                                                 
    def debug ( self, *args, **kwargs ) : return self._log.debug ( *args, **kwargs )
    def trace ( self, *args, **kwargs ) : return self._log.trace ( *args, **kwargs )
    def info ( self, *args, **kwargs ) : return self._log.info ( *args, **kwargs )
    def warning ( self, *args, **kwargs ) : return self._log.warning ( *args, **kwargs )
    def error ( self, *args, **kwargs ) : return self._log.error ( *args, **kwargs )
    def critical ( self, *args, **kwargs ) : return self._log.critical ( *args, **kwargs )

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
