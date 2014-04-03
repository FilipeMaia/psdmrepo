#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PackageVersions.py...
#
#------------------------------------------------------------------------

"""
Class PackageVersions launches subprocesses for slow command "psvn tags <pkg-name>"
in background mode and saves results in log-files for list of package names.
Access methods parse log-files and quickly extract the package version when needed.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@see 

@version $Id$

@author Mikhail S. Dubrovin
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import subprocess # for subprocess.Popen
from commands import getoutput
import tempfile
from time import time, sleep

#from ConfigParametersForApp import cp
#import GlobalUtils          as     gu
#------------------------------

def subproc_in_log(command_seq, logname, env=None, shell=False) :
    # for example, command_seq=['bsub', '-q', cp.batch_queue, '-o', 'log-ls.txt', 'ls -l']
    log = open(logname, 'w')
    p = subprocess.Popen(command_seq, stdout=log, stderr=subprocess.PIPE, env=env, shell=shell) #, stdin=subprocess.STDIN
    #p.wait()
    #err = p.stderr.read() # reads entire file
    #return err

def get_text_from_file(path) :
    f=open(path,'r')
    text = f.read()
    f.close()
    return text

def get_tempfname(mode='w+b',prefix='calibman-',suffix='.txt') :
    return tempfile.NamedTemporaryFile(mode=mode,prefix=prefix,suffix=suffix).name

#------------------------------

class PackageVersions :
    """Get info about tags of packages involved in the project
    """
    def __init__(self, list_of_pkgs = ['CalibManager', 'ImgAlgos', 'PSCalib', 'pdscalibdata', 'CSPadPixCoords']) :
        t0_sec = time()
        self.list_of_pkgs = list_of_pkgs
        self.delete_old_tmp_files()        
        self.make_logfiles_in_background_mode()

        #msg = 'Consumed time to launch subprocesses = %7.3f sec' % (time()-t0_sec)
        #print msg


    def delete_old_tmp_files(self) :
        cmd = 'rm /tmp/calibman-*txt'
        try : 
            output = getoutput(cmd)
            #print output
            #stream = os.popen(cmd)
            #print stream.read()
        except :
            pass # print 'Some problem with cleanup tmp files "rm /tmp/calibman-*txt" '


    def make_logfiles_in_background_mode(self) :
        """Returns dictionary with temporary file names for packages as keys"""

        self.dict_pkg_fname = {pkg:get_tempfname() for pkg in self.list_of_pkgs}
        for pkg, fname in self.dict_pkg_fname.iteritems() :
            cmd = 'psvn tags %s' % pkg
            subproc_in_log(cmd.split(), fname)


    def print_list_of_packages(self) :
        for pkg, fname in self.dict_pkg_fname.iteritems() :
            print pkg, fname


    def get_text_from_log_for_pkg(self, pkg) :
        fname = self.dict_pkg_fname[pkg]
        return get_text_from_file(fname)


    def print_log_for_pkg(self, pkg) :
        print self.get_text_from_log_for_pkg(pkg)


    def get_pkg_version(self, pkg='CalibManager') :
        """Returns the latest version of the package"""
        try :
            output = self.get_text_from_log_for_pkg(pkg).rstrip('\n')
            lines = output.split('\n')
            last_line = lines[-1]
            fields = last_line.split()
            version = fields[-1].rstrip('/')
            #print 'output:\n', output         
            #print 'Last line: ', last_line
            #print 'Version: ', version
            return version
        except :
            return 'V is N/A'


    def text_version_for_all_packages(self) :
        txt = 'Version of packages'
        for pkg in self.list_of_pkgs :
            txt += '\n    %s  %s' % (self.get_pkg_version(pkg), pkg)
        return txt

#------------------------------

def test_packege_version(test_num):

    print 'Test: %d' % test_num

    pv = PackageVersions()

    t0_sec = time()

    if test_num == 0 :
        pv.print_list_of_packages()
    else : 
        sleep(5)
        t0_sec = time()

    if test_num == 1 :
        pv.print_log_for_pkg('CalibManager')

    elif test_num == 2 :
        pkg = 'CalibManager'
        print 'Package %s version: %s' % (pkg, pv.get_pkg_version(pkg))
        
    elif test_num == 3 :
        print pv.text_version_for_all_packages()

    msg = 'Consumed time to test method = %7.3f sec' % (time()-t0_sec)
    print msg

#------------------------------

if __name__ == "__main__" :

    if len(sys.argv)==2 and sys.argv[1] == '-h' :
        msg  = 'Use %s with a single parameter, <test number=0-3>' % sys.argv[0]
        msg += '\n    0 - print_list_of_packages()'
        msg += '\n    1 - print_log_for_pkg("CalibManager")'
        msg += '\n    2 - get_pkg_version("CalibManager")'
        msg += '\n    3 - text_version_for_all_packages()'
        print msg

    else :

        try    :
            test_num = int(sys.argv[1])
            test_packege_version(test_num)
        except :
            test_packege_version(0)

    sys.exit ( 'End of test.' )

#------------------------------
