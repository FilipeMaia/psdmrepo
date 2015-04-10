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
import subprocess # for subprocess.Popen
from commands import getoutput
import tempfile
from time import time, sleep

from CalibManager.SvnProps   import svnprops as spcm
from ImgAlgos.SvnProps       import svnprops as spia
from PSCalib.SvnProps        import svnprops as spps
from pdscalibdata.SvnProps   import svnprops as spcd
from CSPadPixCoords.SvnProps import svnprops as sppc

from CalibManager.SvnPropsViewer import SvnPropsViewer

from CalibManager.GlobalUtils import ready_to_start

#from ConfigParametersForApp import cp
#import GlobalUtils          as     gu
#------------------------------

def subproc_submit(command_seq, logname=None, env=None, shell=False) :
    # for example, command_seq=['bsub', '-q', cp.batch_queue, '-o', 'log-ls.txt', 'ls -l']
    log = subprocess.PIPE if logname is None else open(logname, 'w')
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
    def __init__(self, list_of_pkgs = ['CalibManager', 'ImgAlgos', 'PSCalib', 'pdscalibdata', 'CSPadPixCoords'], print_bits=0) :
        t0_sec = time()
        self.list_of_pkgs = list_of_pkgs
        self.print_bits = print_bits
        self.delete_old_tmp_files()
        if self.print_bits & 1 : print 'PackageVersions: Old temporary files are deleted'
        self.make_logfiles_in_background_mode()
        if self.print_bits & 2 : print 'PackageVersions: make_logfiles_in_background_mode() is started'

        #msg = 'Consumed time to launch subprocesses = %7.3f sec' % (time()-t0_sec)
        #print msg


    def delete_old_tmp_files(self) :
        cmd = 'rm -f /tmp/calibman-*txt'
        try :
            #self.subproc_submit(cmd.split())
            output = getoutput(cmd)
            if self.print_bits & 4 : print 'delete_old_tmp_files(...):\n', output
            #stream = os.popen(cmd)
            #print stream.read()
        except :
            if self.print_bits & 4 : print 'delete_old_tmp_files(...):\nSome problem with cleanup tmp files "%s"' % cmd


    def make_logfiles_in_background_mode(self) :
        """Returns dictionary with temporary file names for packages as keys"""

        self.dict_pkg_fname = {pkg:get_tempfname() for pkg in self.list_of_pkgs}
        for pkg, fname in self.dict_pkg_fname.iteritems() :
            cmd = 'psvn tags %s' % pkg
            subproc_submit(cmd.split(), fname)


    def print_list_of_packages(self) :
        for pkg, fname in self.dict_pkg_fname.iteritems() :
            print pkg, fname


    def get_text_from_log_for_pkg(self, pkg) :
        fname = self.dict_pkg_fname[pkg]
        return get_text_from_file(fname)


    def print_log_for_pkg(self, pkg) :
        print self.get_text_from_log_for_pkg(pkg)


    def get_revision_str(self, props) :
        spv = SvnPropsViewer(props)
        return spv.get_pkg_revision()


    def get_revision_msg(self, props) :
        spv = SvnPropsViewer(props)
        return 'RN-%s' % spv.get_pkg_revision()


    def get_pkg_revision(self, pkg='CalibManager') :
        """Returns package revision number"""
        if   pkg=='CalibManager'   : return self.get_revision_str(spcm)
        elif pkg=='ImgAlgos'       : return self.get_revision_str(spia)
        elif pkg=='PSCalib'        : return self.get_revision_str(spps)
        elif pkg=='pdscalibdata'   : return self.get_revision_str(spcd)
        elif pkg=='CSPadPixCoords' : return self.get_revision_str(sppc)

        #if   pkg=='CalibManager'   : return self.get_pkg_tag(pkg)
        #elif pkg=='ImgAlgos'       : return self.get_pkg_tag(pkg)
        #elif pkg=='PSCalib'        : return self.get_pkg_tag(pkg)
        #elif pkg=='pdscalibdata'   : return self.get_pkg_tag(pkg)
        #elif pkg=='CSPadPixCoords' : return self.get_pkg_tag(pkg)

        else : return 'N/A' 


    def get_pkg_tag(self, pkg='CalibManager') :
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
            return 'V-is-N/A'


    def get_pkg_version(self, pkg='CalibManager') :
        """Returns the latest version of the package"""
        return self.get_pkg_revision(pkg)
        #return self.get_pkg_tag(pkg)


    def get_tag_for_rev(self, pkg='CalibManager', rev_inq='8050') :
        """Returns the package tag for """
        try :
            rev_num = int(rev_inq)
            #print 'rev_num:',  rev_num
            output = self.get_text_from_log_for_pkg(pkg).rstrip('\n')
            lines = output.split('\n')
            for line in lines[1:] :
                #print 'line:', line
                fields = line.split()
                rev = fields[0]                
                tag = fields[-1].rstrip('/')
                #print 'rev, tag =', rev, tag
                if int(rev) >= rev_num :
                    return tag            

        except :
            return 'V-is-N/A'
        

    def text_version_for_all_packages(self) :
        txt = 'Version of packages'
        for pkg in self.list_of_pkgs :
            txt += '\n    %s  %s' % (self.get_pkg_version(pkg), pkg)
        return txt


    def text_rev_and_tag_for_all_packages(self) :
        txt = 'Revision and tag of packages'
        for pkg in self.list_of_pkgs :            
            rev = self.get_pkg_revision(pkg)
            tag = self.get_tag_for_rev(pkg,rev)
            txt += '\n    %s  %s  %s' % (rev, tag, pkg)
            #print 'test of rev, tag, pkg:', rev, tag, pkg
        return txt


#------------------------------

def test_packege_version(test_num):

    print 'Test: %d' % test_num

    if not ready_to_start(check_bits=02, fatal_bits=02) : #1,2,4 = old LUSTRE, Kerberos ticket,  AFS token
        sys.exit('Not ready to start aplication yet...')

    pv = PackageVersions(print_bits=0377)

    t0_sec = time()

    if test_num == 0 :
        pv.print_list_of_packages()
    else : 
        sleep(4)
        t0_sec = time()

    if test_num == 1 :
        pv.print_log_for_pkg('CalibManager')

    elif test_num == 2 :
        pkg = 'CalibManager'
        print 'Package %s, version: %s' % (pkg, pv.get_pkg_version(pkg))
        
    elif test_num == 3 :
        print pv.text_version_for_all_packages()

    elif test_num == 4 :
        pkg = 'CalibManager'
        rev = pv.get_pkg_revision(pkg)
        tag = pv.get_tag_for_rev(pkg,rev)
        print 'Package %s, revision: %s, tag: %s' % (pkg, rev, tag)

    elif test_num == 5 :
        pkg = 'pdscalibdata'
        rev = pv.get_pkg_revision(pkg)
        tag = pv.get_tag_for_rev(pkg,rev)
        print 'Package %s, revision: %s, tag: %s' % (pkg, rev, tag)
        
    elif test_num == 6 :
        pkg = 'CalibManager'
        print 'Package %s latest tag: %s' % (pkg, pv.get_pkg_tag(pkg))

    elif test_num == 7 :
        txt = pv.text_rev_and_tag_for_all_packages()
        print txt


    msg = 'Consumed time to test method = %7.3f sec' % (time()-t0_sec)
    print msg

#------------------------------

if __name__ == "__main__" :

    if len(sys.argv)!=2 or sys.argv[1] == '-h' :
        msg  = 'NOTE!!! psvn command needs in Kerberos ticket !!!'
        msg += '\nUse %s with a single parameter, <test number=0,1,2,...>' % sys.argv[0]
        msg += '\n    0 - print_list_of_packages() and associated tmp files'
        msg += '\n    1 - print_log_for_pkg("CalibManager")'
        msg += '\n    2 - get_pkg_version("CalibManager")'
        msg += '\n    3 - text_version_for_all_packages()'
        msg += '\n    4 - get_pkg_revision("CalibManager")'
        msg += '\n    5 - get_pkg_revision("pdscalibdata")'
        msg += '\n    6 - get_pkg_tag("CalibManager")'
        msg += '\n    7 - text_rev_and_tag_for_all_packages()'
        msg += '\n   -h - this help'
        print msg

    else :

        try    :
            test_num = int(sys.argv[1])
            test_packege_version(test_num)
        except :
            test_packege_version(0)

    sys.exit ( 'End of test.' )

#------------------------------
