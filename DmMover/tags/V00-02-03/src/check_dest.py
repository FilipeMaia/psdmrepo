

import os
import re


def __print_mkdir(path, value):
    print "MKDIR %s %o" % (path, value)

def __print_cmd(cmd):
    print "CMD", cmd

def runcmd(cmd):
    print cmd
    os.system(cmd)


def check_destpath(experiment, instr, ffb=False):
    """
    Create all directories that are expected for and experiment in the
    offline storage. Set the proper acl's.
    
    Returns path to xtc directory
    
    1) create experiment path and set acl's.
    2) Creates and sets acl's for the subdirectories in exppath
    (xtc,hdf5, index, md5, scratch, calib)
    """

    mkdir = os.mkdir

    instrument = instr.lower()
    exppath = experiment.exppath
    xtcpath = experiment.xtcpath
    group = experiment.posix_gid
    expname = experiment.name
    if re.match('\w\w\w\d\d\d\d\d', expname) is not None:
        owner = expname
    else:
        owner = None

    print "Check dest-path", instrument, expname, group, owner, exppath, xtcpath
    
    if not os.path.isdir(exppath):
        print "Create %s" %(exppath)
        mkdir(exppath, 0750)
        runcmd('chgrp ps-data %s' % exppath)
        os.chmod(exppath, 02750)

        runcmd('setfacl -d -m group:ps-data:rx %s' %(exppath))
        if instrument == 'dia':
            runcmd('setfacl -d -m group:ps-users:rx %s' % exppath)
            runcmd('setfacl -m group:ps-users:rx %s' % exppath)
        else:
            runcmd('setfacl -d -m group:ps-%s:rx %s' %(instrument, exppath))
            runcmd('setfacl -d -m group:%s:rx %s' %(group, exppath))
            runcmd('setfacl -m group:%s:rx %s' %(group, exppath))
            runcmd('setfacl -m group:ps-%s:rx %s' %(instrument, exppath))

        if instrument == 'xpp' :
            runcmd('setfacl -m group:ps-xcs:rx %s' % exppath)
            runcmd('setfacl -d -m group:ps-xcs:rx %s' % exppath)
        elif instrument == 'xcs' :
            runcmd('setfacl -m group:ps-xpp:rx %s' % exppath)
            runcmd('setfacl -d -m group:ps-xpp:rx %s' % exppath)
        
    if not os.path.isdir(xtcpath):
        print "Create %s" %(xtcpath)
        mkdir(xtcpath, 0750)
        runcmd('setfacl -d -m user::rwx %s' %(xtcpath))
    
    idxpath = experiment.indexpath
    if not os.path.isdir(idxpath):
        print "Create %s" %(idxpath)
        mkdir(idxpath, 0750)
        runcmd('setfacl -d -m user::r %s' %(idxpath))
        runcmd('setfacl -m user::rwx %s' %(idxpath))

    md5path = experiment.md5path
    if not os.path.isdir(md5path):
        print "Create %s" %(md5path)
        mkdir(md5path, 0750)
        runcmd('setfacl -d -m user::r %s' %(md5path))
        runcmd('setfacl -m user::rwx %s' %(md5path))

    # done now if ffb transfer
    if ffb:
        return

    hdf5path = experiment.hdf5path
    if not ffb and not os.path.isdir(hdf5path):
        print "Create %s" %(hdf5path)
        mkdir(hdf5path, 0750)
        runcmd('setfacl -d -m user::rwx %s' %(hdf5path))

    usrpath = experiment.usrpath
    if not ffb and not os.path.isdir(usrpath):
        print "Create %s" %(usrpath)
        mkdir(usrpath, 0750)
        runcmd('setfacl -d -m user::rwx %s' %(usrpath))

    scratchpath = experiment.scratchpath
    if not os.path.isdir(scratchpath):
        print "Create %s" %(scratchpath)
        mkdir(scratchpath, 0770)
        runcmd('setfacl -d -m group:%s:rwx %s' %(group, scratchpath))
        runcmd('setfacl -m group:%s:rwx %s' %(group, scratchpath))
        runcmd('setfacl -d -m group:ps-%s:rwx %s' %(instrument, scratchpath))
        runcmd('setfacl -m group:ps-%s:rwx %s' %(instrument, scratchpath))

    if instrument == 'cxi':
        calibpath = experiment.calibpath
        if not os.path.isdir(calibpath):
            print "Create %s" %(calibpath)
            mkdir(calibpath, 0770)
            runcmd('setfacl -d -m group:ps-data:rwx %s' %(calibpath))
            runcmd('setfacl -d -m group:ps-users:rx %s' %(calibpath))
            runcmd('setfacl -d -m group:ps-%s:rwx %s' %(instrument, calibpath))
            runcmd('setfacl -d -m group:%s:rwx %s' %(group, calibpath))
            runcmd('setfacl -m group:ps-data:rwx %s' %(calibpath))
            runcmd('setfacl -m group:ps-users:rx %s' %(calibpath))
            runcmd('setfacl -m group:ps-%s:rwx %s' %(instrument, calibpath))
            runcmd('setfacl -m group:%s:rwx %s' %(group, calibpath))
            # to allow access to calib, no -d)
            runcmd('setfacl -m group:ps-users:rx %s' %(exppath)) 
            
