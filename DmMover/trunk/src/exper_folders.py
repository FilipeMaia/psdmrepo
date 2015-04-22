

import os


def __print_mkdir(path, value=None):
    print "MKDIR %s %o" % (path, value)

def __print_cmd(cmd):
    print "CMD", cmd

def runcmd(cmd):
    print cmd
    os.system(cmd)


pjoin = os.path.join
mkdir = os.mkdir

#mkdir = __print_mkdir
#runcmd =  __print_cmd


def check_all_destpath(exper, seldirs=None, dolink=True):
    """ Create all the experiments subdirectories (in the offline files system) and 
    create the links in the experiment directory """
    
    datapath = exper.datapath
    print  exper.name, exper.instr_lower, exper.posix_gid, datapath

    for folder in ('xtc', 'hdf5', 'usr'):
        if seldirs and folder not in seldirs:
            continue
        create_exp_path(folder, exper.name, exper.instr_lower, exper.posix_gid, datapath)
        if dolink:
            create_link(folder, exper.name, exper.instr_lower, datapath)

    datapath = exper.scratchpath
    print  "scratch", datapath
    for folder in ('scratch', 'res', 'ftc', 'calib'):
        if seldirs and folder not in seldirs:
            continue
        create_exp_path(folder, exper.name, exper.instr_lower, exper.posix_gid, datapath)
        if dolink:
            create_link(folder, exper.name, exper.instr_lower, datapath)
        

def check_destpath(exper, instr=None):
    """ Create the experiment subdirectories (in the offline files system) that are needed for the mover
    (xtc,hdf5,usr) and create the links in the experiment directory """

    if instr:
        instrument = instr.lower()
    else:
        instrument = exper.instr_lower

    print  exper.name, instrument, exper.posix_gid, exper.datapath
    for folder in ('xtc', 'hdf5', 'usr'):
        create_exp_path(folder, exper.name, instrument, exper.posix_gid, exper.datapath)
        create_link(folder, exper.name, instrument, exper.datapath)


def check_xtc_path(exper, no_instr_path=False):
    """ Create only the xtc path. On a FFB node call with no_instr_path=True
    as the instr is not part of the path.
    """
    create_exp_path('xtc', exper.name, exper.instr_lower, exper.posix_gid, exper.datapath, no_instr_path)


def check_existing_link(src, name, do_update=False):
    """ Check if link *name* points to *src*. 
    
    With do_update=True update the link to *src*. Otherwise 
    only print the link status.
    """
    
    curr_src = os.readlink(name)
    if curr_src != src:
        print "Warning link mismatch", curr_src, src
        if do_update:
            tmp_name = "%s.tmp" % name
            os.symlink(src, tmp_name) 
            os.rename(tmp_name, name)


def create_link(datatype, exp, instr, physpath, do_run=True, force=False):
    """ Create a link in the canonical experiment folder (/reg/d/psdm/<instr>/<exp>)
    that points to the folder in the offline FS.
    e.g.: /reg/d/psdm/cxi/cxidaq13/xtc -> /reg/data/ana11/cxi/cxidaq13/xtc
    """
    
    link_src = pjoin(physpath, instr, exp, datatype)
    link_name = pjoin("/reg/d/psdm", instr, exp, datatype)

    # First check that the experiment path exists
    cpath = os.path.dirname(link_name) 
    print "Check link:", link_name, link_src, cpath

    if not os.path.exists(cpath):
        if do_run:
            mkdir(cpath, 02755)
            
    if not os.path.exists(link_src):
        print "Warning: Link src is missing", link_src

    if os.path.exists(link_name):
        print "Link exists", link_name
        check_existing_link(link_src, link_name, do_run and force)
    else:
        if do_run:
            print "Create link", link_name, "->", link_src
            os.symlink(link_src, link_name)
        else:
            print "Testonly: Create link", link_name, "->", link_src


def create_exp_path(datatype, exp, instr, posix_grp, physpath, no_instr_path=False):
    """ Create an experiments directory and its sub directory for a datatype.
    
    The datatype = (xtc,hdf5,usr,scratch,ftc,res,calib)
    with no_instr_path the instr name is not used for the experiments directory.
    """
    
    group = posix_grp
    instr = instr.lower()

    if no_instr_path:
        exppath = pjoin(physpath, exp)
    else:
        exppath = pjoin(physpath, instr, exp)

    # create the experiment path and set the proper acls that get
    # inherited by a sub folder
    
    if not os.path.exists(exppath):
        print "Create %s" %(exppath)
        mkdir(exppath, 02750)
        runcmd('chgrp ps-data %s' % exppath)
        # set permissions again as chgrp will remove the sgid bit
        os.chmod(exppath, 02750)
        
        runcmd('setfacl -d -m group:ps-data:rx %s' %(exppath))
        if instr == 'dia':
            runcmd('setfacl -d -m group:ps-users:rx %s' % exppath)
            runcmd('setfacl -m group:ps-users:rx %s' % exppath)
        else:
            runcmd('setfacl -d -m group:ps-%s:rx %s' %(instr, exppath))
            runcmd('setfacl -d -m group:%s:rx %s' %(group, exppath))
            runcmd('setfacl -m group:%s:rx %s' %(group, exppath))
            runcmd('setfacl -m group:ps-%s:rx %s' %(instr, exppath))

        if instr == 'xpp' :
            runcmd('setfacl -m group:ps-xcs:rx %s' % exppath)
            runcmd('setfacl -d -m group:ps-xcs:rx %s' % exppath)
        elif instr == 'xcs' :
            runcmd('setfacl -m group:ps-xpp:rx %s' % exppath)
            runcmd('setfacl -d -m group:ps-xpp:rx %s' % exppath)


    # create the sub-folder

    datapath = pjoin(exppath, datatype)
    if os.path.exists(datapath):
        return 0

    if datatype in ('xtc', 'hdf5', 'usr'):
        print "Create %s" %(datapath)
        mkdir(datapath, 02750)
        runcmd('setfacl -d -m user::rwx %s' %(datapath))
        
        if datatype == 'xtc':
            for subdir in ('md5', 'index', 'smalldata', 'smalldata/md5'):
                spath = pjoin(datapath, subdir)
                print "Create %s" %(spath)
                mkdir(spath, 02750)
                runcmd('setfacl -d -m user::r %s' %(spath))
                runcmd('setfacl -m user::rwx %s' %(spath))
    
    if datatype in ('scratch', 'ftc', 'res'): 
        print "Create %s" %(datapath)
        mkdir(datapath, 0770)
        runcmd('setfacl -d -m group:%s:rwx %s' %(group, datapath))
        runcmd('setfacl -m group:%s:rwx %s' %(group, datapath))
        runcmd('setfacl -d -m group:ps-%s:rwx %s' %(instr, datapath))
        runcmd('setfacl -m group:ps-%s:rwx %s' %(instr, datapath))

    if datatype in ('calib', ):
        print "Create %s" %(datapath)
        mkdir(datapath, 0770)
        runcmd('setfacl -d -m group:ps-data:rwx %s' % (datapath))
        runcmd('setfacl    -m group:ps-data:rwx %s' %(datapath))
        runcmd('setfacl -d -m group:ps-%s:rwx %s' %(instr, datapath))
        runcmd('setfacl    -m group:ps-%s:rwx %s' %(instr, datapath))

        runcmd('setfacl -d -m group:%s:rwx %s' %(group, datapath))
        runcmd('setfacl    -m group:%s:rwx %s' %(group, datapath))

        # for calib folders give <instr>opr account read access
        # requires that the expr directory at least allows 'execute' 
        # for the opr account

        runcmd('setfacl    -m user:%sopr:rx %s' %(instr, exppath))        
        runcmd('setfacl -d -m user:%sopr:rx %s' %(instr, datapath))
        runcmd('setfacl    -m user:%sopr:rx %s' %(instr, datapath))
