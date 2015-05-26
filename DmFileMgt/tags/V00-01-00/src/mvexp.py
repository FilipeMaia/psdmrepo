

import os
import sys
import glob
import time
import logging
import os.path as op

from DmFileMgt.checksum import cmp_md5


trans_ext = ".transdecomp"

class MvExpDir(object):
    def __init__(self, instr, exp, mvdir, ana_old, ana_new):
        """
        >>> exp = MvExpDir('xpp', 'xpp12345', 'xtc', 'ana01', 'ana02'
        pdn: physical directory name
        """
        
        self.instr, self.exp = instr, exp
        self.dirtype = mvdir

        self.exppath = op.join("/reg/d/psdm/", instr, exp)

        self.dirlink = op.join(self.exppath, mvdir)
        self.pdn_old_orig = op.join("/reg/data", ana_old, instr, exp, mvdir)
        self.pdn_old_moved = op.join("/reg/data", ana_old, instr, exp,"moved_%s" %  mvdir)
        
        if op.exists(self.pdn_old_orig):
            self.pdn_old = self.pdn_old_orig
        elif op.exists(self.pdn_old_moved):
            self.pdn_old = self.pdn_old_moved
        else:
            raise IOError("old directory does not exists")

        self.pdn_new = op.join("/reg/data", ana_new, instr, exp, mvdir)


    def show(self):
        
        print "== instr: %s  exper: %s  dir: %s  link: %s (updated: %s) exppath: %s" % (
            self.instr, self.exp, self.dirtype, self.dirlink, self.link_updated(), self.exppath)

        print "  old phys path", self.pdn_old, "  new phys path", self.pdn_new

    def create_new(self):
        """ create new directory """
         
        old_expdir = os.path.dirname(self.pdn_old)
        new_expdir = os.path.dirname(self.pdn_new)

        if not os.path.exists(new_expdir):
            os.mkdir(new_expdir)
            rc = os.system("getfacl -p %s | setfacl --set-file=- %s" % (old_expdir, new_expdir))
            print "create", new_expdir, rc

        if not os.path.exists(self.pdn_new):
            os.mkdir(self.pdn_new)
            rc = os.system("getfacl -p %s | setfacl --set-file=- %s" % (self.pdn_old, self.pdn_new))
            print "create", self.pdn_new, rc

    def update_symlink(self):
        """ update the canonical experiment subfolder to point to the new file system """
    
        if self.link_updated():
            print "Link already updated", self.dirlink, self.pdn_new
        else:
            tmp_linkname = self.dirlink + ".tmp_mv_link"
            os.symlink(self.pdn_new, tmp_linkname)
            os.rename(tmp_linkname, self.dirlink)
            logging.info("Link: %s %s", self.pdn_new, self.dirlink)
            
    def link_updated(self):
        curr_link = os.readlink(self.dirlink)
        return curr_link == self.pdn_new

    def rename_to_moved(self):
        """ rename the old directry """
        if self.link_updated() and self.pdn_old_orig == self.pdn_old:
            print "rename from ", self.pdn_old, "to", self.pdn_old_moved
            os.rename(self.pdn_old, self.pdn_old_moved)
            self.pdn_old = self.pdn_old_moved
            self.show()


class MoveExper:

    def __init__(self, expdir):
        # Experiment dir: MvExpDir() 
        self.expdir = expdir  
        self.do_md5 = False
        self.do_md5_clean = False
        self.rm_src = False

    def transfn(self, fn):
        """ return the transferred filename before a file is moved inplace """
        return fn + trans_ext
        
    def rsync_subdirs(self, subdirs, rm_src=False):
        """ rsync a sub directory to the new location 
        e.g.:  rsync <dir>/subdir <newdir>/.

        If subdirs is (None,) rsynv old/ new/.
        """

        logging.info("Rsync") 
        ropt = "--remove-source-files -Xav" if rm_src else "-Xav"
        for sub in subdirs:
            if sub:
                src = op.join(self.expdir.pdn_old, sub.rstrip('/'))
            else:
                src = self.expdir.pdn_old + '/'

            if not op.exists(src) or not self.expdir.pdn_new:
                logging.warning("rsync: Missing dir %s or %s", src, self.expdir.pdn_new)
                return False
            
            cmd = "rsync %s %s %s/." % (ropt, src, self.expdir.pdn_new)
            rc = os.system(cmd)            
            if rm_src and sub: 
                try:
                    os.rmdir(op.join(self.expdir.pdn_old, sub))
                except OSError as ioerr:
                    logging.error("Failed removing dir %s", self.expdir.pdn_old, sub)
                    
            print cmd
            print "Transferred", rc, src, self.expdir.pdn_new
            
 
    def for_files_in_old(self, fn): 
        """Find files in old location and for each call fn(oldfn, newfn) """

        if self.expdir.dirtype == 'xtc':
            select = 'e*.xtc'
        elif self.expdir.dirtype == 'hdf5':
            select = '*.h5'
        else:
            print "Wrong dirtype extension", expinfo.dirtype
            return

        for srcfn in glob.glob("%s/%s" % (self.expdir.pdn_old, select)):
            trgfn = op.join(self.expdir.pdn_new, op.basename(srcfn))            
            fn(srcfn, trgfn)


    def fn_make_link(self, srcfn, trgfn):
        if not op.exists(trgfn):
            #os.symlink(srcfn,trgfn)
            logging.info("Create link %s -> %s", trgfn, srcfn) 
        else:
            logging.debug("File exists %s", trgfn)

    def do_link(self):
        self.do_sync(linkonly=True)

    def do_sync(self, rm_src=False, linkonly=False):
        """Sync files from old to new location. rm if requested """

        logging.info("do_sync")
        self.rm_src = rm_src
        if rm_src and not self.expdir.link_updated():
            logging.error("Link has not been updated, no cleanup")
            self.rm_src = False
            return False

        # directories have already been created
        if self.expdir.dirtype == 'xtc':
            self.rsync_subdirs(('index', 'md5'), self.rm_src)

        if self.expdir.dirtype == 'usr':
            self.rsync_subdirs((None,), self.rm_src)

        if self.expdir.dirtype in ('xtc', 'hdf5'):
            if linkonly:
                self.for_files_in_old(self.link_files)
            else:
                self.for_files_in_old(self.copy_clean)

    
    # functions that can be called for each old/new-fn pair    
    def fn_transfer(self, srcfn, trgfn):
        """ Transfer a file. """ 
        pass

    def file_status(self, srcfn, trgfn):
        """ Check status of a file in the new file-system

        return: inplace, transfered, onlylink
        other wise raise IOError
        """

        fstatus = None
        # check if file exists or is link
        if os.path.islink(trgfn):
            link_src = os.readlink(trgfn)
            if link_src != srcfn:
                raise IOError("Link error")
            fstatus = "onlylink"
        elif op.isfile(trgfn):
            if op.getsize(trgfn) != op.getsize(srcfn):
                raise IOError("Size mismatch")
            fstatus = "inplace"
        else:
            fstatus = "missing"

        # tmp transferred file exists 
        trans_trgfn = self.transfn(trgfn)
        if op.exists(trans_trgfn):
            if op.getsize(trans_trgfn) != op.getsize(srcfn):
                raise IOError("Size mismatch")
            transferred_stat = True
        else:
            transferred_stat = False

        return (fstatus, transferred_stat)
        
    def link_files(self, srcfn, trgfn):
        """ link files """

        file_stat, trans_stat = self.file_status(srcfn, trgfn)

        logging.debug("link file: stat %s %s %s", file_stat, trans_stat, srcfn)
        if file_stat == "missing":
            os.symlink(srcfn, trgfn)
            logging.debug("linked %s %s", srcfn, trgfn) 

    def copy_clean(self,srcfn, trgfn):
        """ copy, link files and cleanup, called for each file """

        file_stat, trans_stat = self.file_status(srcfn, trgfn)

        logging.debug("copy_clean: stat %s %s %s", file_stat, trans_stat, srcfn)
        if file_stat == "missing":
            os.symlink(srcfn,trgfn)
            logging.debug("linked %s %s",srcfn,trgfn) 
        elif file_stat == "inplace":
            logging.info("Cleanup old file rm=%s %s", self.rm_src, srcfn)
            if self.rm_src:
                self.remove_src(srcfn, trgfn)

        if file_stat != "inplace":
            if trans_stat:
                # file has been transferred to tmp name
                self.move_to_inplace(srcfn, trgfn)
                logging.debug("move inplace %s", trgfn)
            else:
                # transfer file and rm
                self.transfer_file(srcfn, trgfn)
                if self.rm_src:
                    self.move_to_inplace(srcfn, trgfn)
                    self.remove_src(srcfn, trgfn)

    def transfer_file(self, srcfn, trgfn):
        """ Transfer a file. Transfer to temp transfer name """ 
        trans_fn = self.transfn(trgfn)
        trans_tmp = trans_fn + time.strftime("_%Y%m%dT%H%M%S", time.localtime())
        cmd = "bbcp -p -P 10 -v -s 1 %s %s" % (srcfn,trans_tmp)
        logging.debug("Transfer: %s", cmd)
        rc = os.system(cmd)
        if rc == 0:
            os.rename(trans_tmp, trans_fn)
        else:
            raise IOError("File transfer failed", trans_tmp)
            
    def move_to_inplace(self, srcfn, trgfn):
        """ Move the transferred file into place """
        
        trans_fn = self.transfn(trgfn)
        # check size link points to
        if op.getsize(trans_fn) != op.getsize(srcfn):
            logging.error("rename transfn to trgfn failed, size mismatch")
            raise IOError("Size Mismatch")
        if self.do_md5 and not cmp_md5(srcfn, trgfn):
            raise IOError("Checksum  mismatch")

        os.rename(trans_fn, trgfn)

    def remove_src(self, srcfn, trgfn):
        """ remove src file """

        file_stat, trans_stat = self.file_status(srcfn, trgfn)
        if self.do_md5_clean and not cmp_md5(srcfn, trgfn):
            raise IOError("Checksum  mismatch")

        if file_stat == "inplace":
            os.remove(srcfn)

