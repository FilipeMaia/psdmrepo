
from os.path import join as pjoin


class Experiment(object):
    """ Parameter for an experiment 

    Parameters:   id, name, datapath, instr, posix_gid, ....

    datapath: path up to the instrument folder, e.g.: for 
              /reg/data/ana12/cxi/cxit1234 the datapath is /reg/data/ana12

    >>> exp = Experiment(name="cxidaq13")
    >>> print exp.datapath, exp.instr, exp.name 
    """

    def __init__(self, dbsrc=None, name=None, expid=None):
        
        if not dbsrc:
            import DmMover.db_access as db
            src = db
        else:
            src = dbsrc

        if name:
            expID = src.name2id(name)
        elif expid:
            expID = expid
        else:
            self.info = {}
            return 

        self._scratchpath = None
        self._datapath = src.getexp_datapath(expID)
        if not self._datapath:
            raise ValueError("experiment data path not set") 

        self.info = src.getexp(expID)
        self.instr = src.instr4id(expID)
        self.instr_lower = self.instr.lower()

    def __getattr__(self, attr):
        if attr in self.info:
            return self.info[attr]
        elif attr == 'eid' and 'id' in self.info:
            return self.info['id']
        else:
            raise AttributeError(attr)

    @property
    def datapath(self):
        return self._datapath

    @datapath.setter
    def datapath(self, value):
        self._datapath = value
        
    @property
    def scratchpath(self):
        if self.instr_lower in ("cxi", "mec", "xcs", "mob"):
            return self._scratchpath if self._scratchpath else "/reg/data/ana14" 
        elif self.instr_lower in ("amo", "sxr", "xpp", "dia", "usr"):
            return self._scratchpath if self._scratchpath else "/reg/data/ana04" 

        raise NameError("Could not get scratch anapath for %s" % instr)

    @scratchpath.setter
    def scratchpath(self, value):
        self._scratchpath = value
        

class ExperimentInfo(object):
    """ Parameters and path-names for an experiment
 
    >>> exp = ExperimentInfo(....)
    >>> print(exper.name, exper.instr)
    >>> print(eper.xtcpath, exper.hdf5path)
    """ 

    def __init__(self, info, datapath, instrument, no_instr_path=False):
        """ instrument should be upper case name """
        self.info = info
        self.instr = instrument
        self.instr_lower = instrument.lower()

        self.datapath = datapath
        self.no_instrpath = no_instr_path
        if no_instr_path:
            self.exppath = exppath = pjoin(datapath, self.name)
        else:
            self.exppath = exppath = pjoin(datapath, instrument.lower(), self.name)

        self.xtcpath = pjoin(exppath, 'xtc')
        self.md5path = pjoin(exppath, 'xtc', 'md5')
        self.indexpath = pjoin(exppath, 'xtc', 'index')
        self.hdf5path = pjoin(exppath, 'hdf5')
        self.usrpath = pjoin(exppath, 'usr')

        # status if experiment path have been checked/created
        self._check_path = True

    def __getattr__(self, attr):
        if attr in self.info:
            return self.info[attr]
        else:
            raise AttributeError(attr)

    def need_path_check(self):
        return self._check_path

    def checked_path(self):
        self._check_path = False
        
    @property
    def linkpath(self):
        return pjoin('/reg/d/psdm', self.instr, self.name)        

    @property
    def scratchpath(self): 
        return pjoin(self.exppath, 'scratch')

    @property
    def calibpath(self):
        return pjoin(self.exppath, 'calib')
    
    @property
    def smd_xtcpath(self):
        return pjoin(self.exppath, 'xtc', 'smalldata')

    @property
    def smd_md5path(self):
        return pjoin(self.exppath, 'xtc', 'smalldata','md5')
        
