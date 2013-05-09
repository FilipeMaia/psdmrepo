#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module icdb_model...
#
#------------------------------------------------------------------------

"""Model class for Interface Controller database.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import codecs

#---------------------------------
#  Imports of base class module --
#---------------------------------
from InterfaceCtlr.InterfaceDb import InterfaceDb

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons import config
from DbTools.DbConnection import DbConnection
from LusiTime.Time import Time
from RegDB.RegDb import RegDb

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------


#---------------------
#  Class definition --
#---------------------
class IcdbModel ( InterfaceDb ) :
    
    def __init__ ( self ) :
        
        # some parameters come from the configuration
        app_conf = config['app_conf']
        conn_str = app_conf.get( 'icdb.conn_str', '' )
        self._conn = DbConnection(conn_string=conn_str)

    def controller_status(self, id):
        """Returns the controller status as a list dictionaries"""
        icdb = InterfaceDb(self._conn.connection())
        res = icdb.controller_status(id)
        for r in res :
            # strip leading path name from log files
            try:
                log = r['log'].split('/')
                r['log_url'] = 'system/'+log[-2]+'/'+log[-1]
            except:
                pass
        return res

    def controller_stop(self, id):
        """Stop given controller instance"""
                
        icdb = InterfaceDb(self._conn.connection())
        icdb.controller_stop(id)
        res = icdb.controller_status(id)
        if res : return res[0]
        return {}

    def requests(self, rid=None):
        """Returns the list of requests"""

        cursor = self._conn.cursor(True)

        # select filesets first
        vars = ()
        q = """SELECT fs.id, fs.instrument, fs.experiment, fs.run_number run, fs.priority priority,
            DATE_FORMAT(fs.created, GET_FORMAT(DATETIME,'ISO')) created, st.name status
            FROM fileset fs, fileset_status_def st
            WHERE st.id = fs.fk_fileset_status"""
        if rid is not None:
            q += " AND fs.id=%s"
            vars = vars + (rid,)
        cursor.execute(q, vars)

        # store in a dict
        res = {}
        for row in cursor.fetchall():
            res[row['id']] = dict(row)


        # select files next
        q = """SELECT fs.id, fi.name, fi.type
            FROM fileset fs, files fi
            WHERE fi.fk_fileset_id=fs.id"""
        if rid is not None:
            q += " AND fs.id=%s"
        cursor.execute(q, vars)

        # store in a dict
        for row in cursor.fetchall():
            
            id, fname, ftype = row['id'], row['name'], row['type']
            d = res.get(id)
            if d :
                if ftype == 'XTC' :
                    d.setdefault('xtc_files', []).append(fname)
                elif ftype == 'HDF5' :
                    d.setdefault('hdf_files', []).append(fname)
                else :
                    d.setdefault('other_files', []).append(fname)

        # select translator info
        q = """SELECT fs.id, DATE_FORMAT(tr.started, GET_FORMAT(DATETIME,'ISO')) started, 
            DATE_FORMAT(tr.stopped, GET_FORMAT(DATETIME,'ISO')) stopped, tr.log, IFNULL(tr.jobid,0) jobid
            FROM fileset fs, translator_process tr 
            WHERE fs.id=tr.fk_fileset"""
        if rid is not None:
            q += " AND fs.id=%s"
        cursor.execute(q, vars)
        
        # store in a dict
        for row in cursor.fetchall():
            
            d = res.get(row['id'])
            if d :
                d['log'] = row['log']
                d['started'] = row['started']
                d['stopped'] = row['stopped']
                d['jobid'] = row['jobid']

        return res.values()
    
    def create_request(self, instrument, experiment, run, force=False, priority=0):
        """Create new translation request, returns request object."""
        
        icdb = InterfaceDb(self._conn.connection())
        stat = 'WAIT_FILES'
        id = icdb.new_fileset(instrument, experiment, run, 'DATA', [], force, stat, priority)

        res = self.requests(id)
        if res : 
            res = res[0]
        else :
            res = None
        return res

    def change_request_priority(self, id, priority):
        """Change priority of existing request."""
        
        icdb = InterfaceDb(self._conn.connection())
        icdb.change_fileset_priority(id, priority)

        res = self.requests(id)
        if res : 
            res = res[0]
        else :
            res = None
        return res

    def delete_request(self, id):
        """Delete translation request."""
        
        icdb = InterfaceDb(self._conn.connection())
        icdb.stop_fileset_id(id)

    def active_index(self, instrument, experiment):
        """Get the list of active experiments"""

        cursor = self._conn.cursor(True)

        vars = ()
        q = """SELECT instrument, experiment, DATE_FORMAT(since, GET_FORMAT(DATETIME,'ISO')) since
            FROM active_exp"""
        if instrument:
            q += " WHERE instrument = %s"
            vars = vars + (instrument,)
        if experiment:
            if instrument :
                q += " AND experiment = %s"
            else :
                q += " WHERE experiment = %s"
            vars = vars + (experiment,)

        cursor.execute(q, vars)

        return list(cursor.fetchall())

    def activate_experiment ( self, instr, exp ) :
        """Add one more active experiment"""

        icdb = InterfaceDb(self._conn.connection())
        id = icdb.activate_experiment(instr, exp)
        return dict(instrument=instr, experiment=exp, since=Time.now().toString("%F %T"))

    def deactivate_experiment ( self, instr, exp ) :
        """Add one more active experiment"""

        icdb = InterfaceDb(self._conn.connection())
        id = icdb.deactivate_experiment(instr, exp)
        return dict(instrument=instr, experiment=exp)

    def experiments(self):
        """Returns the list of instruments/experiments"""

        cursor = self._conn.cursor(True)

        # get all instruments/experiments
        q = """SELECT DISTINCT instrument, experiment FROM fileset"""
        cursor.execute(q)

        return list(cursor.fetchall())

    def exp_requests(self, instrument, experiment):
        """Returns the list of instruments/experiments"""

        cursor = self._conn.cursor(True)

        # get all instruments/experiments
        q = """SELECT fs.id, fs.instrument, fs.experiment, fs.run_number run, fs.priority priority,
            DATE_FORMAT(fs.created, GET_FORMAT(DATETIME,'ISO')) created, st.name status
            FROM fileset fs, fileset_status_def st
            WHERE st.id = fs.fk_fileset_status AND fs.instrument = %s AND fs.experiment = %s"""
        cursor.execute(q, (instrument, experiment))

        # store in a dict
        res = {}
        for row in cursor.fetchall():
            res[row['id']] = dict(row)

        # select files next
        q = """SELECT fs.id, fi.name, fi.type
            FROM fileset fs, files fi
            WHERE fi.fk_fileset_id=fs.id AND fs.instrument = %s AND fs.experiment = %s"""
        cursor.execute(q, (instrument, experiment))

        # store in a dict
        for row in cursor.fetchall():
            
            id, fname, ftype = row['id'], row['name'], row['type']
            d = res.get(id)
            if d :
                if ftype == 'XTC' :
                    d.setdefault('xtc_files', []).append(fname)
                elif ftype == 'HDF5' :
                    d.setdefault('hdf_files', []).append(fname)
                else :
                    d.setdefault('other_files', []).append(fname)

        # select translator info
        q = """SELECT fs.id, DATE_FORMAT(tr.started, GET_FORMAT(DATETIME,'ISO')) started, 
            DATE_FORMAT(tr.stopped, GET_FORMAT(DATETIME,'ISO')) stopped, tr.log, IFNULL(tr.jobid,0) jobid
            FROM fileset fs, translator_process tr 
            WHERE fs.id=tr.fk_fileset AND fs.instrument = %s AND fs.experiment = %s"""
        cursor.execute(q, (instrument, experiment))
        
        # store in a dict
        for row in cursor.fetchall():
            
            d = res.get(row['id'])
            if d :
                d['log'] = row['log']
                d['started'] = row['started']
                d['stopped'] = row['stopped']
                d['jobid'] = row['jobid']

        return res.values()

    def check_expname(self, instrument, experiment):
        """Check that instrument/experiment name exist in regdb"""
        
        regdb = self._regdb()
        return regdb.find_experiment_by_name(instrument, experiment) is not None

    def get_config_sections(self):
        """
        self.get_config_sections() -> list of strings
        
        Returns the list of section names.
        """ 

        cursor = self._conn.cursor()

        vars = ()
        q = "SELECT DISTINCT section FROM config_def"
        cursor.execute(q)

        return [row[0] for row in cursor.fetchall()]

    def get_config(self, section=None):
        """
        self.get_config([section: str]) -> list of dicts
        
        Returns contents of configuration database as list of dict object,
        every dict object will have this set of keys:
        ("section", "param", "value", "type", "description", "instrument", "experiment").
        Instrument or experiment values can be None.
        
        If section name is given then parameter from single section are returned, otherwise
        all parameters are returned.
        """ 

        cursor = self._conn.cursor(True)

        vars = ()
        q = "SELECT section, param, value, type, description, instrument, experiment FROM config_def"
        if section is not None: 
            q += " WHERE section=%s"
            vars = (section,)
        cursor.execute(q, vars)

        return list(cursor.fetchall())

    def create_config(self, section, param, value, type, description, instrument, experiment):
        """
        self.create_config(section, param, value, type, description, instrument, experiment)

        Create new configuration parameter. Instrument or experiment values can be None.
        """ 

        cursor = self._conn.cursor()

        # use NULLs if they are empty
        instrument = instrument or None
        experiment = experiment or None

        # create parameter
        columns = ['section', 'param', 'value', 'type', 'description', 'instrument', 'experiment']
        qpar = (section, param, value, type, description, instrument, experiment)
        valstr = ','.join(["%s"] * len(columns))
        q = "INSERT  INTO config_def (%s) VALUES (%s)" % (','.join(columns), valstr)

        cursor.execute(q, qpar)
        self._conn.connection().commit()

    def update_config(self, section, param, value, instrument, experiment):
        """
        self.update_config(section, param, value, instrument, experiment)

        Update configuration parameter. Instrument or experiment values can be None.
        """ 

        cursor = self._conn.cursor()

        # use NULLs if they are empty
        instrument = instrument or None
        experiment = experiment or None

        # update parameter if it exists
        q = "UPDATE config_def SET value = %s WHERE section = %s AND param = %s"
        qpar = [value, section, param]
        if instrument:
            q += " AND instrument = %s"
            qpar.append(instrument)
        else:
            q += " AND instrument IS NULL"
        if experiment:
            q += " AND experiment = %s"
            qpar.append(experiment)
        else:
            q += " AND experiment IS NULL"

        cursor.execute(q, qpar)
        self._conn.connection().commit()

    def delete_config(self, section, param, instrument, experiment):
        """
        self.delete_config(section, param, instrument, experiment)

        Delete configuration parameter(s). Instrument or experiment values can be None.
        If param is None then all parameters in a section are deleted.
        """ 

        cursor = self._conn.cursor()
        
        # use NULLs if they are empty
        instrument = instrument or None
        experiment = experiment or None

        q = "DELETE FROM config_def WHERE section = %s"
        qpar = [section]
        if param:
            q += " AND param = %s"
            qpar.append(param)
        if instrument:
            q += " AND instrument = %s"
            qpar.append(instrument)
        else:
            q += " AND instrument IS NULL"
        if experiment:
            q += " AND experiment = %s"
            qpar.append(experiment)
        else:
            q += " AND experiment IS NULL"

        cursor.execute(q, qpar)
        self._conn.connection().commit()

    def _regdb(self) :

        icdb = InterfaceDb(self._conn.connection())
    
        # get regdb connection string from database
        config = icdb.read_config([])
        regdbConnStr = config.get("regdb-conn")
        
        return RegDb(DbConnection(conn_string=regdbConnStr))


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
