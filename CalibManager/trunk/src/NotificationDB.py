#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module NotificationDB.py...
#
#------------------------------------------------------------------------

"""
This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

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
import _mysql
from ConfigParametersForApp import cp
import GlobalUtils          as     gu

#------------------------------

class NotificationDB :
    """Is intended for submission of notification records in db
    """

    def __init__(self, server='psdb', table='calibman') :
        self.server = server
        self.table  = table
        self.db = _mysql.connect(self.server, cp.par02, cp.par01[7:3:-1].lower(), cp.par02)


    def cmd_fetch(self) :
        return """SELECT * FROM %s;""" % self.table


    def cmd_create_table(self) :
        cmd = 'CREATE TABLE IF NOT EXISTS ' \
            + self.table \
            + '(' \
            + 'id INT NOT NULL AUTO_INCREMENT, PRIMARY KEY(id)' \
            + ', date VARCHAR(10)' \
            + ', time VARCHAR(8)'  \
            + ', zone VARCHAR(3)'  \
            + ', user VARCHAR(32)' \
            + ', host VARCHAR(32)' \
            + ', cwd  VARCHAR(64)' \
            + ', exp  VARCHAR(8)'  \
            + ', run  VARCHAR(4)'  \
            + ', dets VARCHAR(64)' \
            + ', vers VARCHAR(9)' \
            + ');'
        return cmd


    def get_info_dict(self) :
        info_dict = {}
        date,time,zone = gu.get_current_local_time_stamp().split()
        info_dict['date'] = date
        info_dict['time'] = time
        info_dict['zone'] = zone
        info_dict['user'] = gu.get_enviroment(env='LOGNAME')
        info_dict['host'] = gu.get_hostname()
        info_dict['cwd']  = gu.get_cwd()
        info_dict['exp']  = cp.exp_name.value()
        info_dict['run']  = cp.str_run_number.value()
        info_dict['dets'] = cp.det_name.value()
        info_dict['vers'] = self.get_version()
        return info_dict


    def get_version(self) :
        try :
            return gu.get_pkg_version()
            #return gu.get_pkg_tag('CalibManager') # Very slow, uses: psvn tags CalibManager
            #return cp.package_versions.get_pkg_version('CalibManager')
        except :
            return 'N/A'


    def cmd_insert_record(self) :
        info_dict = self.get_info_dict()
        str_of_keys = ', '.join(info_dict.keys())
        str_of_vals = str(info_dict.values()).strip('[]')
        #print str_of_keys
        #print str_of_vals
        #INSERT INTO example (name, age) VALUES('Timmy Mellowman', '23' );
        return 'INSERT INTO %s (%s) VALUES(%s);' % (self.table, str_of_keys, str_of_vals)


    def query(self, cmd) :
        self.db.query(cmd)


    def get_list_of_recs_for_query(self, cmd) :
        self.db.query(cmd)
        result = self.db.store_result()
        return result.fetch_row(maxrows=0,how=1 ) # all rows, as dictionary


    def get_list_of_recs(self) :
        return self.get_list_of_recs_for_query(self.cmd_fetch())


    def get_list_of_values(self) :
        return [rec.values() for rec in self.get_list_of_recs()]


    def get_list_of_keys(self) :
        cmd = """SELECT * FROM %s WHERE id=1;""" % self.table
        return self.get_list_of_recs_for_query(cmd)[0].keys()


    def is_permitted(self) :
        return True if gu.get_enviroment(env='LOGNAME') == cp.par02 else False


    def msg_about_permission(self) :
        print 'Sorry, this operation is not permitted!'


    def create_table(self) :
        if self.is_permitted() : self.db.query(self.cmd_create_table())
        else                   : self.msg_about_permission()


    def delete_table(self) :
        if self.is_permitted() : self.db.query('DROP TABLE %s' % self.table)
        else                   : self.msg_about_permission()


    def insert_record(self, mode='self-disabled') :
        if mode=='self-disabled' and self.is_permitted() : return
        #try :
        cmd = self.cmd_insert_record() 
        #print 'cmd: %s' % cmd
        self.db.query(cmd)
        #except :
        #    pass


    def close(self) :
        self.db.close()


    def add_record(self) :
        self.insert_record()
        self.close()


    def print_pars(self) :
        print 'server = %s' % self.server
        print 'table  = %s' % self.table


#------------------------------

def test_notification_db(ndb, test_num):

    print 'Test: %d' % test_num

    if test_num == 0 :
        print 'Create table for:'
        ndb.print_pars()
        print 'cmd_create_table(): ', ndb.cmd_create_table()
        ndb.create_table()

    elif test_num == 1 :
        print 'DB content:'
        list_of_recs = ndb.get_list_of_recs_for_query(ndb.cmd_fetch())
        print 'Resp:\n',
        for rec in list_of_recs : print rec

    elif test_num == 2 :
        print 'insert/submit a record in the DB'
        ndb.insert_record(mode='enabled')
        #ndb.insert_record() # default: mode='self-disabled'

    elif test_num == 3 :
        print 'Keys:  %s' % ndb.get_list_of_keys()

    elif test_num == 4 :
        print 'Values:'
        for vals in ndb.get_list_of_values() : print vals

    elif test_num == 5 :
        print 'DB parameters:'
        ndb.print_pars()

    elif test_num == 9 :
        print 'Delete table:'
        ndb.print_pars()
        ndb.delete_table()

#------------------------------

def main_test(ndb):

    if len(sys.argv)==2 and sys.argv[1] == '-h' :
        msg  = 'Use %s with a single parameter, <test number=0,1,2,3,...>' % sys.argv[0]
        msg += '\n    0 - create table in db ...'        
        msg += '\n    1 - print db content'        
        msg += '\n    2 - insert/submit a record in the db'        
        msg += '\n    3 - print keys'        
        msg += '\n    4 - print values'        
        msg += '\n    5 - print DB parameters'        
        msg += '\n    9 - delete table ...'        
        print msg

    else :

        try    :
            test_num = int(sys.argv[1])
            test_notification_db(ndb, test_num)
        except :
            test_notification_db(ndb, 3)
            test_notification_db(ndb, 4)

#------------------------------

if __name__ == "__main__" :

    ndb = NotificationDB()
    main_test(ndb)
    ndb.close()

    sys.exit ( 'End of test NotificationDB' )

#------------------------------
