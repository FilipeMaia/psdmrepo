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
    server = 'psdb'
    table  = 'calibman'

    def __init__(self) :
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
        info_dict['vers'] = gu.get_pkg_version('CalibManager')
        return info_dict


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


    def create_table(self) :
        self.db.query(self.cmd_create_table())


    def delete_table(self) :
        self.db.query('DROP TABLE %s' % self.table)


    def insert_record(self) :
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

#------------------------------

def test_create_table() :
    ndb = NotificationDB()
    print 'cmd_create_table(): ', ndb.cmd_create_table()
    ndb.create_table()
    ndb.close()


def test_get_dict() :
    ndb = NotificationDB()
    list_of_recs = ndb.get_list_of_recs_for_query(ndb.cmd_fetch())
    print 'Resp:\n',
    for rec in list_of_recs : print rec
    #ndb.delete_table()
    ndb.close()


def test_insert_record() :
    ndb = NotificationDB()
    ndb.insert_record()
    ndb.close()

#------------------------------

if __name__ == "__main__" :

    #test_create_table()
    #test_insert_record()
    test_get_dict()

    sys.exit (0)

#------------------------------
