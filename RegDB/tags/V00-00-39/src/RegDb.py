#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: $
#
# Description:
#  Module RegDb...
#
#------------------------------------------------------------------------

""" Interface class for RegDb.

This software was developed for the LUSI project.  If you use all or
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: $

@author Igor Gaponenko
"""


#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 702 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

from LusiTime.Time import Time

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------

class RegDb ( object ) :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, conn, log=None ) :
        """Constructor.

        @param conn      database connection object
        """

        self._conn = conn
        self._log = log or logging.getLogger()

    def begin(self):
        self._conn.cursor().execute( 'BEGIN' )

    def commit(self):
        self._conn.cursor().execute( 'COMMIT' )


    # =================================
    # Find experiment by its identifier
    # =================================

    def find_experiment_by_id(self, id):
        """
        Find experiment by its identifier. Return none or False if no such experiment
        or a dictionary with parameters of the experiment. The following keys will
        be found in the dictionary:
            - instr_id
            - instr_name
            - instr_descr
            - id
            - name
            - descr
            - leader_account
            - posix_gid
            - contact_info
            - registration_time
            - begin_time
            - end_time
        """

        # find the experiment
        cursor = self._conn.cursor( True )
        cursor.execute( """SELECT i.name AS instr_name, i.descr AS instr_descr, e.* FROM experiment e, instrument i WHERE e.id=%s AND e.instr_id=i.id""",  (id, ))
        rows = cursor.fetchall()
        if not rows : return None
        if len(rows) > 1:
            raise Exception("Too many rows for experiment id %s" % id)

        row = rows[0]

        row['registration_time'] = Time.from64( row['registration_time'] )
        row['begin_time']        = Time.from64( row['begin_time'] )
        row['end_time']          = Time.from64( row['end_time'] )

        return row


    def find_experiment_by_name(self, instrName, expName):
        """
        Find experiment by its name (plus instrument name). Returns
        the same result as find_experiment_by_id().
        """

        # find the experiment
        cursor = self._conn.cursor( True )
        q = """SELECT i.name AS instr_name, i.descr AS instr_descr, e.* 
            FROM experiment e, instrument i 
            WHERE i.name=%s AND e.name=%s AND e.instr_id=i.id"""
        cursor.execute(q, (instrName, expName))
        rows = cursor.fetchall()
        if not rows : return None
        if len(rows) > 1:
            raise Exception("Too many rows for experiment %s:%s" % (instrName, expName))

        row = rows[0]

        row['registration_time'] = Time.from64( row['registration_time'] )
        row['begin_time']        = Time.from64( row['begin_time'] )
        row['end_time']          = Time.from64( row['end_time'] )

        return row


    def last_experiment_switch(self, instr):
        """Get the latest experiment for the given instrument name,
        returns tuple (expName, time, user), time is LusiTime.Time object,
        user is user name who requested switch. Returns None if cannot find
        information for given instrument name."""

        cursor = self._conn.cursor()
        q = """SELECT e.name, sw.switch_time, sw.requestor_uid 
            FROM expswitch sw, experiment e, instrument i
            WHERE sw.exper_id = e.id AND e.instr_id = i.id AND i.name=%s
            ORDER BY sw.switch_time DESC LIMIT 1"""
        cursor.execute(q, (instr,))
        
        rows = cursor.fetchall()
        if not rows: return None

        name, time, user = rows[0]
        time = Time.from64(time);
            
        return (name, time, user)
