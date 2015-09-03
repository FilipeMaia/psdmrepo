#!/bin/env /usr/bin/python

"""
Test modules found in the local directory. Note that this test is designed
not to have side effects such as modifying any database or file system content.

"""

print """
__________________________________
testing module psdm.file_status..."""

import psdm.file_status as file_status

files2check = [
    (144, 'xtc', 'e144-r0243-s01-c00.xtc'),
    (144, 'xtc', 'e144-r0243-s01-c01.xtc'),
    (144, 'xtc', 'e144-r0243-s01-c02.xtc'),
    (145, 'xtc', 'e145-r0011-s00-c00.xtc'),
    (145, 'xtc', 'e145-r0012-s00-c00.xtc'),
    (144, 'xtc', 'e144-r0243-s01-c04.xtc'),
    ( 55, 'xtc', 'e55-r0042-s00-c00.xtc' ),
    ( 55, 'xtc', 'e55-r0041-s00-c00.xtc' ),
]

def evaluate(triplet,status):
    data_migration_flag = '-'
    if file_status.IN_MIGRATION_DATABASE in status.flags(): data_migration_flag = 'x'

    disk_flag = '-'
    if file_status.DISK in status.flags(): disk_flag = 'x'

    hpss_flag = '-'
    if file_status.HPSS in status.flags(): hpss_flag = 'x'

    print "   {:>8} | {:>9} | {:>25} | {:>12} |            {:>1}          |    {:>1}    |    {:>1}".format(int(triplet[0]),triplet[1],triplet[2],status.size_bytes(),data_migration_flag,disk_flag,hpss_flag)

    return True

print """
  ----------+-----------+---------------------------+--------------+-----------------------+---------+---------
   exper_id | file type | file name                 | size (bytes) | IN MIGRATION DATABASE | ON DISK | ON HPSS
  ----------+-----------+---------------------------+--------------+-----------------------+---------+---------"""

fs = file_status.file_status(ws_login_user='psdm_reader', ws_login_password='pcds')

files2select = fs.filter(files2check,evaluate)
