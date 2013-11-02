import file_status

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

    print "   %8d | %9s | %25s |            %1s          |    %1s    |    %1s" % (int(triplet[0]),triplet[1],triplet[2],data_migration_flag,disk_flag,hpss_flag)

    return True

print "\n".join([
    "",
    "  ----------+-----------+---------------------------+-----------------------+---------+---------",
    "   exper_id | file type | file name                 | IN MIGRATION DATABASE | ON DISK | ON HPSS ",
    "  ----------+-----------+---------------------------+-----------------------+---------+---------"])

fs = file_status.file_status(ws_login_user='psdm_reader', ws_login_password='pcds')

files2select = fs.filter(files2check,evaluate)

print ""

