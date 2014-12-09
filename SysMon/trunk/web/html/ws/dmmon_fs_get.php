<?php

/**
 * This service will return the statistics for the file system usage in
 * the specified context and an interval of time.
 * 
 * Parameters:
 * 
 *   [<fs_id>] [<interval_sec>]
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $fs_id        = $SVC->optional_int('fs_id', 0) ;
    $interval_sec = $SVC->optional_int('interval_sec', 0) ;

    $fs = $fs_id ? array (
        $SVC->safe_assign ($SVC->sysmon()->fs_mon_def_by_id($fs_id) ,
                           "no file system exists for id={$fs_id}")) :
        $SVC->sysmon()->fs_mon_def() ;

    $fs_data = array() ;
    foreach ($fs as $f) {
        $stats = array() ;
        foreach ($SVC->sysmon()->fs_mon_stat($f['id']) as $s) {
            $insert_time = $s['insert_time'] ;
            array_push (
                $stats ,
                array (
                    'insert_time' => array (
                        'time' => $insert_time->toStringShort() ,
                        'day'  => $insert_time->toStringDay() ,
                        'hms'  => $insert_time->toStringHMS() ,
                        'sec'  => $insert_time->sec ,
                        'nsec' => $insert_time->nsec
                    ) ,
                    'used'      => $s['used'] ,
                    'available' => $s['available'] ,
                )
            ) ;
        }
        $f['stats'] = $stats ;
        array_push (
            $fs_data ,
            $f
        ) ;
    }
    return array('filesystems' => $fs_data) ;
}) ;
  
?>
