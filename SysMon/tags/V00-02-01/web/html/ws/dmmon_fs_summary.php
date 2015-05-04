<?php

/**
 * This service will return the summary statistics for the file system usage
 * for all known file systems
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $fs_data = array() ;
    foreach ($SVC->sysmon()->fs_mon_summary() as $s) {
        array_push (
            $fs_data ,
            array (
                'name'      => $s['name'] ,
                'used'      => $s['used'] ,
                'available' => $s['available']
            )
        ) ;
    }
    return array('filesystems' => $fs_data) ;
}) ;
  
?>
