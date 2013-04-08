<?php

/**
 * This service will return an information record on the specified file
 * system (if a non-zero identifier is provide) or all known file systems
 * (if 0 is passed as a file system identifier). A scope of the operation
 * is defined by the second parameter.
 * 
 * Parameters:
 * 
 *   <id> <scope>
 *
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'sysmon/sysmon.inc.php' ;

// ATTENTION: This will increase the default value for the maximum
//            execution time limit from 30 seconds to 300 seconds.
//            The later is the connection timeout for IIS and Apache.
//            So, it makes no sense to increase it further w/o redesigning
//            this algorithm.
//
set_time_limit( 300 );


\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $id    = $SVC->required_int('id') ;
    $scope = strtolower($SVC->required_str('scope')) ;

    $result = null ;
    switch ($scope) {
        case 'summary'    : $result = $SVC->sysmon()->file_system_summary($id ? $id : null) ; break ;
        case 'extensions' : $result = $SVC->sysmon()->file_extensions    ($id ? $id : null) ; break ;
        case 'types'      : $result = $SVC->sysmon()->file_types         ($id ? $id : null) ; break ;
        case 'sizes'      : $result = $SVC->sysmon()->file_sizes         ($id ? $id : null) ; break ;
        case 'ctime'      : $result = $SVC->sysmon()->file_ctime         ($id ? $id : null) ; break ;
        default :
            $SVC->abort("unsupported scope: '{$scope}'") ;
    }
    $SVC->finish(array ($scope => $result)) ;
}) ;
  
?>
