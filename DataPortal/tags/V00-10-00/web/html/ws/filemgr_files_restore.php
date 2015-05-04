<?php

/*
 * This script will process a request for restoring a set of files
 * from HPSS to an OFFLINE disk storage managed by the Data Management System.
 * A data set is defined by the following parameters:
 * 
 *   <exper_id> <run_number> <file_type> <storage_class>
 * 
 * When finished the script will return a JSOB object with the completion
 * status of the operation.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use FileMgr\FileMgrIrodsWs ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    //$SVC->abort("We're sorry! Restoring files from tapes is temporarily disabled due to technical reasons. Please, try again in 24 hours") ;

    $exper_id =                 $SVC->required_int('exper_id') ;
    $runnum   =                 $SVC->required_int('runnum') ;
    $type     = strtolower(trim($SVC->required_str('type'))) ;
    $storage  = strtoupper(trim($SVC->required_str('storage'))) ;


    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     'No such experiment');

    /* ATTENTION: This test is temporary disabled to allow moving files
     *            adopted from other experiments.
     *
    $run = $SVC->safe_assign ($experiment->find_run_by_num($runnum) ,
                              "No such run in the experiment",
                              array( 'medium_quota_used_gb' => $SVC->configdb()->calculate_medium_quota($exper_id))) ;
    */

    // Find the files to be restored. And make sure there are no outstanding
    // restore requests for that file yet.

    $runs = null ;

    FileMgrIrodsWs::runs (
        $runs ,
        $experiment->instrument()->name() ,
        $experiment->name() ,
        $type ,
        $runnum.'-'.$runnum) ;

    $SVC->assert ($runs ,
                  'server encountered an internal error when trying to get a list of files for the run',
                  array('medium_quota_used_gb' => $SVC->configdb()->calculate_medium_quota($exper_id))) ;

    $files = array() ;
    foreach ($runs as $run)
        foreach ($run->files as $file)
            if ($file->resource == 'hpss-resc' &&
                !$SVC->configdb()->find_file_restore_request (array (
                    'exper_id'           => $exper_id ,
                    'runnum'             => $runnum ,
                    'file_type'          => $type ,
                    'irods_filepath'     => "{$file->collName}/{$file->name}" ,
                    'irods_src_resource' => 'hpss-resc' ,
                    'irods_dst_resource' => 'lustre-resc')))
                array_push($files, $file) ;

    // Restore selected files by their full path and a source resource

    foreach ($files as $file)
        $SVC->configdb()->add_file_restore_request (
            array (
                'exper_id'           => $exper_id ,
                'runnum'             => $runnum ,
                'file_type'          => $type ,
                'irods_filepath'     => "{$file->collName}/{$file->name}" ,
                'irods_src_resource' => 'hpss-resc' ,
                'irods_dst_resource' => 'lustre-resc'
            )
        );

    return array (
        'medium_quota_used_gb' => $SVC->configdb()->calculate_medium_quota($exper_id)
    ) ;
}) ;

?>
