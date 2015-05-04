<?php

/*
 * This service will process a request for deleting a set of files
 * from an OFFLINE disk storage managed by the Data Management System.
 * A data set is defined by the following parameters:
 * 
 *   <exper_id> <run_number> <storage_class> <file_type>
 * 
 * When finished the script will return a JSOB object with the completion
 * status of the operation.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use FileMgr\RestRequest ;

function handler ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;
    $runnum   = $SVC->required_int('runnum') ;

    $type = $SVC->required_enum (
        'type' ,
         array('xtc', 'hdf5') ,
         array('ignore_case' => true ,      // when comparing parameters
               'convert'     => 'tolower'   // before storing elements in the result list
         )) ;

    $storage = $SVC->required_enum (
        'storage' ,
        array('SHORT-TERM', 'MEDIUM-TERM') ,
        array('ignore_case' => true ,       // when comparing parameters
              'convert'     => 'toupper'    // before storing elements in the result list
        )) ;

    $experiment = $SVC->safe_assign (
        $SVC->logbook()->find_experiment_by_id($exper_id) ,
        "no experiment found for id={$exper_id}" );

    $SVC->assert (
            $SVC->authdb()->hasPrivilege($SVC->authdb()->authName(), null, 'StoragePolicyMgr', 'edit') ,
            "you don't possess sufficient privileges for this operation") ;

    /* ATTENTION: This test is temporary disabled to allow moving files
     *            adopted from other experiments.
     *
    $run = $experiment->find_run_by_num($runnum) ;
    if (!$run)
        $SVC->abort (
            "no run {$runnum} in the experiment id={$exper_id} " ,
            array( 'medium_quota_used_gb' => $SVC->configdb()->calculate_medium_quota($exper_id))) ;
    */

    // Find the files to be deleted. Only consider file which are associated
    // with the specified storage class.

    $files = array() ;
    foreach ($SVC->irodsdb()->runs (
        $experiment->instrument()->name(),
        $experiment->name(),
        $type ,
        $runnum ,
        $runnum) as $run)

        foreach ($run->files as $file)
            if ($file->resource == 'lustre-resc') {
                $request = $SVC->configdb()->find_medium_store_file (
                    array (
                        'exper_id'       => $exper_id,
                        'runnum'         => $runnum,
                        'file_type'      => $type ,
                        'irods_filepath' => "{$file->collName}/{$file->name}" ,
                        'irods_resource' => 'lustre-resc'
                    )
                ) ;
                switch ($storage) {
                    case 'SHORT-TERM'  : if ( is_null($request)) array_push($files, $file) ; break ;
                    case 'MEDIUM-TERM' : if (!is_null($request)) array_push($files, $file) ; break ;
                }
            }

    // Proceed with the operation

    foreach ($files as $file) {

        $irods_filepath = "{$file->collName}/{$file->name}" ;

        // Delete selected files by their full path and a replica.

        $request = new RestRequest (
            "/replica{$irods_filepath}/{$file->replica}" ,
            'DELETE'
        ) ;
        $request->execute() ;
        $responseInfo = $request->getResponseInfo() ;
        $http_code = intval($responseInfo['http_code']) ;
        switch ($http_code) {
            case 200 : break ;
            case 404 :
                $SVC->abort (
                   "file '{$file->name}' doesn't exist" ,
                   array ('medium_quota_used_gb' => $SVC->configdb()->calculate_medium_quota($exper_id))
                ) ;
            default :
                $SVC->abort (
                    "failed to delete file '{$file->name}' because of HTTP error {$http_code}" ,
                    array ('medium_quota_used_gb' => $SVC->configdb()->calculate_medium_quota($exper_id))
                ) ;
        }
        if ($storage === 'MEDIUM-TERM') {
            $SVC->configdb()->remove_medium_store_file($exper_id, $runnum, $type, $irods_filepath) ;
        }

        // Delete selected files by their full path and a replica. Also make sure
        // no old entries remain in the file restore queue. Otherwise it would
        // (falsefully) appear as if the deleted file is being restored.

        $SVC->configdb()->delete_file_restore_request (
            array (
                'exper_id'           => $exper_id ,
                'runnum'             => $runnum ,
                'file_type'          => $type ,
                'irods_filepath'     => $irods_filepath ,
                'irods_src_resource' => 'hpss-resc' ,
                'irods_dst_resource' => 'lustre-resc'
            )
        ) ;
    }
    return array (
        'medium_quota_used_gb' => $SVC->configdb()->calculate_medium_quota($exper_id)
    ) ;
}

\DataPortal\ServiceJSON::run_handler ('GET', 'handler') ;

?>
