<?php

/*
 * This script will process a request for moving a set of files
 * between the SHORT-TERM and MEDIUM-TERM data storage. This operation
 * only applies to files managed by the Data MAnagement System.
 * A data set is defined by the following parameters:
 * 
 *   <exper_id> <run_number> <storage_class> <file_type>
 *
 * The '<storage_class>' parameter defines the source storage of the operation.
 * The files (if found) will be moved to an opposite storage class.
 *
 * When finished the script will return the amount of the MEDIUM-TERM storage
 * space which is available to teh experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;

define ('BYTES_IN_GB', 1024.0 * 1024.0 * 1024.0) ;

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
            $SVC->logbookauth()->canEditMessages($experiment->id()) ||
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

    // Find the files to be moved. And skip files which are already at their
    // intended destination.

    $files = array() ;
    foreach ($SVC->irodsdb()->runs (
        $experiment->instrument()->name() ,
        $experiment->name() ,
        $type ,
        $runnum, $runnum ) as $run) {

        foreach ($run->files as $file) {
            if ($file->resource === 'lustre-resc') {
                $request = $SVC->configdb()->find_medium_store_file (
                    array (
                        'exper_id'       => $exper_id ,
                        'runnum'         => $runnum ,
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
        }
    }

    // Move the files to their destination storage.

    switch( $storage ) {

        case 'SHORT-TERM' :

            // Evaluate quota availability (if the one applies for the experiment)
            // before moving the files.
            //
            $medium_quota_gb = intval($SVC->configdb()->get_policy_param('MEDIUM-TERM',   'QUOTA')) ;
            $param = $experiment->regdb_experiment()->find_param_by_name('MEDIUM-TERM-DISK-QUOTA') ;
            if ($param) {
                $gb = intval($param->value()) ;
                if ($gb) $medium_quota_gb = $gb ;
            }
            if ($medium_quota_gb) {
                $medium_quota_used_gb = $SVC->configdb()->calculate_medium_quota($exper_id) ;
                $size_gb = 0 ;
                foreach ($files as $file)
                    $size_gb += $file->size / BYTES_IN_GB ;

                if ($medium_quota_used_gb + $size_gb > $medium_quota_gb) {
                    $SVC->abort (
                        'You can no longer move files into the MEDIUM-TERM storage ' .
                        'either because the experiment has run out of its storage quota or because the amount of data ' .
                        'involved in this request would result in exceeding  quota limit of the experiment.' ,
                        array (
                            'medium_quota_used_gb' => $medium_quota_gb
                        )
                    ) ;
                }
            }
            foreach ($files as $file) {
                $irods_filepath = "{$file->collName}/{$file->name}" ;
                $SVC->configdb()->add_medium_store_file (
                    array (
                        'exper_id'       => $exper_id ,
                        'runnum'         => $runnum ,
                        'file_type'      => $type ,
                        'irods_filepath' => $irods_filepath ,
                        'irods_resource' => 'lustre-resc' ,
                        'irods_size'     => $file->size
                    )
                ) ;
            }
            break ;

        case 'MEDIUM-TERM' :

            foreach ($files as $file) {
                $irods_filepath = "{$file->collName}/{$file->name}" ;
                $SVC->configdb()->remove_medium_store_file($exper_id, $runnum, $type, $irods_filepath) ;
            }
            break ;
    }

    return array (
        'medium_quota_used_gb' => $SVC->configdb()->calculate_medium_quota($exper_id)
    ) ;
}

\DataPortal\ServiceJSON::run_handler ('GET', 'handler') ;

?>
