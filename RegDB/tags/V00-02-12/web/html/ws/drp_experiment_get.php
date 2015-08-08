<?php

/*
 * Return the data usage statistics for an experiment
 * 
 * AUTHORIZATION: not required
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime ;

define ('GB', 1024.*1024.*1024.) ;
define ('SEC_PER_MONTH', 30*24*3600) ;

define ('MAX_MONTH', 24) ;

$now = LusiTime::now() ;

function last_access_month_ago ($atime) {
    global $now ;
    $sec = $now->sec - $atime ;
    $months = floor($sec / SEC_PER_MONTH) ;
    if ($months < 1) return 1 ;
    if ($months > MAX_MONTH) return MAX_MONTH ;
    return $months ;
}

function file_has_expired ($policy, $storage, $file) {
    global $now ;
    $sec = $now->sec - $file->ctime ;
    return ($sec / SEC_PER_MONTH) >= $policy[$storage]['retention'] ;
}

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;
    
    $exper = $SVC->safe_assign (
        $SVC->logbook()->find_experiment_by_id($exper_id) ,
        "No experiment found for id: {$exper_id}") ;

    // The amended policy will be used to determine which files
    // have been expired.
    $policy = $SVC->configdb()->experiment_policy2array($SVC, $exper->regdb_experiment()) ;

    // Initialize the temporary array wich will be used to collect
    // the statistics.
    $stats = array (
        'total' => array (
            'SHORT-TERM' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 0 ,
                'purge_allowed'  => 0) ,
            'MEDIUM-TERM' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 0 ,
                'purge_allowed'  => 0) ,
            'HPSS' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 0 ,
                'purge_allowed'  => 0)) ,
        'expired' => array (
            'SHORT-TERM' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 1 ,
                'purge_allowed'  => 1) ,
            'MEDIUM-TERM' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 1 ,
                'purge_allowed'  => 1) ,
            'HPSS' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 0 ,
                'purge_allowed'  => 0))
    ) ;
    for ($m = MAX_MONTH; $m > 0; $m--)
        $stats["{$m}"] = array (
            'SHORT-TERM' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 1 ,
                'purge_allowed'  => 1) ,
            'MEDIUM-TERM' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 1 ,
                'purge_allowed'  => 1) ,
            'HPSS' => array (
                'xtc'            => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'hdf5'           => array('num_files' => 0, 'size_gb' => 0, 'runs' => array()) ,
                'review_allowed' => 0 ,
                'purge_allowed'  => 0)) ;

    // Collect the statistics for all known file types
    $KNOWN_TYPES = array('xtc', 'hdf5') ;

    foreach ($KNOWN_TYPES as $type) {

        $type_uppercase = strtoupper($type) ;

        foreach (
            $SVC->irodsdb()->runs (
                $exper->instrument()->name() ,
                $exper->name() ,
                $type) as $run) {

            $runnum = $run->run ;

            // These variables will be set once and the same for all files of
            // the dataset.

            // 
            $storage           = null ;
            $storage_num_files = 0 ;
            $storage_size_gb   = 0. ;

            // if any files expires then all files are considred as expired
            $expired           = false ;
            $expired_num_files = 0 ;
            $expired_size_gb   = 0. ;

            // the most recent access month within a dataset applies to all 
            $month             = MAX_MONTH ;
            $month_num_files   = 0 ;
            $month_size_gb     = 0 ;
    
            foreach ($run->files as $file) {
                switch ($file->resource) {
                    case 'hpss-resc':
                        $stats['total']['HPSS'][$type]['num_files'] += 1 ;
                        $stats['total']['HPSS'][$type]['size_gb']   += floor($file->size / GB) ;
                        break ;

                    case 'lustre-resc':
                        if (is_null($storage)) {
                            $storage = $SVC->configdb()->find_medium_store_file (array (
                                'exper_id'       => $exper_id ,
                                'runnum'         => $runnum ,
                                'file_type'      => $type_uppercase ,
                                'irods_filepath' => "{$file->collName}/{$file->name}" ,
                                'irods_resource' => $file->resource)) ? 'MEDIUM-TERM' : 'SHORT-TERM' ;
                        }
                        $storage_num_files += 1 ;
                        $storage_size_gb   += floor($file->size / GB) ;

                        $expired = $expired || file_has_expired ($policy, $storage, $file) ;
                        if ($expired) {
                            $expired_num_files += 1 ;
                            $expired_size_gb   += floor($file->size / GB) ;
                        }
                        $month =  min($month, last_access_month_ago($file->atime)) ;
                        if ($month) {
                            $month_num_files += 1 ;
                            $month_size_gb   += floor($file->size / GB) ;
                        }
                        break ;
                }
            }
            if (!is_null($storage)) {
                $stats['total'][$storage][$type]['num_files'] += $storage_num_files ;
                $stats['total'][$storage][$type]['size_gb']   += $storage_size_gb ;
                if ($expired) {
                               $stats['expired'][$storage][$type]['num_files'] += $expired_num_files ;
                               $stats['expired'][$storage][$type]['size_gb']   += $expired_size_gb ;
                    array_push($stats['expired'][$storage][$type]['runs'],     $runnum)  ;
                }
                           $stats["{$month}"][$storage][$type]['num_files'] += $month_num_files ;
                           $stats["{$month}"][$storage][$type]['size_gb']   += $month_size_gb ;
                array_push($stats["{$month}"][$storage][$type]['runs'],     $runnum)  ;
            }
        }
    }
    // Transfer statistics into an array to be reported to
    // the caller.
    $DISK_STORAGE_CLASSES = array('SHORT-TERM','MEDIUM-TERM') ;

    // Disable 'review' and 'purge' for categories where no data
    // have been found.
    foreach ($DISK_STORAGE_CLASSES as $storage) {
        $category = 'expired' ;
        $num_files = 0 ;
        foreach ($KNOWN_TYPES as $type) {
            $num_files += $stats[$category][$storage][$type]['num_files'] ;
        }
        if (($m <= 1) || !$num_files) {
            $stats[$category][$storage]['review_allowed'] = 0 ;
            $stats[$category][$storage]['purge_allowed']  = 0 ;
        }
    }
    $experiment_data = array (
        array (
            'category'    => 'total' ,
            'title'       => 'TOTAL' ,
            'SHORT-TERM'  => $stats['total']['SHORT-TERM'] ,
            'MEDIUM-TERM' => $stats['total']['MEDIUM-TERM'] ,
            'HPSS'        => $stats['total']['HPSS']) ,
        array (
            'category'    => 'expired' ,
            'title'       => 'Expired (by Policy)' ,
            'SHORT-TERM'  => $stats['expired']['SHORT-TERM'] ,
            'MEDIUM-TERM' => $stats['expired']['MEDIUM-TERM'] ,
            'HPSS'        => $stats['expired']['HPSS']) ,
    ) ;

    for ($m = MAX_MONTH; $m > 0; $m--) {
        $category = "{$m}" ;

        // Disable 'review' and 'purge' for the recently created
        // data in order to protect them from being accidentally
        // deleted. Do the same for categories where no data
        // have been found.
        foreach ($DISK_STORAGE_CLASSES as $storage) {
            $num_files = 0 ;
            foreach ($KNOWN_TYPES as $type) {
                $num_files += $stats[$category][$storage][$type]['num_files'] ;
            }
            if (($m <= 1) || !$num_files) {
                $stats[$category][$storage]['review_allowed'] = 0 ;
                $stats[$category][$storage]['purge_allowed']  = 0 ;
            }
        }        
        array_push($experiment_data, array (
            'category'    => $category ,
            'title'       => "{$category} m" ,
            'SHORT-TERM'  => $stats[$category]['SHORT-TERM'] ,
            'MEDIUM-TERM' => $stats[$category]['MEDIUM-TERM'] ,
            'HPSS'        => $stats[$category]['HPSS'])) ;
    }

    return array (
        'general_policy'    => $SVC->configdb()->general_policy2array($SVC) ,
        'experiment_policy' => $policy ,
        'experiment_data'   => $experiment_data
    ) ;
}) ;