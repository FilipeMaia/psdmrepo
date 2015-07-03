<?php

/**
 * This service will search for the files matching the specified criteria.
 *
 * Mandatory parameters:
 * 
 *   <exper_id>                           - experiment identifier
 * 
 * Optional parameters of the filter:
 *
 *   <type>     xtc[,hdf5]                - file types (case doesn't matter)
 *   <runs>     [<number>][-][<number>]   - a range of runs
 *   <checksum> {0|1}                     - a flag indicating if a checksum is available for files
 *   <archived> {0|1}                     - a flag indicating if the files are archived
 *   <local>    {0|1}                     - a flag indicating if there is a local copy of files
 *   <storage>  {SHORT-TERM|MEDIUM-TERM}  - a storage class
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime;

define('KB', 1024.0) ;
define('MB', 1024.0 * KB) ;
define('GB', 1024.0 * MB) ;

define('SECONDS_IN_HOUR', 3600) ;
define('SECONDS_IN_DAY',    24 * SECONDS_IN_HOUR) ;
define('SECONDS_IN_WEEK',    7 * SECONDS_IN_DAY) ;
define('SECONDS_IN_MONTH',  31 * SECONDS_IN_DAY) ;

function autoformat_size ($bytes) {
    $normalized = $bytes;
    $format     = '%d';
    $units      = '';
    if      ($bytes < KB) {}
    else if ($bytes < MB) { $normalized = $bytes / KB; $format = $bytes < 10 * KB ? '%.1f' : '%d'; $units = 'KB'; }
    else if ($bytes < GB) { $normalized = $bytes / MB; $format = $bytes < 10 * MB ? '%.1f' : '%d'; $units = 'MB'; }
    else                  { $normalized = $bytes / GB; $format = $bytes < 10 * GB ? '%.1f' : '%d'; $units = 'GB'; }
    return sprintf($format, $normalized).' <span style="font-weight:bold; font-size:9px;">'.$units.'</span>';
}

function add_files (&$files, $infiles, $type, $file2run, $checksum, $archived, $local) {
    
    $result = array () ;

    foreach ($infiles as $file) {

        if ($file->type == 'collection') continue ;

        // Skip files for which we don't know run numbers

        if (!array_key_exists($file->name, $file2run)) continue ;
        
        if (!array_key_exists($file->name, $result)) {
            $result[$file->name] =
                array (
                    'name'           => $file->name ,
                    'irods_filepath' => "{$file->collName}/{$file->name}" ,
                    'run'            => $file2run[$file->name] ,
                    'type'           => $type ,
                    'size'           => $file->size ,
                    'created'        => $file->ctime ,
                    'archived'       => '<span style="color:red;">No</span>' ,
                    'archived_flag'  => 0 ,
                    'local'          => '<span style="color:red;">No</span>' ,
                    'local_flag'     => 0 ,
                    'checksum'       => '') ;    
        }
        if ($file->resource == 'hpss-resc') {
            $result[$file->name]['archived']      = 'Yes' ;
            $result[$file->name]['archived_flag'] = 1 ;
        }
        if ($file->resource == 'lustre-resc') {
            $result[$file->name]['local']      = 'Yes' ;
            $result[$file->name]['local_flag'] = 1 ;
            $result[$file->name]['local_path'] = $file->path ;
            $result[$file->name]['created']    = $file->ctime ;
        }
        if ($file->checksum) {
            $result[$file->name]['checksum'] = $file->checksum ;
        }
    }

    /* Filter out result by eliminating files which do not pass the archived and/or local
     * filter requirements
     */
    foreach ($result as $file) {
        if ((!is_null($checksum) && ($checksum ^ ($file['checksum'] != ''))) ||
            (!is_null($archived) && ($archived ^  $file['archived_flag'])) ||
            (!is_null($local)    && ($local    ^  $file['local_flag']))) continue ;
        $files[$file['name']] = $file ;
    }
}

function allowed_stay ($expire_ctime, $sec) {
    $expire_time = new LusiTime($expire_ctime, 0) ;
    $expire_day_str = $expire_time->toStringDay() ;
    if ($sec < SECONDS_IN_HOUR) {
        return array (
            'expiration'   => "<span style=\"color:red\">{$expire_day_str}</span>" ,
            'allowed_stay' => "<span style=\"color:red\">expired</span>"
       ) ;
    }
    if ($sec < SECONDS_IN_WEEK) {
        return array (
            'expiration'   => $expire_day_str,
            'allowed_stay' => intval($sec / (SECONDS_IN_DAY))." days"
       ) ;
    }
    return array (
        'expiration'   => $expire_day_str,
        'allowed_stay' => intval($sec / (SECONDS_IN_WEEK))." weeks"
   ) ;
}

/**
 * Get quota policies. First use the system wide default. Then check
 * if there are any experiment-specific ovverides and apply them (if
 * any found).
 *
 * @param ServiceJSON       $SVC
 * @param LogBookExperiment $experiment
 * @return array
 */
function retention_policy ($SVC, $experiment) {

    function policy_time (
        $SVC ,
        $experiment ,
        $storage_class ,
        $policy ,
        $experiment_storage_policy) {

        $global_time  = LusiTime::parse($SVC       ->configdb()        ->get_policy_param  ($storage_class, $policy)) ;
        $param =                        $experiment->regdb_experiment()->find_param_by_name($experiment_storage_policy) ;
        if ($param) {
            $local_time = LusiTime::parse($param->value()) ;
            if ($local_time) return $local_time ;
        }
        return $global_time ;
    }
    function policy_int (
        $SVC ,
        $experiment ,
        $storage_class ,
        $policy ,
        $experiment_storage_policy) {

        $global = intval($SVC       ->configdb()        ->get_policy_param  ($storage_class, $policy)) ;
        $param  =        $experiment->regdb_experiment()->find_param_by_name($experiment_storage_policy) ;
        if ($param) {
            $local = intval($param->value()) ;
            if ($local) return $local ;
        }
        return $global ;
    }
    return array (
        'short_quota_ctime_time'  => policy_time ($SVC, $experiment, 'SHORT-TERM',  'CTIME',     'SHORT-TERM-DISK-QUOTA-CTIME') ,
        'short_retention_months'  => policy_int  ($SVC, $experiment, 'SHORT-TERM',  'RETENTION', 'SHORT-TERM-DISK-QUOTA-RETENTION') ,
        'medium_quota_ctime_time' => policy_time ($SVC, $experiment, 'MEDIUM-TERM', 'CTIME',     'MEDIUM-TERM-DISK-QUOTA-CTIME') ,
        'medium_retention_months' => policy_int  ($SVC, $experiment, 'MEDIUM-TERM', 'RETENTION', 'MEDIUM-TERM-DISK-QUOTA-RETENTION') ,
        'medium_quota_gb'         => policy_int  ($SVC, $experiment, 'MEDIUM-TERM', 'QUOTA',     'MEDIUM-TERM-DISK-QUOTA')
    ) ;
}

/**
 * Return the summary statistics for all known runs and file types
 *
 * @param ServiceJSON       $SVC
 * @param LogBookExperiment $experiment
 * @return array
 */
function summary_stats ($SVC, $experiment) {

    $first_run = $experiment->find_first_run() ;
    $last_run  = $experiment->find_last_run() ;

    $min_run   = 0 ; // will be calculated
    $max_run   = 0 ; // will be calculated

    $xtc_size       = 0.0 ;
    $xtc_num_files  = 0 ;
    $xtc_archived   = 0 ;
    $xtc_local_copy = 0 ;

    $hdf5_size       = 0.0 ;
    $hdf5_num_files  = 0 ;
    $hdf5_archived   = 0 ;
    $hdf5_local_copy = 0 ;

    $unique_runs  = array () ;  // 
    $unique_files = array () ;  // to avoid double counting

    foreach ($SVC->irodsdb()->runs($experiment->instrument()->name(), $experiment->name(), 'xtc') as $run) {
        
        $runnum = intval($run->run) ;
        if (!$min_run || $runnum < $min_run) $min_run = $runnum ;
        if (!$max_run || $runnum > $max_run) $max_run = $runnum ;

        $unique_runs[$runnum] = $runnum ;

        $files = $run->files ;
        foreach ($files as $file) {
            if (!array_key_exists($file->name, $unique_files)) {
                $unique_files[$file->name] = $run->run ;
                $xtc_num_files++ ;
                $xtc_size += $file->size / (GB) ;
            }
            if ($file->resource === 'hpss-resc'  ) $xtc_archived++ ;
            if ($file->resource === 'lustre-resc') $xtc_local_copy++ ;
        }
    }

    // Adjust the total number of files using the one reported by the DAQ
    // if some files haven't been migrated or registered in the iRODS DB.
    //
    // NOTE: This correction will not work properly if the experiment
    //       has adopted runs/files from some other experiment.

    $xtc_num_files_DAQ = (is_null($first_run) || is_null($last_run)) ? 0 : count($experiment->regdb_experiment()->files()) ;
    if ($xtc_num_files_DAQ > $xtc_num_files) $xtc_num_files = $xtc_num_files_DAQ ;

    foreach ($SVC->irodsdb()->runs($experiment->instrument()->name(), $experiment->name(), 'hdf5') as $run) {

        $runnum = intval($run->run) ;
        if (!$min_run || $runnum < $min_run) $min_run = $runnum ;
        if (!$max_run || $runnum > $max_run) $max_run = $runnum ;

        $files = $run->files ;
        foreach ($files as $file) {
            if (!array_key_exists($file->name, $unique_files)) {
                $unique_files[$file->name] = $run->run ;
                $hdf5_num_files++ ;
                $hdf5_size += $file->size / (GB) ;
            }
            if ($file->resource === 'hpss-resc'  ) $hdf5_archived++ ;
            if ($file->resource === 'lustre-resc') $hdf5_local_copy++ ;
        }
    }

    return array (
        "runs"    => count($unique_runs) ,//$num_runs ,
        "min_run" => $min_run ,
        "max_run" => $max_run ,

        "xtc" => array (
            "size"          => sprintf("%.0f", $xtc_size) ,
            "files"         => $xtc_num_files ,
            "archived"      => $xtc_archived ,
            "archived_html" => $xtc_num_files ?
                $xtc_num_files == $xtc_archived ?
                    '100%' :
                    '<span style="color:red;">' . $xtc_archived.'</span> / ' . $xtc_num_files :
                '' ,
            "disk"          => $xtc_local_copy ,
            "disk_html"     => $xtc_num_files ?
                $xtc_num_files == $xtc_local_copy ?
                    '100%' :
                    '<span style="color:red;">' .
                        sprintf("%2.0f", floor(100.0 * $xtc_local_copy / $xtc_num_files)) .
                        '%</span> (' . $xtc_local_copy . ' / ' . $xtc_num_files.')' :
                ''
        ) ,

        "hdf5" => array (
            "size"          => sprintf("%.0f", $hdf5_size) ,
            "files"         => $hdf5_num_files ,
            "archived"      => $hdf5_archived ,
            "archived_html" => $hdf5_num_files ?
                $hdf5_num_files == $hdf5_archived ?
                    '100%' :
                    '<span style="color:red;">' . $hdf5_archived.'</span> /  '. $hdf5_num_files :
                '' ,
            "disk"          => $hdf5_local_copy ,
            "disk_html"     => $hdf5_num_files ?
                $hdf5_num_files == $hdf5_local_copy ?
                    '100%' :
                    '<span style="color:red;">' .
                        sprintf("%2.0f", floor(100.0 * $hdf5_local_copy / $hdf5_num_files)) .
                        '%</span> (' .
                        $hdf5_local_copy . ' / '.$hdf5_num_files.')' :
                ''
        )
    ) ;
}

/**
 * Get the migration delay of the known files of the experiment
 * and prepare a dictionary mapping file names into the correponding
 * objects. If the file is found in the database then possible values
 * for the migration delay can be:
 *
 * - return null if the migration hasn't started
 * - otherwise it has to be a positive number (0 or more)
 * - negative values aren't allowed
 *
 * TODO: Make sure the result also accounts for runs/files 'adopted' from
 *       other experiments.
 *
 * @param LogBookExperiment $experiment
 * @return array
 */
function get_migration_delay($experiment) {


    $migration_status = array () ;
    foreach ($experiment->regdb_experiment()->data_migration_files() as $file) {
        $migration_status[$file->name()] = $file ;
    }
    $migration_delay = array () ;
    foreach ($experiment->regdb_experiment()->files() as $file) {
        $filename =
            sprintf("e%d-r%04d-s%02d-c%02d.xtc" ,
                $experiment->id() ,
                $file->run() ,
                $file->stream() ,
                $file->chunk()) ;
        if (array_key_exists($filename, $migration_status)) {
            $start_time = $migration_status[$filename]->start_time() ;
            if ($start_time) {
                $seconds = $start_time->to_float() - $file->open_time()->to_float() ;
                if ($seconds < 0) $seconds = 0. ;
                $migration_delay[$filename] = intval($seconds) ;
            } else {
                $migration_delay[$filename] = null ;
            }
        }
    }
    return $migration_delay ;
}

function handler ($SVC) {
    
    $KNOWN_TYPES = array('xtc', 'hdf5') ;

    // -------------------------------------
    // Parse input parameters of the request
    // -------------------------------------
    
    $exper_id      = $SVC->required_int  ('exper_id') ;
    $types         = $SVC->optional_list ('types',    $KNOWN_TYPES ,    // allowed values
                                                      $KNOWN_TYPES ,    // default
                                                      array('ignore_case'     => true ,         // when comparing parameters
                                                            'convert'         => 'tolower' ,    // before storing elements in the result list
                                                            'skip_duplicates' => true           // to prevent duplicates
                                                      )) ;
    $range_of_runs = $SVC->optional_range('runs',     array('min' => null, 'max' => null)) ;
    $checksum      = $SVC->optional_enum ('checksum', array('0','1'), null) ;
    $archived      = $SVC->optional_enum ('archived', array('0','1'), null) ;
    $local         = $SVC->optional_enum ('local',    array('0','1'), null) ;
    $storage       = $SVC->optional_enum ('storage',  array('SHORT-TERM', 'MEDIUM-TERM'), null) ;

    $now = LusiTime::now();

    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     "no experiment found for id={$exper_id}") ;

    $instrument = $experiment->instrument() ;

    $policy = retention_policy ($SVC, $experiment) ;

    $migration_delay = get_migration_delay($experiment) ;

    // -----------------------------------------------------------
    // Compile detailed information about files in a range of runs
    // ------------------------------------------------------------

    // Build two structures:
    //
    //   - a mapping from file names to the corresponding run numbers.
    //     This information will be shown in the GUI.
    //
    //   - a list of all known files

    $files_reported_by_iRODS = array () ;
    $file2run                = array () ;
    $files                   = array () ;

    foreach ($types as $type) {
        foreach (
            $SVC->irodsdb()->runs (
                $instrument->name() ,
                $experiment->name() ,
                $type ,
                $range_of_runs['min'] ,
                $range_of_runs['max']) as $run) {

            foreach ($run->files as $file) {
                $files_reported_by_iRODS [$file->name] = True ;
                $file2run                [$file->name] = $run->run ;
            }
            add_files($files, $run->files, $type, $file2run, $checksum, $archived, $local) ;
        }
    }

    // Build a map in which run numbers will be keys and lists of the corresponding
    // file descriptions will be the values.

    $files_by_runs = array () ;
    foreach ($files as $file) {
        $runnum = intval($file['run']) ;
        if (!array_key_exists($runnum, $files_by_runs)) {
            $files_by_runs[$runnum] = array () ;
        }
        array_push($files_by_runs[$runnum], $file) ;
    }
    
    // Calculate a range of runs in iRODS files. Note that some of those
    // runs/files might be adopted from other experiments. This may result
    // in gaps in the range, as well as overlaps for some run numbers.
    
    $run_numbers_in_irods_files = array_keys($files_by_runs) ;
    sort($run_numbers_in_irods_files) ;

    $min_run = count($run_numbers_in_irods_files) ? $run_numbers_in_irods_files[0] : 0 ;
    $max_run = count($run_numbers_in_irods_files) ? $run_numbers_in_irods_files[count($run_numbers_in_irods_files)-1] : 0 ;

    // Make proper adjustments to the range of runs passed as a parameter
    // to the service.

    if (is_null($range_of_runs['min']) || ($range_of_runs['min'] < $min_run)) $range_of_runs['min'] = $min_run ;
    if (is_null($range_of_runs['max']) || ($range_of_runs['max'] > $max_run)) $range_of_runs['max'] = $max_run ;

    // Postprocess the above created array to missing gaps for runs
    // with empty collections of files.

    for($runnum = $range_of_runs['min']; $runnum <= $range_of_runs['max']; $runnum++) {
        if (!array_key_exists($runnum, $files_by_runs)) {
            $files_by_runs[$runnum] = array () ;
        }
    }

    $total_size_gb           = 0 ;
    $overstay_by_storage_run = array () ;

    $nonempty_runs = array () ;

    for ($runnum = $range_of_runs['min']; $runnum <= $range_of_runs['max']; $runnum++) {

        $run_entries = array () ;

        // Stage I: iRODS files
        // 
        // At this step only consider runs for which iRODS file have been found.

        if (array_key_exists($runnum, $files_by_runs)) {

            foreach ($files_by_runs[$runnum] as $file) {

                $file_storage = 'SHORT-TERM' ;
                if ($file['local_flag']) {
                    $request = $SVC->configdb()->find_medium_store_file(
                        array (
                            'exper_id'       => $exper_id ,
                            'runnum'         => $runnum ,
                            'file_type'      => strtoupper($file['type']) ,
                            'irods_filepath' => $file['irods_filepath'] ,
                            'irods_resource' => 'lustre-resc'
                       )
                   ) ;
                    if (!is_null($request)) $file_storage = 'MEDIUM-TERM' ;
                }
                if (!is_null($storage) && ($storage != $file_storage)) continue ;

                $filename = $file['name'] ;

                $bytes = $file['size'] ;
                $size_gb = $bytes / (GB) ;
                $total_size_gb += $size_gb ;

                // Override the file creation for the purposes of calculating the file expiration
                // parameters in case if such override is registered for the experiment. Note that
                // each storage gets a separate override (if any).

                $short_ctime = intval($file['created']) ;
                if ($policy['short_quota_ctime_time'] &&
                   ($short_ctime < $policy['short_quota_ctime_time']->sec)) $short_ctime = $policy['short_quota_ctime_time']->sec ;

                $medium_ctime = intval($file['created']) ;
                if ($policy['medium_quota_ctime_time'] &&
                   ($medium_ctime < $policy['medium_quota_ctime_time']->sec)) $medium_ctime = $policy['medium_quota_ctime_time']->sec ;

                $short_allowed_stay_sec  = $file['local_flag'] ? max(0, $policy['short_retention_months' ] * SECONDS_IN_MONTH - ($now->sec - $short_ctime )) : 0 ;
                $medium_allowed_stay_sec = $file['local_flag'] ? max(0, $policy['medium_retention_months'] * SECONDS_IN_MONTH - ($now->sec - $medium_ctime)) : 0 ;

                $short_expire_ctime  = $short_ctime  + $policy['short_retention_months' ] * SECONDS_IN_MONTH ;
                $medium_expire_ctime = $medium_ctime + $policy['medium_retention_months'] * SECONDS_IN_MONTH ;

                $allowed_stay = array (
                    'SHORT-TERM' => array_merge (
                        array ('seconds' => $short_allowed_stay_sec),
                        $file['local_flag'] ?
                            allowed_stay($short_expire_ctime, $short_allowed_stay_sec) :
                            array (
                                'expiration'   => '',
                                'allowed_stay' => ''
                           )
                   ) ,
                   'MEDIUM-TERM' => array_merge (
                        array ('seconds' => $medium_allowed_stay_sec) ,
                        $file['local_flag'] ?
                            allowed_stay($medium_expire_ctime, $medium_allowed_stay_sec) :
                            array (
                                'expiration'   => '' ,
                                'allowed_stay' => ''
                           )
                    )
                ) ;
                if ($file['local_flag'] && ($allowed_stay[$file_storage]['seconds'] < SECONDS_IN_HOUR)) {

                    if (!array_key_exists($file_storage, $overstay_by_storage_run))
                        $overstay_by_storage_run[$file_storage] = array (
                            'total_runs'    => 0 ,
                            'total_files'   => 0 ,
                            'total_size_gb' => 0 ,
                            'runs'          => array ()
                        ) ;

                    if (!array_key_exists($runnum, $overstay_by_storage_run[$file_storage])) {
                        $overstay_by_storage_run[$file_storage]['total_runs'] += 1 ;
                        $overstay_by_storage_run[$file_storage]['runs'][$runnum] = array (
                            'files'   => 0 ,
                            'size_gb' => 0
                        ) ;
                    }
                    $overstay_by_storage_run[$file_storage]['total_files'  ]            += 1 ;
                    $overstay_by_storage_run[$file_storage]['total_size_gb']            += $size_gb ;
                    $overstay_by_storage_run[$file_storage]['runs'][$runnum]['files'  ] += 1 ;
                    $overstay_by_storage_run[$file_storage]['runs'][$runnum]['size_gb'] += $size_gb ;
                }

                $entry = array (
                    'runnum'                 => $runnum ,
                    'filename'               => $filename ,
                    'name'                   => $filename ,
                    'storage'                => $file_storage ,
                    'type'                   => strtoupper($file['type']) ,
                    'size_auto'              => autoformat_size($bytes) ,
                    'size_bytes'             => $bytes ,
                    'size_kb'                => sprintf($bytes < 10 * KB ? "%.1f" : "%d", $bytes / KB) ,
                    'size_mb'                => sprintf($bytes < 10 * MB ? "%.1f" : "%d", $bytes / MB) ,
                    'size_gb'                => sprintf($bytes < 10 * GB ? "%.1f" : "%d", $bytes / GB) ,
                    'size'                   => number_format($bytes) ,
                    'created'                => date("Y-m-d", $file['created']).'&nbsp;&nbsp;&nbsp;'.date("H:i:s", $file['created']) ,
                    'created_seconds'        => $file['created'] ,
                    'archived'               => $file['archived'] ,
                    'archived_flag'          => $file['archived_flag'] ,
                    'local'                  =>
                        $file['local_flag'] ?
                        '<a class="link" href="javascript:display_path('."'".$file['local_path']."'".')">path</a>' :
                        $file['local'] ,
                    'local_flag'             => $file['local_flag'] ,
                    'checksum'               => $file['checksum'] ,

                    'allowed_stay'           => $allowed_stay ,

                    'restore_flag'           => 0 ,
                    'restore_requested_time' => '' ,
                    'restore_requested_uid'  => ''
                ) ;
                if (($file_storage == 'SHORT-TERM') && !$file['local_flag']) {
                    $request = $SVC->configdb()->find_file_restore_request(
                        array (
                            'exper_id'           => $exper_id ,
                            'runnum'             => $runnum ,
                            'file_type'          => $file['type'] ,
                            'irods_filepath'     => $file['irods_filepath'] ,
                            'irods_src_resource' => 'hpss-resc' ,
                            'irods_dst_resource' => 'lustre-resc '
                       )
                    ) ;
                    if (!is_null($request)) {
                        $entry['local'] = '<span style="color:black;">Restoring from tape...</span>' ;
                        $entry['restore_flag'] = 1 ;
                        $entry['restore_requested_time'] = $request['requested_time']->toStringShort() ;
                        $entry['restore_requested_uid']  = $request['requested_uid'] ;
                    }
                }
                if (array_key_exists($filename, $migration_delay)) {
                    $start_migration_delay_sec = $migration_delay[$filename] ;
                    if (!is_null($start_migration_delay_sec))
                        $entry['start_migration_delay_sec'] = $start_migration_delay_sec ;
                }
                array_push($run_entries, $entry) ;
            }
        }

        // Stage II: files which are still being (or a supposed to be) migrated
        //
        // Add XTC files which haven't been reported to iRODS because they have either
        // never migrated from ONLINE or because they have been permanently deleted.

        if (in_array ('xtc', $types) && (is_null($storage) || ($storage == 'SHORT-TERM'))) {

            foreach ($experiment->regdb_experiment()->files($runnum) as $file) {

                $filename = sprintf("e%d-r%04d-s%02d-c%02d.xtc" ,
                                $experiment->id() ,
                                $file->run() ,
                                $file->stream() ,
                                $file->chunk()) ;

                $allowed_stay = array (
                    'SHORT-TERM' => array (
                        'seconds'      => 0 ,
                        'expiration'   => '' ,
                        'allowed_stay' => ''
                    ) ,
                    'MEDIUM-TERM' => array (
                        'seconds'      => 0 ,
                        'expiration'   => '' ,
                        'allowed_stay' => ''
                    )
                ) ;
                if (!array_key_exists($filename, $files_reported_by_iRODS)) {
                    $entry = array (
                        'runnum'                 => $runnum ,
                        'filename'               => $filename ,
                        'name'                   => '<span style="color:red;">'.$filename.'</span>' ,
                        'type'                   => 'XTC' ,
                        'storage'                => 'SHORT-TERM' ,
                        'size_auto'              => '' ,
                        'size_bytes'             => 0 ,
                        'size_kb'                => '' ,
                        'size_mb'                => '' ,
                        'size_gb'                => '' ,
                        'size'                   => '' ,
                        'created'                => date("Y-m-d H:i:s", $file->open_time()->sec) ,
                        'created_seconds'        => $file->open_time()->sec ,
                        'archived'               => '' ,
                        'archived_flag'          => 0 ,
                        'local'                  => '<span style="color:red;">never migrated from DAQ or deleted</span>' ,
                        'local_flag'             => 0 ,
                        'checksum'               => '' ,
                        'allowed_stay'           => $allowed_stay ,
                        'restore_flag'           => 0 ,
                        'restore_requested_time' => '' ,
                        'restore_requested_uid'  => ''
                    ) ;
                    if (array_key_exists($filename, $migration_delay)) {
                        $start_migration_delay_sec = $migration_delay[$filename] ;
                        if (!is_null($start_migration_delay_sec)) {
                            $entry['start_migration_delay_sec'] = $start_migration_delay_sec ;
                            $entry['local'] = '<span style="color:black;">is migrating from DAQ to OFFLINE...</span>' ;
                        }
                    }
                    array_push($run_entries, $entry) ;
                }
            }
        }

        // Make a summary entry for all files (of any status) for the current run.
        // This is what will be reported back to this Web service's clients.

        if (count($run_entries)) {
            $run = $experiment->find_run_by_num($runnum) ;
            $run_url = is_null($run) ?
                $runnum :
                '<a class="link" href="javascript:global_elog_search_run_by_num('.$run->num().', true)" title="click to see a LogBook record for this run">'.$runnum.'</a>' ;
            array_push(
                $nonempty_runs,
                array (
                    'url'    => $run_url ,
                    'runnum' => $runnum ,
                    'files'  => $run_entries
                )
            ) ;
        }
    }

    return array (

        "summary"       => summary_stats($SVC, $experiment) ,

        "runs"          => $nonempty_runs ,
        "total_size_gb" => sprintf("%.0f", $total_size_gb) ,
        "overstay"      => $overstay_by_storage_run ,

        "policies" => array (
            "SHORT-TERM" => array (
                "retention_months" => $policy['short_retention_months']
            ) ,
            "MEDIUM-TERM" => array (
                "retention_months" => $policy['medium_retention_months'] ,
                "quota_gb"         => $policy['medium_quota_gb'] ,
                "quota_used_gb"    => $SVC->configdb()->calculate_medium_quota($experiment->id())
            )
        )
    ) ;
}

\DataPortal\ServiceJSON::run_handler ('GET', 'handler') ;

?>
