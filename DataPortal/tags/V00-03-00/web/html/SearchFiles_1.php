<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

use DataPortal\Config;
use DataPortal\DataPortalException;

define( 'KB', 1024.0 );
define( 'MB', 1024.0 * 1024.0 );
define( 'GB', 1024.0 * 1024.0 * 1024.0 );

define( 'SECONDS_IN_3_MONTHS', 3 * 31 * 24 * 3600 );

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past


/* Package the error message into a JSON object and return the one back
 * to a caller. The script's execution will end at this point.
 */ 
function report_error( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg );
    print <<< HERE
{
  "Status": {$status_encoded},
  "Message": {$msg_encoded}
}
HERE;
    exit;
}

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned an embedable HTML element (<div>).
 */
if( !isset( $_GET['exper_id'] )) report_error( "no valid experiment identifier in the request" );
$exper_id = trim( $_GET['exper_id'] );

/*
 * Oprtional parameter of the filter:
 *
 * runs     - a range of runs
 * type     - file type (allowed values 'xtc', 'hdf5'. The case doesn't matter.)
 * archived - a flag indicating if the files are archived (allowed values: '0', '1')
 * local    - a flag indicating if there is a local copy of files (allowed values: '0', '1')
 */
$range_of_runs=null;
if( isset( $_GET['runs'] )) {
    $range_of_runs = trim( $_GET['runs'] );
    if( $range_of_runs == '' )
        report_error( 'run range parameter shall not have an empty value' );
}

$types = null;
if( isset( $_GET['types'] )) {
    $types = explode( ',', strtolower( trim( $_GET['types'] )));
}
if( is_null( $types ) || !count( $types )) {
    $types = array( 'xtc', 'hdf5' );
}

$checksum = null;
if( isset( $_GET['checksum'] )) {
    $str = trim( $_GET['checksum'] );
    if( $str == '1' ) $checksum = true;
    else if( $str == '0' ) $checksum = false;
    else {
        report_error( 'unsupported value of the checksum parameter: '.$str );
    }
}

$archived = null;
if( isset( $_GET['archived'] )) {
    $str = trim( $_GET['archived'] );
    if( $str == '1' ) $archived = true;
    else if( $str == '0' ) $archived = false;
    else {
        report_error( 'unsupported value of the archived parameter: '.$str );
    }
}

$local = null;
if( isset( $_GET['local'] )) {
    $str = trim( $_GET['local'] );
    if( $str == '1' ) $local = true;
    else if( $str == '0' ) $local = false;
    else {
        report_error( 'unsupported value of the local parameter: '.$str );
    }
}

$storage = isset( $_GET['storage'] ) ? trim( $_GET['storage'] ) : null;

function pre($str, $width=null) {
    return '<pre>'.(is_null($width) ? $str : sprintf("%{$width}s", $str)).'</pre>'; }

function autoformat_size($bytes) {
    if     ( $bytes < KB ) return sprintf(                               "%d   ", $bytes );
    else if( $bytes < MB ) return sprintf($bytes < 10 * KB ? "%.1f KB" : "%d KB", $bytes / KB);
    else if( $bytes < GB ) return sprintf($bytes < 10 * MB ? "%.1f MB" : "%d MB", $bytes / MB);
	else                   return sprintf($bytes < 10 * GB ? "%.1f GB" : "%d GB", $bytes / GB);
}

function add_files(&$files, $infiles, $type, $file2run, $checksum, $archived, $local) {
    
    $result = array();

    foreach( $infiles as $file ) {

        if( $file->type == 'collection' ) continue;

        // Skip files for which we don't know run numbers
        //
        if( !array_key_exists( $file->name, $file2run )) continue;
        
        if( !array_key_exists( $file->name, $result )) {
            $result[$file->name] =
                array (
                    'name'           => $file->name,
                    'irods_filepath' => "{$file->collName}/{$file->name}",
                    'run'            => $file2run[$file->name],
                	'type'           => $type,
                    'size'           => $file->size,
                    'created'        => $file->ctime,
                    'archived'       => '<span style="color:red;">No</span>',
                    'archived_flag'  => 0,
                    'local'          => '<span style="color:red;">No</span>',
                    'local_flag'     => 0,
                    'checksum'       => '' );    
        }
        if( $file->resource == 'hpss-resc' ) {
            $result[$file->name]['archived']      = 'Yes';
            $result[$file->name]['archived_flag'] = 1;
        }
        if( $file->resource == 'lustre-resc' ) {
            $result[$file->name]['local']      = 'Yes';
            $result[$file->name]['local_flag'] = 1;
            $result[$file->name]['local_path'] = $file->path;
            $result[$file->name]['created']    = $file->ctime;
        }
        if( $file->checksum ) {
            $result[$file->name]['checksum'] = $file->checksum;
        }
    }

    /* Filter out result by eliminating files which do not pass the archived and/or local
     * filter requirements
     */
    foreach( $result as $file ) {
        if(( !is_null( $checksum ) && ( $checksum ^ ( $file['checksum'] != '' ))) ||
    	   ( !is_null( $archived ) && ( $archived ^   $file['archived_flag'] )) ||
           ( !is_null( $local )    && ( $local    ^   $file['local_flag'] ))) continue;
        $files[$file['name']] = $file;
    }
}

$allowed_stay4storage_defaults = array (
    'SHORT-TERM'  =>          SECONDS_IN_3_MONTHS,
    'MEDIUM-TERM' =>  1 * 4 * SECONDS_IN_3_MONTHS       // 1 year for the testing purposes
);
function allowed_stay( $sec ) {
    if( $sec <                3600 ) return '<span style="color:red">OVERSTAY</span>';
    if( $sec <           24 * 3600 ) return intval( $sec / (                3600 ))."h";
    if( $sec <       7 * 24 * 3600 ) return intval( $sec / (           24 * 3600 ))."d";
    if( $sec <      31 * 24 * 3600 ) return intval( $sec / (       7 * 24 * 3600 ))."w ".intval(( $sec % (       7 * 24 * 3600 )) / (      24 * 3600 ))."d";
    if( $sec < 12 * 31 * 24 * 3600 ) return intval( $sec / (      31 * 24 * 3600 ))."m ".intval(( $sec % (      31 * 24 * 3600 )) / (  7 * 24 * 3600 ))."w";
    return                                  intval( $sec / ( 12 * 31 * 24 * 3600 ))."y ".intval(( $sec % ( 12 * 31 * 24 * 3600 )) / ( 31 * 24 * 3600 ))."m";
}

/*
 * Analyze and process the request
 */
try {

    $now = LusiTime::now();

    $logbook = new LogBook();
    $logbook->begin();

    $config = Config::instance();
    $config->begin();

    $experiment = $logbook->find_experiment_by_id( $exper_id ) or report_error("No such experiment");
    $instrument = $experiment->instrument();

    $first_run = $experiment->find_first_run();
    $last_run  = $experiment->find_last_run();
    $num_runs  = $experiment->num_runs();
    $min_run   = is_null( $first_run ) ? 0 : $first_run->num();
    $max_run   = is_null( $last_run ) ? 0 : $last_run->num();

    /* Compile the summary data counters.
     */
    $xtc_size       = 0.0;
    $xtc_num_files  = 0;
    $xtc_archived   = 0;
    $xtc_local_copy = 0;

    $hdf5_size       = 0.0;
    $hdf5_num_files  = 0;
    $hdf5_archived   = 0;
    $hdf5_local_copy = 0;

    $runs2files = array();

    $range_xtc = FileMgrIrodsWs::run_range(
        $experiment->instrument()->name(),
        $experiment->name(),
        'xtc');

    if( $range_xtc['total'] > 0 ) {

        $xtc_runs = null;

        FileMgrIrodsWs::runs(
            $xtc_runs,
            $experiment->instrument()->name(),
            $experiment->name(),
            'xtc',
            $range_xtc['min'].'-'.$range_xtc['max'] );

        foreach( $xtc_runs as $run ) {
            $unique_files = array();  // per this run
            $files = $run->files;
            foreach( $files as $file ) {
                if( !array_key_exists( $file->name, $unique_files )) {
                    $unique_files[$file->name] = $run->run;
                    $xtc_num_files++;
                    $xtc_size += $file->size / (1024.0 * 1024.0 * 1024.0);
                    if( !array_key_exists( $run->run, $runs2files )) $runs2files[$run->run] = array('run'=>$run->run,'xtc'=>array(),'hdf5'=>array());
                    array_push( $runs2files[$run->run]['xtc'], $file );
                }
                if( $file->resource == 'hpss-resc'   ) $xtc_archived++;
                if( $file->resource == 'lustre-resc' ) $xtc_local_copy++;
            }
        }
    }
    $xtc_num_files_DAQ = ( is_null( $first_run ) || is_null( $last_run )) ? 0 : count( $experiment->regdb_experiment()->files());    
    if( $xtc_num_files_DAQ > $xtc_num_files ) $xtc_num_files = $xtc_num_files_DAQ;

    $range_hdf5 = FileMgrIrodsWs::run_range(
        $experiment->instrument()->name(),
        $experiment->name(),
        'hdf5');

    if( $range_hdf5['total'] > 0 ) {

        $hdf5_runs = null;

        FileMgrIrodsWs::runs(
            $hdf5_runs,
            $experiment->instrument()->name(),
            $experiment->name(),
            'hdf5',
            $range_hdf5['min'].'-'.$range_hdf5['max'] );

        foreach( $hdf5_runs as $run ) {

            $unique_files = array();  // per this run
            $files = $run->files;
            foreach( $files as $file ) {
                if( !array_key_exists( $file->name, $unique_files )) {
                    $unique_files[$file->name] = $run->run;
                    $hdf5_num_files++;
                    $hdf5_size += $file->size / (1024.0 * 1024.0 * 1024.0);
                    if( !array_key_exists( $run->run, $runs2files )) $runs2files[$run->run] = array('run'=>$run->run,'xtc'=>array(),'hdf5'=>array());
                    array_push( $runs2files[$run->run]['hdf5'], $file );
                }
                if( $file->resource == 'hpss-resc'   ) $hdf5_archived++;
                if( $file->resource == 'lustre-resc' ) $hdf5_local_copy++;
            }
        }
    }
    $xtc_size_str    = json_encode( sprintf( "%.0f", $xtc_size ));
    $hdf5_size_str   = json_encode( sprintf( "%.0f", $hdf5_size ));
    $runs_encoded    = json_encode( $runs2files );

    $xtc_archived_html = json_encode(
    	$xtc_num_files == 0 ?
    	'n/a' : (	$xtc_num_files == $xtc_archived ?
    				'100%' :
    				'<span style="color:red;">'.$xtc_archived.'</span> / '.$xtc_num_files )
   	);
    $xtc_local_copy_html = json_encode(
    	$xtc_num_files == 0 ?
    	'n/a' : (	$xtc_num_files == $xtc_local_copy ?
			    	'100%' :
    				'<span style="color:red;">'.sprintf("%2.0f", floor( 100.0*$xtc_local_copy/$xtc_num_files )).'%</span> ( '.$xtc_local_copy.' / '.$xtc_num_files.' )' )
   	);
    $hdf5_archived_html = json_encode(
    	$hdf5_num_files == 0 ?
   		'n/a' : (	$hdf5_num_files == $hdf5_archived ?
    				'100%' :
    				'<span style="color:red;">'.$hdf5_archived.'</span> / '.$hdf5_num_files )
   	);
    $hdf5_local_copy_html = json_encode(
    	$hdf5_num_files == 0 ?
   		'n/a' : (	$hdf5_num_files == $hdf5_local_copy ?
   					'100%' :
   					'<span style="color:red;">'.sprintf("%2.0f", floor( 100.0*$hdf5_local_copy/$hdf5_num_files )).'%</span> ( '.$hdf5_local_copy.' / '.$hdf5_num_files.' )' )
    );

    /* Get the migration delay of the known files of the experiment
     * and prepare a dictionary mapping file names into the correponding
     * objects. If the file is found in the database then possible values
     * for the migration delay can be:
     *
     * - return null if the migration hasn't started
     * - otherwise it has to be a positive number (0 or more)
     * - negative values aren't allowed
     */
    $migration_status = array();
    foreach( $experiment->regdb_experiment()->data_migration_files() as $file )
        $migration_status[$file->name()] = $file;

    $migration_delay = array();
    foreach( $experiment->regdb_experiment()->files() as $file ) {
        $filename =
            sprintf("e%d-r%04d-s%02d-c%02d.xtc",
                $experiment->id(),
                $file->run(),
                $file->stream(),
                $file->chunk());
        if( array_key_exists($filename, $migration_status)) {
            $start_time = $migration_status[$filename]->start_time();
            if( $start_time ) {
                $seconds = $start_time->to_float() - $file->open_time()->to_float();
                if( $seconds < 0 ) $seconds = 0.;
                $migration_delay[$filename] = intval($seconds);
            } else {
                $migration_delay[$filename] = null;
            }
        }
    }

    /* If no specific run range is provided find out the one by probing all
     * known file types.
     */
    $effective_range_of_runs = $range_of_runs;
    if( is_null( $effective_range_of_runs )) {
        $effective_range_of_runs = ( is_null($first_run) || is_null( $last_run )) ? '0-0' : $first_run->num().'-'.$last_run->num();
    }

    /* Build two structures:
     * - a mapping from file names to the corresponding run numbers. This information will be shown in the GUI.
     * - a list fop all known files.
     */
    $files_reported_by_iRODS = array();
    $file2run = array();
    $files    = array();
    foreach( $types as $type ) {
        if(( $type == 'xtc'  ) and ( $range_xtc ['total'] == 0 )) continue;
        if(( $type == 'hdf5' ) and ( $range_hdf5['total'] == 0 )) continue;
        $runs = null;
        FileMgrIrodsWs::runs( $runs, $instrument->name(), $experiment->name(), $type, $effective_range_of_runs );
        if( !is_null( $runs ))
	        foreach( $runs as $run ) {
    	        foreach( $run->files as $file ) {
    	        	$file2run[$file->name] = $run->run;
    	        	$files_reported_by_iRODS[$file->name] = True;
    	        }
        	    add_files( $files, $run->files, $type, $file2run, $checksum, $archived, $local );
        	}
    }

    /* Build a map in which run numbers will be keys and lists of the corresponding
     * file descriptions will be the values.
     */
    $files_by_runs = array();
    foreach( $files as $file ) {
        $runnum = $file['run'];
        if( !array_key_exists( $runnum, $files_by_runs )) {
            $files_by_runs[$runnum] = array();
        }
        array_push( $files_by_runs[$runnum], $file );
    }

    /* Postprocess the above created array to missing gaps for runs
     * with empty collections of files.
     * 
     * DO THIS ONLY IF NO SPECIFIC RANGE OF RUNS WAS
     * PROVIDED TO THE SCRIPT!!!
     */
    if( is_null( $range_of_runs )) {
	    if( !( is_null( $first_run ) || is_null( $last_run ))) {
		    for( $runnum = $first_run->num(); $runnum <= $last_run->num(); $runnum++ ) {
    			if( !array_key_exists( $runnum, $files_by_runs )) {
    				$files_by_runs[$runnum] = array();
	    		}
		    }
	    }
    }

    $success_encoded = json_encode("success");
    $updated_str     = json_encode( $now->toStringShort());

    print <<< HERE
{ "Status": {$success_encoded},
  "updated": {$updated_str},
  "summary": {
    "runs": {$num_runs},
    "min_run" : {$min_run},
    "max_run" : {$max_run},
    "xtc" : { "size": {$xtc_size_str},  "files": {$xtc_num_files},  "archived": {$xtc_archived},  "archived_html": {$xtc_archived_html},  "disk": {$xtc_local_copy},  "disk_html": {$xtc_local_copy_html} },
    "hdf5": { "size": {$hdf5_size_str}, "files": {$hdf5_num_files}, "archived": {$hdf5_archived}, "archived_html": {$hdf5_archived_html}, "disk": {$hdf5_local_copy}, "disk_html": {$hdf5_local_copy_html} }
  },
  "runs": [
HERE;

    $total_size_gb   = 0;
    $overstay_by_storage_run = array();

    $first_run_entry = true;

    $run_numbers = array_keys( $files_by_runs );
    sort( $run_numbers, SORT_NUMERIC );

    foreach( $run_numbers as $runnum ) {

        $run_entries = array();

        foreach( $files_by_runs[$runnum] as $file ) {

            $file_storage = 'SHORT-TERM';
            if( $file['local_flag'] ) {
                $request = $config->find_medium_store_file(
                    array(
                        'exper_id'       => $exper_id,
                        'runnum'         => $runnum,
                        'file_type'      => strtoupper( $file['type'] ),
                        'irods_filepath' => $file['irods_filepath'],
                        'irods_resource' => 'lustre-resc'
                    )
                );
                if( !is_null($request)) $file_storage = 'MEDIUM-TERM';
            }
            if( !is_null($storage) && ( $storage != $file_storage )) continue;

            $filename = $file['name'];

            $bytes = $file['size'];
            $size_gb = $bytes / (1024.0 * 1024.0 * 1024.0);
            $total_size_gb += $size_gb;

            $allowed_stay_sec = $file['local_flag'] ? max( 0, $allowed_stay4storage_defaults[$file_storage] - ( $now->sec - $file['created'] )) : 0;
            $allowed_stay     = $file['local_flag'] ? allowed_stay( $allowed_stay_sec ) : '';
        
            if( $file['local_flag'] && ( $allowed_stay_sec < 3600 )) {

                if( !array_key_exists( $file_storage, $overstay_by_storage_run ))
                    $overstay_by_storage_run[$file_storage] = array(
                        'total_runs'    => 0,
                        'total_files'   => 0,
                        'total_size_gb' => 0,
                        'runs'          => array()
                    );

                if( !array_key_exists( $runnum, $overstay_by_storage_run[$file_storage] )) {
                    $overstay_by_storage_run[$file_storage]['total_runs'] += 1;
                    $overstay_by_storage_run[$file_storage]['runs'][$runnum] = array(
                        'files'   => 0,
                        'size_gb' => 0
                    );
                }
                $overstay_by_storage_run[$file_storage]['total_files'  ]            += 1;
                $overstay_by_storage_run[$file_storage]['total_size_gb']            += $size_gb;
                $overstay_by_storage_run[$file_storage]['runs'][$runnum]['files'  ] += 1;
                $overstay_by_storage_run[$file_storage]['runs'][$runnum]['size_gb'] += $size_gb;
            }

            $entry = array(
                'runnum'                 => $runnum,
                'filename'               => $filename,
                'name'                   => $filename,
                'storage'                => $file_storage,
                'type'                   => strtoupper( $file['type'] ),
                'size_auto'              => autoformat_size( $bytes ),
                'size_bytes'             => $bytes,
                'size_kb'                => sprintf( $bytes < 10 * KB ? "%.1f" : "%d", $bytes / KB ),
                'size_mb'                => sprintf( $bytes < 10 * MB ? "%.1f" : "%d", $bytes / MB ),
                'size_gb'                => sprintf( $bytes < 10 * GB ? "%.1f" : "%d", $bytes / GB ),
                'size'                   => number_format( $bytes ),
                'created'                => date( "Y-m-d H:i:s", $file['created'] ),
                'created_seconds'        => $file['created'],
                'archived'               => $file['archived'],
                'archived_flag'          => $file['archived_flag'],
                'local'                  =>
                    $file['local_flag'] ?
                    '<a class="link" href="javascript:display_path('."'".$file['local_path']."'".')">path</a>' :
                    $file['local'],
                'local_flag'             => $file['local_flag'],
                'checksum'               => $file['checksum'],
                'allowed_stay_sec'       => $allowed_stay_sec,
                'allowed_stay'           => $allowed_stay,

                'restore_flag'           => 0,
                'restore_requested_time' => '',
                'restore_requested_uid'  => ''
            );
            if(( $file_storage == 'SHORT-TERM' ) && !$file['local_flag'] ) {
                $request = $config->find_file_restore_request(
                    array(
                        'exper_id'           => $exper_id,
                        'runnum'             => $runnum,
                        'file_type'          => $file['type'],
                        'irods_filepath'     => $file['irods_filepath'],
                        'irods_src_resource' => 'hpss-resc',
                        'irods_dst_resource' => 'lustre-resc'
                    )
                );
                if( !is_null($request)) {
                    $entry['local'] = '<span style="color:black;">Restoring from archive...</span>';
                    $entry['restore_flag'] = 1;
                    $entry['restore_requested_time'] = $request['requested_time']->toStringShort();
                    $entry['restore_requested_uid'] = $request['requested_uid'];
                }
            }
            if(array_key_exists($filename, $migration_delay)) {
                $start_migration_delay_sec = $migration_delay[$filename];
                if( !is_null($start_migration_delay_sec))
                    $entry['start_migration_delay_sec'] = $start_migration_delay_sec;
            }
            array_push($run_entries, $entry);
        }

        /* Add XTC files which haven't been reported to iRODS because they have either
         * never migrated from ONLINE or because they have been permanently deleted.
         */
        if( in_array( 'xtc', $types ) && ( is_null($storage) || ( $storage == 'SHORT-TERM' ))) {

            foreach( $experiment->regdb_experiment()->files( $runnum ) as $file ) {

                $filename = sprintf("e%d-r%04d-s%02d-c%02d.xtc",
                                $experiment->id(),
                                $file->run(),
                                $file->stream(),
                                $file->chunk());

                if( !array_key_exists( $filename, $files_reported_by_iRODS )) {
                    $entry = array(
                        'runnum'                 => $runnum,
                        'filename'               => $filename,
                        'name'                   => '<span style="color:red;">'.$filename.'</span>',
                        'type'                   => 'XTC',
                        'storage'                => 'SHORT-TERM',
                        'size_auto'              => '<span style="color:red;">n/a</span>',
                        'size_bytes'             => 0,
                        'size_kb'                => '<span style="color:red;">n/a</span>',
                        'size_mb'                => '<span style="color:red;">n/a</span>',
                        'size_gb'                => '<span style="color:red;">n/a</span>',
                        'size'                   => '<span style="color:red;">n/a</span>',
                        'created'                => date( "Y-m-d H:i:s", $file->open_time()->sec ),
                        'created_seconds'        => $file->open_time()->sec,
                        'archived'               => '<span style="color:red;">n/a</span>',
                        'archived_flag'          => 0,
                        'local'                  => '<span style="color:red;">never migrated from DAQ or deleted</span>',
                            'local_flag'         => 0,
                        'checksum'               => '',
                        'allowed_stay_sec'       => 0,
                        'allowed_stay'           => '',
                        'restore_flag'           => 0,
                        'restore_requested_time' => '',
                        'restore_requested_uid'  => ''
                    );
                    if(array_key_exists($filename, $migration_delay)) {
                        $start_migration_delay_sec = $migration_delay[$filename];
                        if( !is_null($start_migration_delay_sec)) {
                            $entry['start_migration_delay_sec'] = $start_migration_delay_sec;
                            $entry['local'] = '<span style="color:black;">is migrating from DAQ to OFFLINE...</span>';
                        }
                    }
                    array_push($run_entries, $entry);
                }
            }
        }
        if( count($run_entries)) {

            $run = $experiment->find_run_by_num( $runnum );
            $run_url = is_null( $run ) ?
                $runnum :
                '<a class="link" href="/apps/logbook?action=select_run_by_id&id='.$run->id().'" target="_blank" title="click to see a LogBook record for this run">'.$runnum.'</a>';

            if($first_run_entry) $first_run_entry = false;
            else print ',';
            print ' { "url": '.json_encode($run_url).
                  ' , "runnum": '.$runnum.
                  ' , "files": '.json_encode($run_entries).
                  ' }';
        }
    }
    $total_size_gb_str = json_encode( sprintf( "%.0f", $total_size_gb ));
    $overstay_str = json_encode( $overstay_by_storage_run );
    print <<< HERE
  ],
  "total_size_gb": {$total_size_gb_str},
  "overstay": {$overstay_str}
}
HERE;
    
} catch( LogBookException    $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( LusiTimeException   $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( FileMgrException    $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( DataPortalException $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
