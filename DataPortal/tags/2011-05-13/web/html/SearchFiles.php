<?php

require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use DataPortal\DataPortal;

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

/* Report the error according to the desired output format. If this is just
 * a plain HTML then produce an HTML document. For JSON package the error
 * message into a JSON object and return the one back to a caller. The script's
 * execution will end at this point.
 */
function report_error( $msg ) {
	if( isset( $_GET['json'] )) {
	    $status_encoded = json_encode( "error" );
    	$msg_encoded = json_encode( '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg );
    	print <<< HERE
{
  "Status": {$status_encoded},
  "Message": {$msg_encoded}
}
HERE;
	} else {
		print '<span style="color:red;">Error: </span>'.$msg;
	}
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

/* This flag, if present, would tell the script to return a plain list
 * of selected file paths on a local disk for those files for which
 * this would apply.
 */
$import_format = isset( $_GET['import_format'] );

/* This flag, if present, will tell the script to produce JSON output
 *
 * NOTE: This flag will also invalidate the 'import_format' (if any).
 */
$json = isset( $_GET['json'] );
if( $json ) $import_format = false;


function pre( $str, $width=null ) {
    if( is_null( $width )) return '<pre>'.$str.'</pre>';
    return '<pre>'.sprintf( "%{$width}s", $str ).'</pre>';
}

function add_files( &$files, $infiles, $type, $file2run, $checksum, $archived, $local ) {
    
    $result = array();

    foreach( $infiles as $file ) {

        if( $file->type == 'collection' ) continue;

        // Skip files for which we don't know run numbers
        //
        if( !array_key_exists( $file->name, $file2run )) continue;
        
        if( !array_key_exists( $file->name, $result )) {
            $result[$file->name] =
                array (
                    'name'     => $file->name,
                    'run'      => $file2run[$file->name],
                	'type'     => $type,
                    'size'     => $file->size,
                    'created'  => $file->ctime,
                    'archived'      => '<span style="color:red;">No</span>',
                    'archived_flag' => false,
                    'local'         => '<span style="color:red;">No</span>',
                    'local_flag'    => false,
                    'checksum' => '' );    
        }
        if( $file->resource == 'hpss-resc' ) {
            $result[$file->name]['archived'] = 'Yes';
            $result[$file->name]['archived_flag'] = true;
        }
        if( $file->resource == 'lustre-resc' ) {
            $result[$file->name]['local'] = 'Yes';
            $result[$file->name]['local_flag'] = true;
            $result[$file->name]['local_path'] = $file->path;
            $result[$file->name]['created'] = $file->ctime;
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

/*
 * Analyze and process the request
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $exper_id ) or report_error("No such experiment");
    $instrument = $experiment->instrument();

    /* If no specific run range is provided find out the one by probing all
     * known file types.
     */
    if( is_null( $range_of_runs )) {
        $range = FileMgrIrodsWs::max_run_range( $instrument->name(), $experiment->name(), array('xtc','hdf5'));
        $range_of_runs = $range['min'].'-'.$range['max'];
    }

    /* Build a mapping from file names to the corresponding run numbers.
     * This information will be shown in the GUI.
     */
    $file2run = array();
    foreach( $types as $type ) {
        $runs = null;
        FileMgrIrodsWs::runs( $runs, $instrument->name(), $experiment->name(), $type, $range_of_runs );
        foreach( $runs as $run ) {
            $files = $run->files;
            foreach( $files as $file ) {
                $file2run[$file->name] = $run->run;
            }
        }
    }

    /* Build a list fop all known files.
     */
    $files = array();
    foreach( $types as $type ) {
        $this_type_files = null;
        FileMgrIrodsWs::files( $this_type_files, '/psdm-zone/psdm/'.$instrument->name().'/'.$experiment->name().'/'.$type );
        add_files( $files, $this_type_files, $type, $file2run, $checksum, $archived, $local );
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

    if( $json ) {

    	$success_encoded = json_encode("success");
    	$updated_str     = json_encode( LusiTime::now()->toStringShort());

    	header( 'Content-type: application/json' );
    	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

   		print <<< HERE
{ "Status": {$success_encoded},
  "updated": {$updated_str},
  "runs": [
HERE;
		$total_size_gb = 0;
		$first_run = true;
    	foreach( array_keys( $files_by_runs ) as $runnum ) {
    
            $run = $experiment->find_run_by_num( $runnum );
   	        $run_url = is_null( $run ) ?
       	        $runnum :
               	'<a class="link" href="/apps/logbook?action=select_run_by_id&id='.$run->id().'" target="_blank" title="click to see a LogBook record for this run">'.$runnum.'</a>';

			if($first_run) $first_run = false;
			else print ',';
			print ' { "url": '.json_encode($run_url).', "files": [ ';

			$first_file = true;
            foreach( $files_by_runs[$runnum] as $file ) {
       
            	$total_size_gb += $file['size'] / (1024.0 * 1024.0 * 1024.0);
				if($first_file) $first_file = false;
				else print ',';
				print json_encode( array(
					'runnum'          => $runnum,
   					'name'            => $file['name'],
   					'type'            => strtoupper( $file['type'] ),
					'size_bytes'      => $file['size'],
    				'size'            => number_format( $file['size'] ),
   					'created'         => date( "Y-m-d H:i:s", $file['created'] ),
					'created_seconds' => $file['created'],
   					'archived'        => $file['archived'],
   					'local'           =>
						$file['local_flag'] ?
   	            		'<a class="link" href="javascript:display_path('."'".$file['local_path']."'".')">path</a>' :
               	   		$file['local'],
               	   	'checksum' => $file['checksum']
  				));
   	        }
   	        print ' ] }';
        }
        $total_size_gb_str = json_encode( sprintf( "%.0f", $total_size_gb ));
    	print <<< HERE
  ],
  "total_size_gb": {$total_size_gb_str}
}
HERE;

    } else {

    	header( 'Content-type: text/html' );
    	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
    	if( $import_format ) {

    		echo '<pre>';

	    	foreach( array_keys( $files_by_runs ) as $runnum )
    	        foreach( $files_by_runs[$runnum] as $file )
        	        if( $file['local_flag'] )
            	        print '  '.$file['local_path']."\n";

	        echo '</pre>';

    	} else {
    
	        echo DataPortal::table_begin_html(
				array(
					array( 'name' => 'Run',       'width' =>  30 ),
					array( 'name' => 'File Name', 'width' => 400 ),
					array( 'name' => 'Type',      'width' =>  40 ),
					array( 'name' => 'Size',      'width' =>  90 ),
					array( 'name' => 'Created',   'width' =>  90 ),
					array( 'name' => 'Archived',  'width' =>  30 ),
					array( 'name' => 'Disk',      'width' =>  40 )
				)
			);
	        foreach( array_keys( $files_by_runs ) as $runnum ) {
    
	            $run = $experiment->find_run_by_num( $runnum );
    	        if( is_null( $run )) {
        	        $run_url = pre( $runnum );
            	} else {
                	$run_url = pre( '<a class="link" href="/apps/logbook?action=select_run_by_id&id='.$run->id().'" target="_blank" title="click to see a LogBook record for this run">'.$runnum.'</a>' );
	            }
	            $filenum = 0;
    	        $files = $files_by_runs[$runnum];
	            foreach( $files as $file ) {

	                $name = pre( $file['name'] );
    	            if( $file['local_flag'] ) {
        	            $local = pre( '<a class="link" href="javascript:display_path('."'".$file['local_path']."'".')">path</a>' );
            	    } else {
                	    $local = pre( $file['local'] );
	                }
    	            $type     = pre( strtoupper( $file['type'] ));
	                $size     = pre( number_format( $file['size'] ), 17 );    // less than 10 TB
	                $created  = pre( date( "Y-m-d H:i:s", $file['created'] ));
                	$archived = pre( $file['archived'] );
        
	                $filenum++;
    	            $end_of_group = $filenum == count( $files );
    				echo DataPortal::table_row_html(
    					array(
    						$filenum == 1 ? $run_url : '',
    						$name,
    						$type,
	    					$size,
    						$created,
    						$archived,
    						$local
   						),
   						$end_of_group
	    			);
    	        }
	            //DataPortal::table_separator();
	        }
			echo DataPortal::table_end_html();
    	}
    }
    
} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }
  catch( FileMgrException  $e ) { report_error( $e->toHtml()); }

?>
