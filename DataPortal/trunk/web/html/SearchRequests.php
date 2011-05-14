<?php

require_once( 'dataportal/dataportal.inc.php' );
require_once( 'regdb/regdb.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use DataPortal\Translator;
use DataPortal\DataPortal;
use DataPortal\DataPortalException;

use RegDB\RegDBException;

use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

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
 * runs       - a range of runs
 * status      - a desired status for translation requests
 * show_files - a flag indicating if files should be shown as well for each run
 */
if( isset( $_GET['runs'] )) {
    $range_of_runs = trim( $_GET['runs'] );
    if( $range_of_runs == '' ) report_error( 'run range parameter shall not have an empty value' );
}

$known_simple_statuses = array( 'FINISHED' => 1, 'FAILED' => 1, 'TRANSLATING' => 1, 'QUEUED' => 1, 'NOT-TRANSLATED' => 1 );
$status = null;
if( isset( $_GET['status'] )) {
	$status = strtoupper( trim( $_GET['status'] ));
    if( $status == '' ) report_error( 'translation status parameter shall not have an empty value' );
    if( !array_key_exists( $status, $known_simple_statuses )) report_error( 'unsupported value of the translatioon status parameter' );
}

$show_files = isset( $_GET['show_files'] );

/* This flag, if present, will tell the script to produce JSON output
 */
$json = isset( $_GET['json'] );
if( $json ) $import_format = false;

/* Utility functions
 */
function decorated_request_status( $status ) {
	$color = 'black';
    switch( $status ) {
    	case 'QUEUED':		$color = '#c0c0c0'; break;
    	case 'TRANSLATING':	$color = 'green'; break;
    	case 'FAILED':		$color = 'red'; break;
    }
    return '<span style="color:'.$color.'";">'.$status.'</span>';
}

function pre( $str, $width=null, $color=null ) {
	$color_style = is_null( $color ) ? '' : ' style="color:'.$color.';"';
	if( is_null( $width )) return "<pre {$color_style}>".$str.'</pre>';
	return "<pre {$color_style}>".sprintf( "%{$width}s", $str ).'</pre>';
}

/* -----------------------------
 * Begin the main algorithm here
 * -----------------------------
 */
try {

	$requests = Translator::get_requests( $exper_id, $range_of_runs, $status, $show_files );

	if( $json ) {

    	header( 'Content-type: application/json' );
    	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

   		print
   		'{ "Status": '.json_encode("success").
   		', "updated": '.json_encode( LusiTime::now()->toStringShort()).
   		', "requests": '.json_encode( $requests ).
   		'}';

    } else {

    	header( 'Content-type: text/html' );
    	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    	
    	echo DataPortal::table_begin_html(
			array(
				array( 'name' => 'Run',        'width' =>  30 ),
				array( 'name' => 'End of Run', 'width' =>  90 ),
				array( 'name' => 'File',       'width' => 200 ),
				array( 'name' => 'Size',       'width' =>  90 ),
				array( 'name' => 'Status',     'width' =>  30 ),
				array( 'name' => 'Changed',    'width' =>  90 ),
				array( 'name' => 'Log',        'width' =>  30 ),
				array( 'name' => 'Priority',   'width' =>  40 ),
				array( 'name' => 'Actions',    'width' =>  40 ),
				array( 'name' => 'Comments',   'width' => 200 )
			)
		);
    	foreach( $requests as $request ) {

    		$state = $request['state'];

    		/* Now show the very first summary line for that run. This may be the only
             * line in case if a user doesn't want to see the files, or if no XTC files
             * are present.
             */
    		$run_url = '<a class="link" href="/apps/logbook?action=select_run_by_id&id='.$state['run_id'].'" target="_blank" title="click to see a LogBook record for this run">'.$state['run_number'].'</a>';    		
    		$id = $state['id'];
    		$row = array(
				$run_url,
   				$state['end_of_run'],
   				'',
   				'',
   				decorated_request_status( $state['status'] ),
   				$state['changed'],
   				$state['log_available'] ? '<a class="link" href="translate/'.$id.'/'.$id.'.log" target="_blank" title="click to see the log file for the last translation attempt">log</a>' : '',
   				$state['status'] == 'QUEUED' ? '<span id="priority_'.$id.'">'.$state['priority'].'</span>' : '',
   				$state['actions'],
   				$state['comments']
  			);
           	$numfiles = count( $request['xtc'] ) + count( $request['hdf5'] );
  			echo DataPortal::table_row_html(
    			$row,
   				!( $show_files && ( $numfiles != 0 ))
    		);
           	$filenum  = 0;
            foreach( array( 'xtc', 'hdf5') as $type ) {
           		foreach( $request[$type] as $f ) {
		   			$filenum++;
               		$end_of_group = ( $filenum == $numfiles );
					$row = array( '', '', $f['name'], $f['size'], '', '', '', '', '', '' );
					echo DataPortal::table_row_html( $row, $end_of_group );
           		}
            }
    	}
    	echo DataPortal::table_end_html();
    }
    
} catch( RegDBException      $e ) { report_error( $e->toHtml()); }
  catch( LogBookException    $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( FileMgrException    $e ) { report_error( $e->toHtml()); }
  catch( DataPortalException $e ) { report_error( $e->toHtml()); }
  
?>