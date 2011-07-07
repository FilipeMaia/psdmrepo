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

/* Gather the data statistics accross all experiments which have ever
 * been conducted. Produce a report in a desired format (HTML or JSON).
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

/* This optional p[aramater, if present, will tell the script to produce JSON output
 */
$json = isset( $_GET['json'] );

define( 'BYTES_IN_GB', 1000*1000*1000 );

try {

    $logbook = new LogBook();
    $logbook->begin();

    // Build an array of experiment specific info using the begin time of
    // the experiment's run as the array key. Consider "real" experiments
    // which have at least one run taken.
    //
    $experiments = array();
    $instruments = array();
    $years       = array();

    foreach( $logbook->experiments() as $experiment ) {
    	if( $experiment->is_facility()) continue;

    	// Ignore experiments which haven't taken (yet) any data
    	//
    	$first_run = $experiment->find_first_run();	if( is_null($first_run)) continue;
    	$last_run  = $experiment->find_last_run (); if( is_null($last_run))  continue;

   		// Find all runs and files per run. Narrow the search to HPSS archived files
   		// only because files from older experiments may be missing on the local disk.
   		//
   		$range_of_runs = $first_run->num().'-'.$last_run->num();
   		$num_runs      = 0;
   		$num_files     = array( 'xtc' => 0, 'hdf5' => 0 );
   		$size_gb       = array( 'xtc' => 0, 'hdf5' => 0 );

   		foreach( array( 'xtc', 'hdf5' ) as $type ) {
        	$runs = null;
        	FileMgrIrodsWs::runs( $runs, $experiment->instrument()->name(), $experiment->name(), $type, $range_of_runs );
        	if( is_null($runs)) continue;
        	foreach( $runs as $irods_run ) {
        		$num_runs++;
        		$this_run_files   = 0;
        		$this_run_size_gb = 0;
            	foreach( $irods_run->files as $file ) {
            		if( $file->resource == 'hpss-resc') {
            			$num_files[$type] ++;
            			$size_gb  [$type] += $file->size / BYTES_IN_GB;
            			$this_run_files   ++;
            			$this_run_size_gb += $file->size / BYTES_IN_GB;
            		}
            	}
            	$logbook_run = $experiment->find_run_by_num( $irods_run->run );
            	if( !is_null($logbook_run)) {
			    	$year  = $logbook_run->begin_time()->year();
			    	$month = $logbook_run->begin_time()->month();
			    	if( !array_key_exists( $year, $years )) $years[$year] = array();
		    		if( !array_key_exists( $month, $years[$year] )) $years[$year][$month] = array(
		    			'num_runs'        => 0,
		    			'num_files_xtc'   => 0,
			    		'num_files_hdf5'  => 0,
			    		'size_tb_xtc'     => 0,
			    		'size_tb_hdf5'    => 0
		    		);
					$years[$year][$month]['num_runs']          ++;
		    		$years[$year][$month]['num_files_'.$type ] += $this_run_files;
			    	$years[$year][$month]['size_tb_'.$type   ] += $this_run_size_gb / 1000;
            	}
        	}
    	}
    	if( !$num_runs ) continue;

    	$experiments[$first_run->begin_time()->to64()] = array (
   			'instr_name'      => $experiment->instrument()->name(),
    		'exper_name'      => $experiment->name(),
    		'exper_id'        => $experiment->id(),
    	    'first_run_begin' => $first_run->begin_time()->toStringShort(),
    	    'last_run_begin'  => $last_run->begin_time()->toStringShort(),
    		'first_run_num'   => $first_run->num(),
   			'last_run_num'    => $last_run->num(),
    		'num_runs'        => $num_runs,
    		'num_files_xtc'   => $num_files['xtc' ],
    		'num_files_hdf5'  => $num_files['hdf5'],
    		'size_tb_xtc'     => $size_gb  ['xtc' ] / 1000,
    		'size_tb_hdf5'    => $size_gb  ['hdf5'] / 1000
    	);
    	if( array_key_exists( $experiment->instrument()->name(), $instruments )) {
			$instruments[$experiment->instrument()->name()]['num_runs'      ] += $num_runs;
    		$instruments[$experiment->instrument()->name()]['num_files_xtc' ] += $num_files['xtc'];
    		$instruments[$experiment->instrument()->name()]['num_files_hdf5'] += $num_files['hdf5'];
	    	$instruments[$experiment->instrument()->name()]['size_tb_xtc'   ] += $size_gb  ['xtc' ] / 1000;
    		$instruments[$experiment->instrument()->name()]['size_tb_hdf5'  ] += $size_gb  ['hdf5'] / 1000;
    	} else {
    		$instruments[$experiment->instrument()->name()] = array (
    			'num_runs'       => $num_runs,
    			'num_files_xtc'  => $num_files['xtc' ],
    			'num_files_hdf5' => $num_files['hdf5'],
	    		'size_tb_xtc'    => $size_gb  ['xtc' ] / 1000,
    			'size_tb_hdf5'   => $size_gb  ['hdf5'] / 1000
    		);
    	}
    }
    $experiment_keys = array_keys( $experiments );
    sort( $experiment_keys, SORT_NUMERIC );

    $total_runs         = 0;
    $total_files_xtc    = 0;
    $total_files_hdf5   = 0;
    $total_size_tb_xtc  = 0;
    $total_size_tb_hdf5 = 0;

    foreach( array_keys( $instruments ) as $instr_name ) {
    	$i = $instruments[$instr_name];
    	$total_runs         += $i['num_runs'];
		$total_files_xtc    += $i['num_files_xtc' ];
		$total_files_hdf5   += $i['num_files_hdf5'];
		$total_size_tb_xtc  += $i['size_tb_xtc' ];
		$total_size_tb_hdf5 += $i['size_tb_hdf5'];
	}

	$total_files   = $total_files_xtc + $total_files_hdf5;
	$total_size_tb = $total_size_tb_xtc + $total_size_tb_hdf5;

    $instrument_keys = array_keys( $instruments );
   	sort( $instrument_keys );

   	$year_keys = array_keys( $years );
    sort( $year_keys, SORT_NUMERIC );
   	
   	if( $json ) {

    	$success_encoded = json_encode("success");
    	$updated_str     = json_encode( LusiTime::now()->toStringShort());

    	header( 'Content-type: application/json' );
    	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    	$total_size_tb_xtc_fmt  = sprintf( "%.1f", $total_size_tb_xtc );
    	$total_size_tb_hdf5_fmt = sprintf( "%.1f", $total_size_tb_hdf5 );

   		print <<< HERE
{ "Status": {$success_encoded},
  "updated": {$updated_str},
  "total": {
    "runs": {$total_runs},
    "files": { "xtc" : {$total_files_xtc}, "hdf5" : {$total_files_hdf5} },
    "size_tb: { "xtc" : {$total_size_tb_xtc_fmt}, "hdf5" : {$total_size_tb_hdf5_fmt} }
  },
  "instruments": [

HERE;
		$first = true;
    	foreach( $instrument_keys as $instr_name ) {
       		$i = $instruments[$instr_name];

       		$num_runs       = $i['num_runs'];
       		$num_files_xtc  = $i['num_files_xtc'];
       		$num_files_hdf5 = $i['num_files_hdf5'];
       		$size_tb_xtc    = sprintf( "%.1f", $i['size_tb_xtc'] );
       		$size_tb_hdf5   = sprintf( "%.1f", $i['size_tb_hdf5'] );

       		if($first) $first = false;
			else print ',';

			print <<<HERE
    { "name": {$instr_name},
      "runs": {$num_runs},
      "files": { "xtc" : {$num_files_xtc}, "hdf5" : {$num_files_hdf5} },
      "size_tb: { "xtc" : {$size_tb_xtc}, "hdf5" : {$size_tb_hdf5} }
    }
HERE;
    	}
		print <<<HERE

  ],
  "experiments": [

HERE;
		$first = true;
		foreach( $experiment_keys as $k ) {
			$e = $experiments[$k];

			$instr_name      = $e['instr_name'];
			$exper_name      = $e['exper_name'];
			$exper_id        = $e['exper_id'];
			$first_run_begin = json_encode( $e['first_run_begin'] );
			$last_run_begin  = json_encode( $e['last_run_begin'] );
			$num_runs        = $e['num_runs'];
       		$num_files_xtc   = $e['num_files_xtc'];
       		$num_files_hdf5  = $e['num_files_hdf5'];
       		$size_tb_xtc     = sprintf( "%.1f", $e['size_tb_xtc'] );
       		$size_tb_hdf5    = sprintf( "%.1f", $e['size_tb_hdf5'] );

       		if($first) $first = false;
			else print ',';

			print <<<HERE

    { "instr_name": {$instr_name},
      "exper_name": {$exper_name},
      "exper_id": {$exper_id},
      "first_run_begin" : {$first_run_begin},
      "last_run_begin" : {$last_run_begin},
      "runs": {$num_runs},
      "files": { "xtc" : {$num_files_xtc}, "hdf5" : {$num_files_hdf5} },
      "size_tb: { "xtc" : {$size_tb_xtc}, "hdf5" : {$size_tb_hdf5} }
    }
HERE;
		}
		print <<<HERE

  ]
}
HERE;

    } else {

    	header( 'Content-type: text/html' );
    	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    	print <<<HERE
<div style="padding-left:20px; padding-right:20px;">

  <h2>About</h2>
  <div style="padding-left:20px;">
    <p>The information found on this page represents a summary data statistics for
       all LCLS experiments we have had so far. The information is break down into
       five sections:</p>
       <ul>
         <li><a href="#total">Total numbers accross all experiments</a></li>
         <li><a href="#instruments">For each instrument</a></li>
         <li><a href="#experiments">For each experiment</a></li>
         <li><a href="#months">For each month of data taking</a></li>
         <li><a href="#accumulated">Accumulated (total) statistics progression by month</a></li>
       </ul>
    <p>In case if someone may want to incorporate this information into
       a dynamic HTML page the report can be also be obtained in JSON format
       from <a href="?json" target="_blank">here</a>.</p>
  </div>
HERE;
		print <<<HERE

  <div id="total">
    <h2>Total</h2>

HERE;
    	print
'<pre>'.
'    Runs            : '.$total_runs.'<br>'.
'    Files           : '.$total_files.'<br>'.
'        XTC         : '.$total_files_xtc.'<br>'.
'        HDF5        : '.$total_files_hdf5.'<br>'.
'    Total size [TB] : '.sprintf( "%.1f", $total_size_tb ).'<br>'.
'        XTC         : '.sprintf( "%.1f", $total_size_tb_xtc ).'<br>'.
'        HDF5        : '.sprintf( "%.1f", $total_size_tb_hdf5 ).
'</pre>';
    	
		print <<<HERE

  </div>
  <div id="instruments">
    <h2>Instruments</h2>

HERE;
    	
    	foreach( $instrument_keys as $instr_name ) {
       		$i = $instruments[$instr_name];
       		print
'<pre>'.
'  <b>'.$instr_name.'</b><br>'.
'<br>'.
'    Runs            : '.$i['num_runs'].'<br>'.
'    Files           : '.($i['num_files_xtc'] + $i['num_files_hdf5']).'<br>'.
'        XTC         : '.$i['num_files_xtc'].'<br>'.
'        HDF5        : '.$i['num_files_hdf5'].'<br>'.
'    Total size [TB] : '.sprintf( "%.1f", $i['size_tb_xtc'] + $i['size_tb_hdf5'] ).'<br>'.
'        XTC         : '.sprintf( "%.1f", $i['size_tb_xtc'] ).'<br>'.
'        HDF5        : '.sprintf( "%.1f", $i['size_tb_hdf5'] ).
'</pre>';

		}
		print <<<HERE

  </div>
  <div id="experiments">
    <h2>Experiments</h2>

HERE;

    	foreach( $experiment_keys as $k ) {
			$e = $experiments[$k];
			print
'<pre>'.
'  <b>'.$e['instr_name'].'</b> / <b>'.$e['exper_name'].'</b>'.' [ ID:'.$e['exper_id'].' ] [ '.$e['first_run_begin'].' - '.$e['last_run_begin'].' ]<br>'.
'<br>'.
'    Runs            : '.$e['num_runs'].'<br>'.
'    Files           : '.($e['num_files_xtc'] + $e['num_files_hdf5']).'<br>'.
'        XTC         : '.$e['num_files_xtc'].'<br>'.
'        HDF5        : '.$e['num_files_hdf5'].'<br>'.
'    Total size [TB] : '.sprintf( "%.1f", $e['size_tb_xtc'] + $e['size_tb_hdf5'] ).'<br>'.
'        XTC         : '.sprintf( "%.1f", $e['size_tb_xtc'] ).'<br>'.
'        HDF5        : '.sprintf( "%.1f", $e['size_tb_hdf5'] ).
'</pre>';

		}
		print <<<HERE

  </div>
  <div id="months">
    <h2>Each month of data taking</h2>

HERE;

    	foreach( $year_keys as $year ) {

    		$month_keys = array_keys( $years[$year] );
			sort( $month_keys, SORT_NUMERIC );

			foreach( $month_keys as $month ) {

				print
'<pre>'.
'  <b>'.$year.'</b>-<b>'.sprintf( "%02d", $month ).'</b><br>'.
'<br>'.
'    Runs            : '.$years[$year][$month]['num_runs'].'<br>'.
'    Files           : '.($years[$year][$month]['num_files_xtc'] + $years[$year][$month]['num_files_hdf5']).'<br>'.
'        XTC         : '.$years[$year][$month]['num_files_xtc'].'<br>'.
'        HDF5        : '.$years[$year][$month]['num_files_hdf5'].'<br>'.
'    Total size [TB] : '.sprintf( "%.1f", $years[$year][$month]['size_tb_xtc'] + $years[$year][$month]['size_tb_hdf5'] ).'<br>'.
'        XTC         : '.sprintf( "%.1f", $years[$year][$month]['size_tb_xtc'] ).'<br>'.
'        HDF5        : '.sprintf( "%.1f", $years[$year][$month]['size_tb_hdf5'] ).
'</pre>';
			}
		}

		print <<<HERE
  </div>
  <div id="accumulated">
    <h2>Accumulated (total) statistics progression by month</h2>

HERE;

		$now       = LusiTime::now();
		$now_year  = $now->year();
		$now_month = $now->month();

		$accumulated_num_runs       = 0;
		$accumulated_num_files_xtc  = 0;
		$accumulated_num_files_hdf5 = 0;
		$accumulated_size_tb_xtc    = 0;
		$accumulated_size_tb_hdf5   = 0;

		for( $year = 2009; $year <= 2050; $year++ ) {
			if( !array_key_exists( $year, $years )) continue;
    		for( $month = 1; $month <= 12; $month++ ) {

    			if( array_key_exists( $year, $years ) && array_key_exists( $month, $years[$year] )) {
    				$accumulated_num_runs       += $years[$year][$month]['num_runs'      ];
					$accumulated_num_files_xtc  += $years[$year][$month]['num_files_xtc' ];
					$accumulated_num_files_hdf5 += $years[$year][$month]['num_files_hdf5'];
					$accumulated_size_tb_xtc    += $years[$year][$month]['size_tb_xtc'   ];
					$accumulated_size_tb_hdf5   += $years[$year][$month]['size_tb_hdf5'  ];
    			}
    			if( $accumulated_num_runs == 0 ) continue;
				print
'<pre>'.
'  <b>'.$year.'</b>-<b>'.sprintf( "%02d", $month ).'</b><br>'.
'<br>'.
'    Runs            : '.$accumulated_num_runs.'<br>'.
'    Files           : '.($accumulated_num_files_xtc + $accumulated_num_files_hdf5).'<br>'.
'        XTC         : '.$years[$year][$month]['num_files_xtc'].'<br>'.
'        HDF5        : '.$years[$year][$month]['num_files_hdf5'].'<br>'.
'    Total size [TB] : '.sprintf( "%.1f", $accumulated_size_tb_xtc + $accumulated_size_tb_hdf5 ).'<br>'.
'        XTC         : '.sprintf( "%.1f", $accumulated_size_tb_xtc ).'<br>'.
'        HDF5        : '.sprintf( "%.1f", $accumulated_size_tb_hdf5 ).
'</pre>';

				if(( $now_year == $year ) && ( $now_month == $month )) break;	// nothing to be expected next month
    		}
			if( $now_year == $year ) break;	// noting to be expected next year
		}
		print <<<HERE

  </div>
</div>

HERE;

    }

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }
  catch( FileMgrException  $e ) { report_error( $e->toHtml()); }

?>
