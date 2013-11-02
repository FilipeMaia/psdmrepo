<?php

/** ATTENTION: This is the long running script. So, we have to allow it some
 * to harvest information from various sources.
 */

set_time_limit ( 300 );

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
 * been conducted.
 */
function report_error( $msg ) {
	print '<span style="color:red;">Error: </span>'.$msg;
    exit;
}

define( 'BYTES_IN_TB', 1024.0 * 1024.0 * 1024.0 * 1024.0 );

$since = null;
if( isset($_GET['since'])) {
    $since = LusiTime::parse(trim($_GET['since']));
    if( is_null($since)) report_error('illegar value of parameter: since');
}

try {

    LogBook::instance()->begin();

    $years = array();
    for( $y = 2009; $y <= LusiTime::now()->year()+2; $y += 1 ) {
        $years[$y] = array();
        for( $m = $y == 2009 ? 9 : 1; $m <= 12; $m += 1 )
            $years[$y][$m] = array(
                'NEH' => array(
                    'NEW'          => 0,
                    'MEDIUM-TERM'  => 0,
                    'SHORT-TERM-3' => 0,
                    'SHORT-TERM-6' => 0 ),
                'FEH' => array(
                    'NEW'          => 0,
                    'MEDIUM-TERM'  => 0,
                    'SHORT-TERM-3' => 0,
                    'SHORT-TERM-6' => 0 )
            );
    }
    $instrument2hall = array(
        'AMO' => 'NEH',
        'XPP' => 'NEH',
        'SXR' => 'NEH',
        'XCS' => 'FEH',
        'CXI' => 'FEH',
        'MEC' => 'FEH'
    );
    
    $experiment_quota_left = array();

// TODO: This is the debug line. It will stop the loop after a few iterations in order
//       to see the result. Please, remove this line when finished debugging.
//
//  $counter = 0;

    foreach( LogBook::instance()->experiments() as $experiment ) {

// TODO: This is the debug line. It will stop the loop after a few iterations in order
//       to see the result. Please, remove this line when finished debugging.
//
//     if( $counter++ == 100 ) break;

    	if( $experiment->is_facility()) continue;

        $experiment_quota_left[$experiment->id()] = 10.0;

        // Ignore experiments which haven't taken (yet) any data
    	//
    	$first_run = $experiment->find_first_run();	if( is_null($first_run)) continue;
    	$last_run  = $experiment->find_last_run (); if( is_null($last_run))  continue;

        // Find the start time of each run unless a specific cut-off time 'since' is
        // provided as a parameter to the script.
        //
        $run2begin_time = array();
        foreach( $experiment->runs() as $run )
            $run2begin_time[$run->num()] = is_null($since) ? $run->begin_time() : $since;

   		// Find all runs and files per run. Narrow the search to HPSS archived files
   		// only because files from older experiments may be missing on the local disk.
   		//
   		$range_of_runs = $first_run->num().'-'.$last_run->num();

   		foreach( array( 'xtc', 'hdf5' ) as $type ) {
        	$runs = null;
        	FileMgrIrodsWs::runs( $runs, $experiment->instrument()->name(), $experiment->name(), $type, $range_of_runs );
        	if( is_null($runs)) continue;
        	foreach( $runs as $irods_run ) {
        		$size_tb = 0;
        		foreach( $irods_run->files as $file )
            		if( $file->resource == 'hpss-resc')
            			$size_tb += $file->size / BYTES_IN_TB;

                if( array_key_exists( $irods_run->run, $run2begin_time )) {
                    $begin_time = $run2begin_time[$irods_run->run];
			    	$year_new  = $begin_time->year();
			    	$month_new = $begin_time->month();
		    		$years[$year_new][$month_new][$instrument2hall[$experiment->instrument()->name()]]['NEW'] += $size_tb;
                    foreach( array( 3, 6 ) as $short_term_policy ) {
                        $year  = $year_new;
                        $month = $month_new;
                        for( $m = 1; $m <= $short_term_policy; $m++ ) {
                            $year  += $month == 12 ?   1 : 0;
                            $month += $month == 12 ? -11 : 1;
                            $years[$year][$month][$instrument2hall[$experiment->instrument()->name()]]['SHORT-TERM-'.$short_term_policy] += $size_tb;
                        }
                    }
                    foreach( array( 24 ) as $medium_term_policy ) {
                        if( $experiment_quota_left[$experiment->id()] <= 0.0 ) break;
                        $quota2use = min( array( $experiment_quota_left[$experiment->id()], $size_tb ));
                        $experiment_quota_left[$experiment->id()] -= $quota2use;
                        $year  = $year_new;
                        $month = $month_new;
                        for( $m = 1; $m <= $medium_term_policy; $m++ ) {
                            $year  += $month == 12 ?   1 : 0;
                            $month += $month == 12 ? -11 : 1;
                            $years[$year][$month][$instrument2hall[$experiment->instrument()->name()]]['MEDIUM-TERM'] += $quota2use;
                        }
                    }
            	}
        	}
    	}
    }
?>

<!DOCTYPE html>
<html>
<head>
<title>Analyze Data Accumulation for each month of LCLS </title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 
<link type="text/css" href="../portal/css/Table.css" rel="Stylesheet" />
<style>
#main {
  padding:10px;
  padding-top:0;
}
#table {
  padding-left:10px;
  width:1024px;
}
</style>
</head>
<body>
  <div id="main" >
    <h2>Analyze Data Accumulation for each month of LCLS Data Taking</h2>
    <div id="table">

      <p>The table displays an effect of the New Data Retention Policy on the amount
      of data to be kept on PCDS storage in case if such Policy were enforced from
      the very begging (of LCLS experiments). Two policy options for the SHORT-TERM storage
      have been considered: 3 months and 6 months data retention.
      The numbers represent TBytes.</p>

      <table><tbody>
        <tr>
          <td class="table_hdr" rowspan="3"                            >year</td>
          <td class="table_hdr" rowspan="3"                            >month</td>
          <td class="table_hdr"             colspan="6" align="center" >NEH</td>
          <td class="table_hdr"             colspan="6" align="center" >FEH</td>
          <td class="table_hdr"             colspan="6" align="center" >NEH + FEH</td>
        </tr>
        <tr>
          <td class="table_hdr" rowspan="2"                            >Create</td>
          <td class="table_hdr" rowspan="2"                            >MEDIUM (24)</td>
          <td class="table_hdr"             colspan="2" align="center" >SHORT (3)</td>
          <td class="table_hdr"             colspan="2" align="center" >SHORT (6)</td>
          <td class="table_hdr" rowspan="2"                            >Create</td>
          <td class="table_hdr" rowspan="2"                            >MEDIUM (24)</td>
          <td class="table_hdr"             colspan="2" align="center" >SHORT (3)</td>
          <td class="table_hdr"             colspan="2" align="center" >SHORT (6)</td>
          <td class="table_hdr" rowspan="2"                            >Create</td>
          <td class="table_hdr" rowspan="2"                            >MEDIUM (24)</td>
          <td class="table_hdr"             colspan="2" align="center" >SHORT (3)</td>
          <td class="table_hdr"             colspan="2" align="center" >SHORT (6)</td>
        </tr>
        </tr>
          <td class="table_hdr" >Retain</td>
          <td class="table_hdr" >Total</td>
          <td class="table_hdr" >Retain</td>
          <td class="table_hdr" >Total</td>
          <td class="table_hdr" >Retain</td>
          <td class="table_hdr" >Total</td>
          <td class="table_hdr" >Retain</td>
          <td class="table_hdr" >Total</td>
          <td class="table_hdr" >Retain</td>
          <td class="table_hdr" >Total</td>
          <td class="table_hdr" >Retain</td>
          <td class="table_hdr" >Total</td>
        </tr>
<?php
    $now = LusiTime::now();
    $now_year  = $now->year();
    $now_month = $now->month();
    $month2name = array(
         1 => 'Jan',
         2 => 'Feb',
         3 => 'Mar',
         4 => 'Apr',
         5 => 'May',
         6 => 'Jun',
         7 => 'Jul',
         8 => 'Aug',
         9 => 'Sep',
        10 => 'Oct',
        11 => 'Nov',
        12 => 'Dec'
    );
    $years_keys = array_keys( $years );
    rsort( $years_keys, SORT_NUMERIC );
    foreach( $years_keys as $year ) {
        $month_keys = array_keys( $years[$year] );
        rsort( $month_keys, SORT_NUMERIC );
        foreach( $month_keys as $month ) {
            $month_name = $month2name[$month];

            $NEH_new_size     = intval($years[$year][$month]['NEH']['NEW']);
            $NEH_medium_size  = intval($years[$year][$month]['NEH']['MEDIUM-TERM' ]);
            $NEH_short_size_3 = intval($years[$year][$month]['NEH']['SHORT-TERM-3']);
            $NEH_total_size_3 = intval($NEH_new_size + $NEH_medium_size + $NEH_short_size_3 );
            $NEH_short_size_6 = intval($years[$year][$month]['NEH']['SHORT-TERM-6']);
            $NEH_total_size_6 = intval($NEH_new_size + $NEH_medium_size + $NEH_short_size_6 );

            $FEH_new_size     = intval($years[$year][$month]['FEH']['NEW']);
            $FEH_medium_size  = intval($years[$year][$month]['FEH']['MEDIUM-TERM' ]);
            $FEH_short_size_3 = intval($years[$year][$month]['FEH']['SHORT-TERM-3']);
            $FEH_total_size_3 = intval($FEH_new_size + $FEH_medium_size + $FEH_short_size_3 );
            $FEH_short_size_6 = intval($years[$year][$month]['FEH']['SHORT-TERM-6']);
            $FEH_total_size_6 = intval($FEH_new_size + $FEH_medium_size + $FEH_short_size_6 );

            $BOTH_new_size     = $NEH_new_size     + $FEH_new_size;
            $BOTH_medium_size  = $NEH_medium_size  + $FEH_medium_size;
            $BOTH_short_size_3 = $NEH_short_size_3 + $FEH_short_size_3;
            $BOTH_total_size_3 = $NEH_total_size_3 + $FEH_total_size_3;
            $BOTH_short_size_6 = $NEH_short_size_6 + $FEH_short_size_6;
            $BOTH_total_size_6 = $NEH_total_size_6 + $FEH_total_size_6;

            if( !$NEH_new_size     ) $NEH_new_size     = ' ';
            if( !$NEH_medium_size  ) $NEH_medium_size  = ' ';
            if( !$NEH_short_size_3 ) $NEH_short_size_3 = ' ';
            if( !$NEH_total_size_3 ) $NEH_total_size_3 = ' ';
            if( !$NEH_short_size_6 ) $NEH_short_size_6 = ' ';
            if( !$NEH_total_size_6 ) $NEH_total_size_6 = ' ';

            if( !$FEH_new_size     ) $FEH_new_size     = ' ';
            if( !$FEH_medium_size  ) $FEH_medium_size  = ' ';
            if( !$FEH_short_size_3 ) $FEH_short_size_3 = ' ';
            if( !$FEH_total_size_3 ) $FEH_total_size_3 = ' ';
            if( !$FEH_short_size_6 ) $FEH_short_size_6 = ' ';
            if( !$FEH_total_size_6 ) $FEH_total_size_6 = ' ';

            if( !$BOTH_new_size     ) $BOTH_new_size     = ' ';
            if( !$BOTH_medium_size  ) $BOTH_medium_size  = ' ';
            if( !$BOTH_short_size_3 ) $BOTH_short_size_3 = ' ';
            if( !$BOTH_total_size_3 ) $BOTH_total_size_3 = ' ';
            if( !$BOTH_short_size_6 ) $BOTH_short_size_6 = ' ';
            if( !$BOTH_total_size_6 ) $BOTH_total_size_6 = ' ';

            $extra_styles = '';
            if(( $year == $now_year ) && ( $month == $now_month ))
                $extra_styles = ' background-color:#fff000;';

            print <<<HERE

        <tr style="{$extra_styles}">

          <td class="table_cell table_cell_left " >{$year}</td>
          <td class="table_cell "                 >{$month_name}</td>

          <td class="table_cell "                 style="text-align:right; background-color:#f0f0f0;" >{$NEH_new_size}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$NEH_medium_size}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$NEH_short_size_3}</td>
          <td class="table_cell "                 style="text-align:right; font-weight:bold;"         >{$NEH_total_size_3}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$NEH_short_size_6}</td>
          <td class="table_cell "                 style="text-align:right; font-weight:bold;"         >{$NEH_total_size_6}</td>

          <td class="table_cell "                 style="text-align:right; background-color:#f0f0f0;" >{$FEH_new_size}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$FEH_medium_size}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$FEH_short_size_3}</td>
          <td class="table_cell "                 style="text-align:right; font-weight:bold;"         >{$FEH_total_size_3}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$FEH_short_size_6}</td>
          <td class="table_cell "                 style="text-align:right; font-weight:bold;"         >{$FEH_total_size_6}</td>

          <td class="table_cell "                 style="text-align:right; background-color:#f0f0f0;" >{$BOTH_new_size}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$BOTH_medium_size}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$BOTH_short_size_3}</td>
          <td class="table_cell "                 style="text-align:right; font-weight:bold;"         >{$BOTH_total_size_3}</td>
          <td class="table_cell "                 style="text-align:right;"                           >{$BOTH_short_size_6}</td>
          <td class="table_cell table_cell_right" style="text-align:right; font-weight:bold;"         >{$BOTH_total_size_6}</td>

        </tr>

HERE;
        }
    }
} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }
  catch( FileMgrException  $e ) { report_error( $e->toHtml()); }

?>
      </tbody></table>
    </div>
  </div>
</body>
</head>
