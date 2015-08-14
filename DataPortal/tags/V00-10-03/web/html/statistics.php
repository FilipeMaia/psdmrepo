<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

// Extend the default 30 seconds limit becase the script has to harvest a lot
// of data from various sources.

set_time_limit( 300 );


require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );


use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use FileMgr\FileMgrIrodsDb;
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
  "status": {$status_encoded},
  "message": {$msg_encoded}
}
HERE;
    } else {
        print '<span style="color:red;">Error: </span>'.$msg;
    }
    exit;
}

$instr_name = null;
if( isset( $_GET['instr_name'] )) {
    $instr_name = strtoupper( trim( $_GET['instr_name'] ));
    if( $instr_name == '' ) $instr_name = null;;
}

/*
 * This optional paramater, if present, will tell the script to produce JSON output
 */
$json = isset( $_GET['json'] );

/*
 * This parameter will direct the script to use the slow Web service to access
 * file catalog rather than using iRODS catalog database directly.
 */
$use_ws = isset($_GET['use_ws']);

define( 'BYTES_IN_GB', 1000*1000*1000 );

function pre       ( $str, $style=""             ) { return '<pre style="font-size:125%; '.$style.'">'.$str.'</pre>'; }
function as_int    ( $num, $width=6, $bold=false ) { return $num == 0 ? '&nbsp;' : pre( sprintf( "%{$width}d",   $num ), $bold ? 'font-weight:bold;' : '' ); }
function as_float  ( $num, $width=6, $bold=false ) { return $num == 0 ? '&nbsp;' : pre( sprintf( "%{$width}.1f", $num ), $bold ? 'font-weight:bold;' : '' ); }
function as_text   ( $str          , $bold=false ) { return pre( $str, $bold ? 'font-weight:bold;' : '' ); }
function as_percent( $fraction, $total           ) {
    $percent = 0.0;
    if( $total > 0 ) {
        if( $fraction >= $total ) $percent = 100.0;
        else                      $percent = floor(( $fraction / $total ) * 100.0 );
    }
    return pre( sprintf( "%3.0f%%", $percent ), ( $percent >= 100.0 ? '' : 'color:red;' ));
}
try {

    LogBook::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    // Build an array of experiment specific info using the begin time of
    // the experiment's run as the array key. Consider "real" experiments
    // which have at least one run taken.
    //
    $experiments = array();
    $instruments = array();
    $years       = array();

// TODO: This is the debug line. It will stop the loop after a few iterations in order
//       to see the result. Please, remove this line when finished debugging.
//
// $counter = 0;

    foreach( LogBook::instance()->experiments() as $experiment ) {

        // Optional filtering on the specified instrument if provided

        if( !is_null( $instr_name ) && ($instr_name != $experiment->instrument()->name())) continue;

// TODO: This is the debug line. It will stop the loop after a few iterations in order
//       to see the result. Please, remove this line when finished debugging.
//
// if( $counter++ == 10 ) break;

        if( $experiment->is_facility()) continue;

        // Ignore experiments which haven't taken (yet) any data
        //
        $first_run = $experiment->find_first_run(); if( is_null($first_run)) continue;
        $last_run  = $experiment->find_last_run (); if( is_null($last_run))  continue;

        // Find all runs and files per run. Narrow the search to HPSS archived files
           // only because files from older experiments may be missing on the local disk.
        //
        $range_of_runs  = $first_run->num().'-'.$last_run->num();
        $num_runs       = 0;
        $num_files      = array( 'xtc' => 0, 'hdf5' => 0 );
        $num_files_disk = array( 'xtc' => 0, 'hdf5' => 0 );
        $size_gb        = array( 'xtc' => 0, 'hdf5' => 0 );
        $size_gb_disk   = array( 'xtc' => 0, 'hdf5' => 0 );

        foreach( array( 'xtc', 'hdf5' ) as $type ) {
            $runs = null;
            if ($use_ws) {
                FileMgrIrodsWs::runs( $runs, $experiment->instrument()->name(), $experiment->name(), $type, $range_of_runs );
                if( is_null($runs)) continue;
            } else {
                $runs = FileMgrIrodsDb::instance()->runs ($experiment->instrument()->name(), $experiment->name(), $type, $first_run->num(), $last_run->num());
            }
            foreach( $runs as $irods_run ) {
                $num_runs++;
                $this_run_files   = 0;
                $this_run_files_disk = 0;
                $this_run_size_gb = 0;
                $this_run_size_gb_disk = 0;
                foreach( $irods_run->files as $file ) {
                    if( $file->resource == 'hpss-resc') {
                        $num_files[$type] ++;
                        $size_gb  [$type] += $file->size / BYTES_IN_GB;
                        $this_run_files   ++;
                        $this_run_size_gb += $file->size / BYTES_IN_GB;
                    } else if( $file->resource == 'lustre-resc') {
                        $num_files_disk[$type] ++;
                        $size_gb_disk  [$type] += $file->size / BYTES_IN_GB;
                        $this_run_files_disk   ++;
                        $this_run_size_gb_disk += $file->size / BYTES_IN_GB;
                    }
                }
                $logbook_run = $experiment->find_run_by_num( $irods_run->run );
                if( !is_null($logbook_run)) {
                    $year  = $logbook_run->begin_time()->year();
                    $month = $logbook_run->begin_time()->month();
                    if( !array_key_exists( $year, $years )) $years[$year] = array();
                    if( !array_key_exists( $month, $years[$year] )) $years[$year][$month] = array(
                        'num_runs'            => 0,
                        'num_files_xtc'       => 0,
                        'num_files_xtc_disk'  => 0,
                        'num_files_hdf5'      => 0,
                        'num_files_hdf5_disk' => 0,
                        'size_tb_xtc'         => 0,
                        'size_tb_xtc_disk'    => 0,
                        'size_tb_hdf5'        => 0,
                        'size_tb_hdf5_disk'   => 0
                    );
                    $years[$year][$month]['num_runs'                ] ++;
                    $years[$year][$month]['num_files_'.$type        ] += $this_run_files;
                    $years[$year][$month]['num_files_'.$type.'_disk'] += $this_run_files_disk;
                    $years[$year][$month]['size_tb_'.$type          ] += $this_run_size_gb / 1000;
                    $years[$year][$month]['size_tb_'.$type.'_disk'  ] += $this_run_size_gb_disk / 1000;
                }
            }
        }
        if( !$num_runs ) continue;

        $path = $experiment->regdb_experiment()->find_param_by_name( 'DATA_PATH' );
        $experiments[$last_run->begin_time()->to64()] = array (
            'instr_name'          => $experiment->instrument()->name(),
            'exper_name'          => $experiment->name(),
            'exper_id'            => $experiment->id(),
            'first_run_begin'     => $first_run->begin_time()->toStringDay(),
            'last_run_begin'      => $last_run->begin_time()->toStringDay(),
            'first_run_num'       => $first_run->num(),
            'last_run_num'        => $last_run->num(),
            'num_runs'            => $num_runs,
            'num_files_xtc'       => $num_files     ['xtc' ],
            'num_files_xtc_disk'  => $num_files_disk['xtc' ],
            'num_files_hdf5'      => $num_files     ['hdf5'],
            'num_files_hdf5_disk' => $num_files_disk['hdf5'],
            'size_tb_xtc'         => $size_gb     ['xtc' ] / 1000,
            'size_tb_xtc_disk'    => $size_gb_disk['xtc' ] / 1000,
            'size_tb_hdf5'        => $size_gb     ['hdf5'] / 1000,
            'size_tb_hdf5_disk'   => $size_gb_disk['hdf5'] / 1000,
            'DATA_PATH'           => (is_null( $path ) ? null : $path->value())
        );
        if( array_key_exists( $experiment->instrument()->name(), $instruments )) {
            $instruments[$experiment->instrument()->name()]['num_runs'           ] += $num_runs;
            $instruments[$experiment->instrument()->name()]['num_files_xtc'      ] += $num_files     ['xtc'];
            $instruments[$experiment->instrument()->name()]['num_files_xtc_disk' ] += $num_files_disk['xtc'];
            $instruments[$experiment->instrument()->name()]['num_files_hdf5'     ] += $num_files     ['hdf5'];
            $instruments[$experiment->instrument()->name()]['num_files_hdf5_disk'] += $num_files_disk['hdf5'];
            $instruments[$experiment->instrument()->name()]['size_tb_xtc'        ] += $size_gb       ['xtc' ] / 1000;
            $instruments[$experiment->instrument()->name()]['size_tb_xtc_disk'   ] += $size_gb_disk  ['xtc' ] / 1000;
            $instruments[$experiment->instrument()->name()]['size_tb_hdf5'       ] += $size_gb       ['hdf5'] / 1000;
            $instruments[$experiment->instrument()->name()]['size_tb_hdf5_disk'  ] += $size_gb_disk  ['hdf5'] / 1000;
        } else {
            $instruments[$experiment->instrument()->name()] = array (
                'num_runs'            => $num_runs,
                'num_files_xtc'       => $num_files     ['xtc' ],
                'num_files_xtc_disk'  => $num_files_disk['xtc' ],
                'num_files_hdf5'      => $num_files     ['hdf5'],
                'num_files_hdf5_disk' => $num_files_disk['hdf5'],
                'size_tb_xtc'         => $size_gb       ['xtc' ] / 1000,
                'size_tb_xtc_disk'    => $size_gb_disk  ['xtc' ] / 1000,
                'size_tb_hdf5'        => $size_gb       ['hdf5'] / 1000,
                'size_tb_hdf5_disk'   => $size_gb_disk  ['hdf5'] / 1000
            );
        }
    }
    $experiment_keys = array_keys( $experiments );
    rsort( $experiment_keys, SORT_NUMERIC );

    $total_runs              = 0;
    $total_files_xtc         = 0;
    $total_files_xtc_disk    = 0;
    $total_files_hdf5        = 0;
    $total_files_hdf5_disk   = 0;
    $total_size_tb_xtc       = 0;
    $total_size_tb_xtc_disk  = 0;
    $total_size_tb_hdf5      = 0;
    $total_size_tb_hdf5_disk = 0;
    
    foreach( array_keys( $instruments ) as $instr_name ) {
        $i = $instruments[$instr_name];
        $total_runs              += $i['num_runs'           ];
        $total_files_xtc         += $i['num_files_xtc'      ];
        $total_files_xtc_disk    += $i['num_files_xtc_disk' ];
        $total_files_hdf5        += $i['num_files_hdf5'     ];
        $total_files_hdf5_disk   += $i['num_files_hdf5_disk'];
        $total_size_tb_xtc       += $i['size_tb_xtc'        ];
        $total_size_tb_xtc_disk  += $i['size_tb_xtc_disk'   ];
        $total_size_tb_hdf5      += $i['size_tb_hdf5'       ];
        $total_size_tb_hdf5_disk += $i['size_tb_hdf5_disk'  ];
    }

    $total_files        = $total_files_xtc        + $total_files_hdf5;
    $total_files_disk   = $total_files_xtc_disk   + $total_files_hdf5_disk;
    $total_size_tb      = $total_size_tb_xtc      + $total_size_tb_hdf5;
    $total_size_tb_disk = $total_size_tb_xtc_disk + $total_size_tb_hdf5_disk;
    
    $instrument_keys = array_keys( $instruments );
    sort( $instrument_keys );

    // Fill in gaps in the sequence of years and months
    //
    $now       = LusiTime::now();
    $now_year  = $now->year();
    $now_month = $now->month();

    for( $year = 2009; $year <= 2050; $year++ ) {
        if( !array_key_exists( $year, $years )) $years[$year] = array();
           for( $month = 1; $month <= 12; $month++ ) {
                if(( $year == 2009 ) && ( $month < 10 )) break;  // noting to report before LCLS experiments began taking data
                if( !array_key_exists( $month, $years[$year] )) $years[$year][$month] = array(
                    'num_runs'            => 0,
                    'num_files_xtc'       => 0,
                    'num_files_xtc_disk'  => 0,
                    'num_files_hdf5'      => 0,
                    'num_files_hdf5_disk' => 0,
                    'size_tb_xtc'         => 0,
                    'size_tb_xtc_disk'    => 0,
                    'size_tb_hdf5'        => 0,
                    'size_tb_hdf5_disk'   => 0
                );
                if(( $now_year == $year ) && ( $now_month == $month )) break;    // nothing to be expected next month
        }
        if( $now_year == $year ) break;    // noting to be expected next year
    }
    $year_keys = array_keys( $years );
    rsort( $year_keys, SORT_NUMERIC );

    $data_path = array();
    foreach( $experiments as $e ) {
        $path = $e['DATA_PATH'];
        if( is_null( $path )) continue;
        if( !array_key_exists( $path, $data_path )) {
            $data_path[$path] = array(
                'num_runs'            => 0,
                'num_files_xtc'       => 0,
                'num_files_xtc_disk'  => 0,
                'num_files_hdf5'      => 0,
                'num_files_hdf5_disk' => 0,
                'size_tb_xtc'         => 0,
                'size_tb_xtc_disk'    => 0,
                'size_tb_hdf5'        => 0,
                'size_tb_hdf5_disk'   => 0
            );
        }
        $data_path[$path]['num_runs'           ] += $e['num_runs'           ];
        $data_path[$path]['num_files_xtc'      ] += $e['num_files_xtc'      ];
        $data_path[$path]['num_files_xtc_disk' ] += $e['num_files_xtc_disk' ];
        $data_path[$path]['num_files_hdf5'     ] += $e['num_files_hdf5'     ];
        $data_path[$path]['num_files_hdf5_disk'] += $e['num_files_hdf5_disk'];
        $data_path[$path]['size_tb_xtc'        ] += $e['size_tb_xtc'        ];
        $data_path[$path]['size_tb_xtc_disk'   ] += $e['size_tb_xtc_disk'   ];
        $data_path[$path]['size_tb_hdf5'       ] += $e['size_tb_hdf5'       ];
        $data_path[$path]['size_tb_hdf5_disk'  ] += $e['size_tb_hdf5_disk'  ];
    }
    $data_path_keys = array_keys( $data_path );
    sort( $data_path_keys );


    /* ========================================
     *   REPORT RESULTS IN THE REQUESTED FORM
     * ========================================
     */
    if( $json ) {

        header( 'Content-type: application/json' );
        header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
        header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
        
        print '
{ status:  '.json_encode("success").                        ',
  updated: '.json_encode( LusiTime::now()->toStringShort()).',
  total: {
    runs: '.$total_runs.',
    files:       { xtc : '.                 $total_files_xtc         .', hdf5 : '.                 $total_files_hdf5         .' },
    files_disk:  { xtc : '.                 $total_files_xtc_disk    .', hdf5 : '.                 $total_files_hdf5_disk    .' },
    size_tb:     { xtc : '.sprintf( "%.1f", $total_size_tb_xtc      ).', hdf5 : '.sprintf( "%.1f", $total_size_tb_hdf5      ).' },
    size_tb_disk:{ xtc : '.sprintf( "%.1f", $total_size_tb_xtc_disk ).', hdf5 : '.sprintf( "%.1f", $total_size_tb_hdf5_disk ).' }
  },
  filesystem: [';

        $first = true;
        foreach( $data_path_keys as $path ) {
            $t = $data_path[$path];
            print ( $first ? '' : ',' ).'
    { name: '.json_encode($path).',
      runs: '.$t['num_runs'].',
      files:       { xtc : '.                 $t['num_files_xtc'     ]  .', hdf5 : '.                 $t['num_files_hdf5'     ]  .' },
      files_disk:  { xtc : '.                 $t['num_files_xtc_disk']  .', hdf5 : '.                 $t['num_files_hdf5_disk']  .' },
      size_tb:     { xtc : '.sprintf( "%.1f", $t['size_tb_xtc'       ] ).', hdf5 : '.sprintf( "%.1f", $t['size_tb_hdf5'       ] ).' },
      size_tb_disk:{ xtc : '.sprintf( "%.1f", $t['size_tb_xtc_disk'  ] ).', hdf5 : '.sprintf( "%.1f", $t['size_tb_hdf5_disk'  ] ).' }
    }';
            $first = false;
        }
        print '
  ],
  instruments: [';

        $first = true;
        foreach( $instrument_keys as $instr_name ) {
            $i = $instruments[$instr_name];
            print ( $first ? '' : ',' ).'
    { name: '.json_encode($instr_name).',
      runs: '.$i['num_runs'].',
      files:       { xtc : '.                 $i['num_files_xtc'     ]  .', hdf5 : '.                 $i['num_files_hdf5'     ]  .' },
      files_disk:  { xtc : '.                 $i['num_files_xtc_disk']  .', hdf5 : '.                 $i['num_files_hdf5_disk']  .' },
      size_tb:     { xtc : '.sprintf( "%.1f", $i['size_tb_xtc'       ] ).', hdf5 : '.sprintf( "%.1f", $i['size_tb_hdf5'       ] ).' },
      size_tb_disk:{ xtc : '.sprintf( "%.1f", $i['size_tb_xtc_disk'  ] ).', hdf5 : '.sprintf( "%.1f", $i['size_tb_hdf5_disk'  ] ).' }
    }';
            $first = false;
        }
        print '
  ],
  experiments: [';

        $first = true;
        foreach( $experiment_keys as $k ) {
            $e = $experiments[$k];
            print ( $first ? '' : ',' ).'
    { instr_name      : '.json_encode($e['instr_name']).',
      exper_name      : '.json_encode($e['exper_name']).',
      exper_id        : '.$e['exper_id'  ].',
      first_run_begin : '.json_encode( $e['first_run_begin'] ).',
      last_run_begin  : '.json_encode( $e['last_run_begin' ] ).',
      runs            : '.$e['num_runs'].',
      files           : { xtc : '.                 $e['num_files_xtc'     ].  ', hdf5 : '.                 $e['num_files_hdf5'     ]  .' },
      files_disk      : { xtc : '.                 $e['num_files_xtc_disk'].  ', hdf5 : '.                 $e['num_files_hdf5_disk']  .' },
      size_tb         : { xtc : '.sprintf( "%.1f", $e['size_tb_xtc'       ] ).', hdf5 : '.sprintf( "%.1f", $e['size_tb_hdf5'       ] ).' },
      size_tb_disk    : { xtc : '.sprintf( "%.1f", $e['size_tb_xtc_disk'  ] ).', hdf5 : '.sprintf( "%.1f", $e['size_tb_hdf5_disk'  ] ).' }
    }';
            $first = false;
        }
        print '
  ]
}';

    } else {

        header( 'Content-type: text/html' );
        header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
        header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

        print <<<HERE
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 
<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />
<style type="text/css">
body {
  margin: 0;
  padding: 0;
}
h2 {
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
}
p {
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 13px;
}
table pre {
  margin: 0;
  padding: 0;
}
</style>
</head>
<body>

<div style="padding:20px;">

  <h2>About</h2>
  <div style="padding-left:20px;">
    <p>The information found on this page represents a summary data statistics for
       all LCLS experiments we have had so far. The information is break down into
       five sections:</p>
       <ul>
         <li><a href="#total">Total numbers accross all experiments</a></li>
         <li><a href="#filesystem">Total numbers accross file systems</a></li>
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
      <div style="padding-left:20px;">
        <table><tbody>
          <tr>
            <td class="table_hdr" rowspan=2               >Runs</td>
            <td class="table_hdr" colspan=4 align="center">Files</td>
            <td class="table_hdr" colspan=4 align="center">Size [TB]</td>
          </tr>
          <tr>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >On Disk</td>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >On Disk</td>
            </tr>
HERE;
        print
'
          <tr>
            <td class="table_cell table_bottom table_cell_left  ">'.as_int    ( $total_runs ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $total_files_xtc ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $total_files_hdf5 ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $total_files, 6, true ).'</td>
            <td class="table_cell table_bottom                  ">'.as_percent( $total_files_disk, $total_files ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $total_size_tb_xtc ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $total_size_tb_hdf5 ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $total_size_tb, 6, true ).'</td>
            <td class="table_cell table_bottom table_cell_right ">'.as_percent( $total_size_tb_disk, $total_size_tb ).'</td>
          </tr>
';

        print <<<HERE

        </tr>
      </tbody></table>
    </div>
  </div>
  <div id="filesystem">
    <h2>File System</h2>
      <div style="padding-left:20px;">
        <table><tbody>
          <tr>
            <td class="table_hdr" rowspan=2                >Path</td>
            <td class="table_hdr" rowspan=2                >Runs</td>
            <td class="table_hdr" colspan=4 align="center" >Files</td>
            <td class="table_hdr" colspan=5 align="center" >Size [TB]</td>
          </tr>
          <tr>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >On Disk</td>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >&sum;&nbsp;/&nbsp;max(&sum;)</td>
            <td class="table_hdr" >On Disk</td>
          </tr>

HERE;
        $max_size = 0.0;
        foreach( $data_path_keys as $path ) {
            $t = $data_path[$path];
            $size = $t['size_tb_xtc'] + $t['size_tb_hdf5'];
            if( $size > $max_size ) $max_size = $size;
        }
        $max_width = 120;
        foreach( $data_path_keys as $path ) {
            $t = $data_path[$path];
            $size = $t['size_tb_xtc'] + $t['size_tb_hdf5'];
            $width = $max_size > 0.0 ? sprintf( "%.0f", ceil( $max_width * $size / $max_size )): '1';
            print
'
          <tr>
            <td class="table_cell table_bottom table_cell_left  ">'.as_text   ( $path ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $t['num_runs'      ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $t['num_files_xtc' ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $t['num_files_hdf5'] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $t['num_files_xtc' ] + $t['num_files_hdf5' ], 6, true ).'</td>
            <td class="table_cell table_bottom                  ">'.as_percent( $t['num_files_xtc_disk'] + $t['num_files_hdf5_disk'],
                                                                                $t['num_files_xtc'     ] + $t['num_files_hdf5'     ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $t['size_tb_xtc'   ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $t['size_tb_hdf5'  ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $t['size_tb_xtc'   ] + $t['size_tb_hdf5'], 6, true ).'</td>
            <td class="table_cell table_bottom                  "><div style="float:left; width: '.$width.'px; background-color: #000000;">&nbsp;</div><div style="float:left; width: '.($max_width-$width).'px;">&nbsp;</div><div style="clear:both;"></div></td>
            <td class="table_cell table_bottom table_cell_right ">'.as_percent( $t['size_tb_xtc_disk'] + $t['size_tb_hdf5_disk'],
                                                                                $t['size_tb_xtc'     ] + $t['size_tb_hdf5'     ] ).'</td>
          </tr>
';
        }

        print <<<HERE

        </tr>
      </tbody></table>
    </div>
  </div>
  <div id="instruments">
    <h2>Instruments</h2>
      <div style="padding-left:20px;">
        <table><tbody>
          <tr>
            <td class="table_hdr" rowspan=2                >Instr.</td>
            <td class="table_hdr" rowspan=2                >Runs</td>
            <td class="table_hdr" colspan=4 align="center" >Files</td>
            <td class="table_hdr" colspan=5 align="center" >Size [TB]</td>
          </tr>
          <tr>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >On Disk</td>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >&sum;&nbsp;/&nbsp;max(&sum;)</td>
            <td class="table_hdr" >On Disk</td>
          </tr>

HERE;

        $max_size = 0.0;
        foreach( $instrument_keys as $instr_name ) {
            $i = $instruments[$instr_name];
            $size = $i['size_tb_xtc'] + $i['size_tb_hdf5'];
            if( $size > $max_size ) $max_size = $size;
        }
        $max_width = 120;
        foreach( $instrument_keys as $instr_name ) {
            $i = $instruments[$instr_name];
            $size = $i['size_tb_xtc'] + $i['size_tb_hdf5'];
            $width = $max_size > 0.0 ? sprintf( "%.0f", ceil( $max_width * $size / $max_size )): '1';
            print
'
          <tr>
            <td class="table_cell table_bottom table_cell_left  ">'.as_text   ( $instr_name ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $i['num_runs'      ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $i['num_files_xtc' ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $i['num_files_hdf5'] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $i['num_files_xtc' ] + $i['num_files_hdf5'], 6, true ).'</td>
            <td class="table_cell table_bottom                  ">'.as_percent( $i['num_files_xtc_disk'] + $i['num_files_hdf5_disk'],
                                                                                $i['num_files_xtc'     ] + $i['num_files_hdf5'     ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $i['size_tb_xtc'   ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $i['size_tb_hdf5'  ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $i['size_tb_xtc'   ] + $i['size_tb_hdf5'], 6, true ).'</td>
            <td class="table_cell table_bottom                  "><div style="float:left; width: '.$width.'px; background-color: #000000;">&nbsp;</div><div style="float:left; width: '.($max_width-$width).'px;">&nbsp;</div><div style="clear:both;"></div></td>
            <td class="table_cell table_bottom table_cell_right ">'.as_percent( $i['size_tb_xtc_disk'] + $i['size_tb_hdf5_disk'],
                                                                                $i['size_tb_xtc'     ] + $i['size_tb_hdf5'     ] ).'</td>
          </tr>
';
        }
        print <<<HERE

        </tr>
      </tbody></table>
    </div>
  </div>
  <div id="experiments">
    <h2>Experiments</h2>
      <div style="padding-left:20px;">
        <table><tbody>
          <tr>
            <td class="table_hdr" rowspan=2                >Experiment</td>
            <td class="table_hdr" rowspan=2                >ID</td>
            <td class="table_hdr" rowspan=2                >First Run</td>
            <td class="table_hdr" rowspan=2                >Last Run <span style="font-size:150%; font-weight:bold; color:red;">&nbsp;&uarr;&nbsp;</span></td>
            <td class="table_hdr" rowspan=2                >Runs</td>
            <td class="table_hdr" colspan=4 align="center" >Files</td>
            <td class="table_hdr" colspan=5 align="center" >Size [TB]</td>
            <td class="table_hdr" rowspan=2                >Filesystem</td>
          </tr>
          <tr>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >On Disk</td>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >&sum;&nbsp;/&nbsp;max(&sum;)</td>
            <td class="table_hdr" >On Disk</td>
          </tr>
    
HERE;

        $max_size = 0.0;
        foreach( $experiment_keys as $k ) {
            $e = $experiments[$k];
            $size = $e['size_tb_xtc'] + $e['size_tb_hdf5'];
            if( $size > $max_size ) $max_size = $size;
        }
        $max_width = 120;
        foreach( $experiment_keys as $k ) {
            $e = $experiments[$k];
            $size = $e['size_tb_xtc'] + $e['size_tb_hdf5'];
            $width = $max_size > 0.0 ? sprintf( "%.0f", ceil( $max_width * $size / $max_size )): '1';
            $e_url = '<a href="../portal?exper_id='.$e['exper_id'].'" target="_blank">'.$e['exper_name'].'</a>';
            print
'
          <tr>
            <td class="table_cell table_bottom table_cell_left  ">'.$e_url.'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $e['exper_id'       ], 4 ).'</td>
            <td class="table_cell table_bottom                  ">'.as_text   ( $e['first_run_begin'] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_text   ( $e['last_run_begin' ], 6, true ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $e['num_runs'       ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $e['num_files_xtc'  ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $e['num_files_hdf5' ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $e['num_files_xtc'  ] + $e['num_files_hdf5'], 6, true ).'</td>
            <td class="table_cell table_bottom                  ">'.as_percent( $e['num_files_xtc_disk'] + $e['num_files_hdf5_disk'],
                                                                                $e['num_files_xtc'     ] + $e['num_files_hdf5'     ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $e['size_tb_xtc'    ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $e['size_tb_hdf5'   ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $e['size_tb_xtc'    ] + $e['size_tb_hdf5'], 6, true ).'</td>
            <td class="table_cell table_bottom                  "><div style="float:left; width: '.$width.'px; background-color: #000000;">&nbsp;</div><div style="float:left; width: '.($max_width-$width).'px;">&nbsp;</div><div style="clear:both;"></div></td>
            <td class="table_cell table_bottom                  ">'.as_percent( $e['size_tb_xtc_disk'] + $e['size_tb_hdf5_disk'],
                                                                                $e['size_tb_xtc'     ] + $e['size_tb_hdf5'     ] ).'</td>
            <td class="table_cell table_bottom table_cell_right ">'.as_text   ( $e['DATA_PATH'      ], true ).'</td>
          </tr>
';
        }
        print <<<HERE

        </tr>
      </tbody></table>
    </div>
  </div>
  <div id="months">
    <h2>Each month of data taking</h2>
      <div style="padding-left:20px;">
        <table><tbody>
          <tr>
            <td class="table_hdr" rowspan=2                >Year-Month <span style="font-size:150%; font-weight:bold; color:red;">&nbsp;&uarr;&nbsp;</span></td>
            <td class="table_hdr" rowspan=2                >Runs</td>
            <td class="table_hdr" colspan=4 align="center" >Files</td>
            <td class="table_hdr" colspan=5 align="center" >Size [TB]</td>
          </tr>
          <tr>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >On Disk</td>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >&sum;&nbsp;/&nbsp;max(&sum;)</td>
            <td class="table_hdr" >On Disk</td>
          </tr>

HERE;

        $max_size = 0.0;
        foreach( $year_keys as $year ) {

            $month_keys = array_keys( $years[$year] );
            rsort( $month_keys, SORT_NUMERIC );

            foreach( $month_keys as $month ) {
                $size = $years[$year][$month]['size_tb_xtc'] + $years[$year][$month]['size_tb_hdf5'];
                if( $size > $max_size ) $max_size = $size;
            }
        }
        $max_width = 120;
        foreach( $year_keys as $year ) {

            $month_keys = array_keys( $years[$year] );
            rsort( $month_keys, SORT_NUMERIC );

            foreach( $month_keys as $month ) {
                $size = $years[$year][$month]['size_tb_xtc'] + $years[$year][$month]['size_tb_hdf5'];
                $width = $max_size > 0.0 ? sprintf( "%.0f", ceil( $max_width * $size / $max_size )): '1';
                print
'
          <tr>
            <td class="table_cell table_bottom table_cell_left  ">'.pre       ( sprintf( "%4d - %02d", $year, $month )).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $years[$year][$month]['num_runs'      ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $years[$year][$month]['num_files_xtc' ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $years[$year][$month]['num_files_hdf5'] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $years[$year][$month]['num_files_xtc' ] + $years[$year][$month]['num_files_hdf5'], 6, true ).'</td>
            <td class="table_cell table_bottom                  ">'.as_percent( $years[$year][$month]['num_files_xtc_disk'] + $years[$year][$month]['num_files_hdf5_disk'],
                                                                                $years[$year][$month]['num_files_xtc'     ] + $years[$year][$month]['num_files_hdf5'     ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $years[$year][$month]['size_tb_xtc'   ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $years[$year][$month]['size_tb_hdf5'  ] ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $years[$year][$month]['size_tb_xtc'   ] + $years[$year][$month]['size_tb_hdf5'], 6, true ).'</td>
            <td class="table_cell table_bottom                  "><div style="float:left; width: '.$width.'px; background-color: #000000;">&nbsp;</div><div style="float:left; width: '.($max_width-$width).'px;">&nbsp;</div><div style="clear:both;"></div></td>
            <td class="table_cell table_bottom table_cell_right ">'.as_percent( $years[$year][$month]['size_tb_xtc_disk'] + $years[$year][$month]['size_tb_hdf5_disk'],
                                                                                $years[$year][$month]['size_tb_xtc'     ] + $years[$year][$month]['size_tb_hdf5'     ] ).'</td>
          </tr>
';    
            }
        }

        print <<<HERE

        </tr>
      </tbody></table>
    </div>
  </div>
  <div id="accumulated">
    <h2>Accumulated (total) statistics progression by month</h2>
      <div style="padding-left:20px;">
        <table><tbody>
          <tr>
            <td class="table_hdr" rowspan=2                >Year-Month <span style="font-size:150%; font-weight:bold; color:red;">&nbsp;&darr;&nbsp;</span></td>
            <td class="table_hdr" rowspan=2                >Runs</td>
            <td class="table_hdr" colspan=4 align="center" >Files</td>
            <td class="table_hdr" colspan=5 align="center" >Size [TB]</td>
          </tr>
          <tr>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >On Disk</td>
            <td class="table_hdr" >XTC</td>
            <td class="table_hdr" >HDF5</td>
            <td class="table_hdr" >&sum;</td>
            <td class="table_hdr" >&sum;&nbsp;/&nbsp;max(&sum;)</td>
            <td class="table_hdr" >On Disk</td>
          </tr>

HERE;
        $max_size = 0.0;
        for( $year = 2009; $year <= 2050; $year++ ) {
            if( !array_key_exists( $year, $years )) continue;
            for( $month = 1; $month <= 12; $month++ ) {
                if( array_key_exists( $year, $years ) && array_key_exists( $month, $years[$year] )) {
                    $max_size += $years[$year][$month]['size_tb_xtc'] + $years[$year][$month]['size_tb_hdf5'];
                }
                if(( $now_year == $year ) && ( $now_month == $month )) break;    // nothing to be expected next month
            }
            if( $now_year == $year ) break;    // noting to be expected next year
        }
        
        $accumulated_num_runs            = 0;
        $accumulated_num_files_xtc       = 0;
        $accumulated_num_files_xtc_disk  = 0;
        $accumulated_num_files_hdf5      = 0;
        $accumulated_num_files_hdf5_disk = 0;
        $accumulated_size_tb_xtc         = 0;
        $accumulated_size_tb_xtc_disk    = 0;
        $accumulated_size_tb_hdf5        = 0;
        $accumulated_size_tb_hdf5_disk   = 0;
        
        $max_width = 120;
        for( $year = 2009; $year <= 2050; $year++ ) {
            if( !array_key_exists( $year, $years )) continue;
            for( $month = 1; $month <= 12; $month++ ) {

                if( array_key_exists( $year, $years ) && array_key_exists( $month, $years[$year] )) {
                    $accumulated_num_runs            += $years[$year][$month]['num_runs'           ];
                    $accumulated_num_files_xtc       += $years[$year][$month]['num_files_xtc'      ];
                    $accumulated_num_files_xtc_disk  += $years[$year][$month]['num_files_xtc_disk' ];
                    $accumulated_num_files_hdf5      += $years[$year][$month]['num_files_hdf5'     ];
                    $accumulated_num_files_hdf5_disk += $years[$year][$month]['num_files_hdf5_disk'];
                    $accumulated_size_tb_xtc         += $years[$year][$month]['size_tb_xtc'        ];
                    $accumulated_size_tb_xtc_disk    += $years[$year][$month]['size_tb_xtc_disk'   ];
                    $accumulated_size_tb_hdf5        += $years[$year][$month]['size_tb_hdf5'       ];
                    $accumulated_size_tb_hdf5_disk   += $years[$year][$month]['size_tb_hdf5_disk'  ];
                }
                if( $accumulated_num_runs == 0 ) continue;
                $size = $accumulated_size_tb_xtc + $accumulated_size_tb_hdf5;
                $width = $max_size > 0.0 ? sprintf( "%.0f", ceil( $max_width * $size / $max_size )): '1';
                print
'
          <tr>
            <td class="table_cell table_bottom table_cell_left  ">'.pre       ( sprintf( "%4d - %02d", $year, $month )).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $accumulated_num_runs ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $accumulated_num_files_xtc ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $accumulated_num_files_hdf5 ).'</td>
            <td class="table_cell table_bottom                  ">'.as_int    ( $accumulated_num_files_xtc + $accumulated_num_files_hdf5, 6, true ).'</td>
            <td class="table_cell table_bottom                  ">'.as_percent( $accumulated_num_files_xtc_disk + $accumulated_num_files_hdf5_disk,
                                                                                $accumulated_num_files_xtc      + $accumulated_num_files_hdf5 ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $accumulated_size_tb_xtc ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $accumulated_size_tb_hdf5 ).'</td>
            <td class="table_cell table_bottom                  ">'.as_float  ( $accumulated_size_tb_xtc + $accumulated_size_tb_hdf5, 6, true ).'</td>
            <td class="table_cell table_bottom                  "><div style="float:left; width: '.$width.'px; background-color: #000000;">&nbsp;</div><div style="float:left; width: '.($max_width-$width).'px;">&nbsp;</div><div style="clear:both;"></div></td>
            <td class="table_cell table_bottom table_cell_right ">'.as_percent( $accumulated_size_tb_xtc_disk + $accumulated_size_tb_hdf5_disk,
                                                                                $accumulated_size_tb_xtc      + $accumulated_size_tb_hdf5 ).'</td>
          </tr>
';
                if(( $now_year == $year ) && ( $now_month == $month )) break;    // nothing to be expected next month
            }
            if( $now_year == $year ) break;    // noting to be expected next year
        }
        print <<<HERE

        </tr>
      </tbody></table>
    </div>
  </div>
</div>

</body>
</html>

HERE;

    }

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }
  catch( FileMgrException  $e ) { report_error( $e->toHtml()); }

?>
