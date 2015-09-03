
<!DOCTYPE html">
<html>
<head>
<link type="text/css" href="../portal/css/Table.css" rel="Stylesheet" />
</head>
<body>

<div style="padding:20px;">
  <h2>This script will populate database cache with output of the iRods Web Service operation:</h2>
  <h3>GET /runs/{instrument}/{experiment}/{type}/{runs}</h3>

<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use DataPortal\Config;
use DataPortal\DataPortalException;

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

/* This script will populate database cache with output of the iRods Web Service
 * operation:
 *
 *   GET /runs/{instrument}/{experiment}/{type}/{runs}
 */
function report_error( $msg ) {
	print '<div style="color:red;">Error: </div>'.$msg;
    exit;
}

try {

    LogBook::instance()->begin();

    $config = Config::instance();
    $config->begin();

    $cache = $config->create_irods_cache('statistics');
    print <<<HERE
  
  <table><tbody>

    <tr>
      <td class="table_hdr" >id</td>
      <td class="table_hdr" >application</td>
      <td class="table_hdr" >create_time</td>
      <td class="table_hdr" >create_time [nanoseconds]</td>
    </tr>

    <tr>
      <td class="table_cell table_cell_left  " align="right" >{$cache->id}</td>
      <td class="table_cell                  "               >{$cache->application}</td>
      <td class="table_cell                  "               >{$cache->create_time}</td>
      <td class="table_cell table_cell_right " align="right" >{$cache->create_time->to64()}</td>
    </tr>

  </tbody></table>

HERE;

    $experiments = array();
    $instruments = array();

    print <<<HERE

  <table><tbody>

    <tr>
      <td class="table_hdr" >Instrument</td>
      <td class="table_hdr" >Experiment</td>
      <td class="table_hdr" >#runs</td>
      <td class="table_hdr" >#files (XTC)</td>
      <td class="table_hdr" >#files (HDF5)</td>
      <td class="table_hdr" >#files (total)</td>
      <td class="table_hdr" >seconds</td>
    </tr>

HERE;

    $total_num_experiments = 0;
    $total_num_runs        = 0;
    $total_num_files       = array('xtc' => 0, 'hdf5' => 0);
    $total_seconds         = 0;

    $instrument_names = array();
    foreach (LogBook::instance()->instruments() as $instrument) array_push($instrument_names, $instrument->name());
    sort($instrument_names);
    
    foreach ($instrument_names as $instr_name) {
        foreach (LogBook::instance()->experiments_for_instrument($instr_name) as $experiment) {

            $exper_name = $experiment->name();

            if( $experiment->is_facility()) continue;

            $total_num_experiments += 1;

            // Ignore experiments which haven't taken (yet) any data
            //
            $first_run = $experiment->find_first_run();	if( is_null($first_run)) continue;
            $last_run  = $experiment->find_last_run (); if( is_null($last_run))  continue;

            $range_of_runs = $first_run->num().'-'.$last_run->num();
            $num_runs      = $last_run->num() - $first_run->num() + 1;

            $total_num_runs += $num_runs;

            $num_files = array('xtc' => 0, 'hdf5' => 0);

            $start_time = LusiTime::now()->to_float();
            foreach (array( 'xtc', 'hdf5' ) as $type) {
                $runs = null;
                FileMgrIrodsWs::runs( $runs, $instr_name, $exper_name, $type, $range_of_runs );
                if (is_null($runs)) continue;
                foreach ($runs as $r) {
                    $num_files[$type] += count($r->files);
                }
                $config->add_files_irods_cache($cache->id, $experiment->id(), $type, $runs);
            }
            $stop_time = LusiTime::now()->to_float();

            $seconds = $stop_time - $start_time;
            $total_seconds += $seconds;

            $seconds_str = sprintf("%.0f", $seconds);

            $num_files_xtc   = $num_files['xtc'];
            $num_files_hdf5  = $num_files['hdf5'];
            $num_files_total = $num_files_xtc + $num_files_hdf5;

            $total_num_files['xtc']  += $num_files_xtc;
            $total_num_files['hdf5'] += $num_files_hdf5;

            print <<<HERE
    <tr>
      <td class="table_cell table_cell_left  "                                         >{$instr_name}</td>
      <td class="table_cell                  "                                         >{$exper_name}</td>
      <td class="table_cell                  " align="right" style="font-weight:bold;" >{$num_runs}</td>
      <td class="table_cell                  " align="right"                           >{$num_files_xtc}</td>
      <td class="table_cell                  " align="right"                           >{$num_files_hdf5}</td>
      <td class="table_cell                  " align="right" style="font-weight:bold;" >{$num_files_total}</td>
      <td class="table_cell table_cell_right " align="right" style="font-weight:bold;" >{$seconds_str}</td>
    </tr>

HERE;
            flush();
            ob_flush();
        }
    }
    $total_num_files_xtc   = $total_num_files['xtc'];
    $total_num_files_hdf5  = $total_num_files['hdf5'];
    $total_num_files_total = $total_num_files_xtc + $total_num_files_hdf5;
    $total_seconds_str     = sprintf("%.0f", $total_seconds);

    print <<<HERE

    <tr style="background-color:#f0f0f0;" >
      <td class="table_cell table_cell_left  " >&nbsp;</td>
      <td class="table_cell                  " align="right" >{$total_num_experiments}</td>

      <td class="table_cell                  " align="right" >{$total_num_runs}</td>
      <td class="table_cell                  " align="right" >{$total_num_files_xtc}</td>
      <td class="table_cell                  " align="right" >{$total_num_files_hdf5}</td>
      <td class="table_cell                  " align="right" >{$total_num_files_total}</td>
      <td class="table_cell table_cell_right " align="right" style="font-weight:bold;" >{$total_seconds_str}</td>
    </tr>

  </tbody></table>
    
HERE;

    $config->commit();

} catch (DataPortalException $e) { report_error( $e->toHtml()); }
  catch (LogBookException    $e) { report_error( $e->toHtml()); }
  catch (LusiTimeException   $e) { report_error( $e->toHtml()); }
  catch (FileMgrException    $e) { report_error( $e->toHtml()); }

?>

</div>
</body>
</html>