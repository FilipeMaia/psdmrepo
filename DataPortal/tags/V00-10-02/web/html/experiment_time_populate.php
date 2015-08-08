<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\ExpTimeMon;
use DataPortal\DataPortalException;

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;

/**
 * This script will populate the system monitoring database with data
 * representing LCLS beam time usage for all known runs.
 *
 */
function report_error($msg) {
	echo $msg;
    exit;
}

function cmp_runs_by_begin_time($a, $b) {
    if($a == $b) return 0;
    return ($a < $b) ? -1 : 1;
}

/* Parse optional parameter to the script:
 *
 *   min_gap_width_sec
 * 
 *     - a positive integer value is expected. The value will be compared
 *       with what's stored in the database, and if the database has a different
 *       value then the full history will be regenerated. Note, that this will
 *       also cover a case when nothing was stored for that parameter in the database. 
 */
$min_gap_width_sec = null;
if( isset($_GET['min_gap_width_sec'])) {
    $min_gap_width_sec = intval(trim($_GET['min_gap_width_sec']));
    if(!$min_gap_width_sec)
        report_error('please, provide a positive number for parameter: min_gap_width_sec');
}
$no_beam_correction4gaps = isset($_GET['no_beam_correction4gaps']);
$force = isset($_GET['force']);

try {
    AuthDB::instance()->begin();

    if (!AuthDB::instance()->hasRole(
            AuthDB::instance()->authName(),
            null,
            'BeamTimeMonitor',
            'Editor')) report_error('not authorized to use this service');

    LogBook::instance()->begin();

    RegDB::instance()->begin();
    ExpTimeMon::instance()->begin();

    // Load configuration parameters stored at the last invocation
    // of the script. The parameters are going to drive how far this script
    // should go back in history. The value of the parameters can also be adjusted
    // 
    //
    $config = ExpTimeMon::instance()->beamtime_config();
    print <<<HERE
<h3>Configuration loaded from the database:</h3>
<div style="padding-left:10;">
  <table><thead>
HERE;
    foreach($config as $param => $value) {
        print <<<HERE
    <tr>
      <td align=left><b>{$param}</b></td>
      <td> : {$value}</td>
    </tr>
HERE;
    }
    print <<<HERE
  </thead><table>
</div>
HERE;

    ExpTimeMon::instance()->populate (
        'XRAY_DESTINATIONS',
        $min_gap_width_sec,
        $no_beam_correction4gaps,
        $force );

    AuthDB::instance()->commit();
    LogBook::instance()->commit();
    RegDB::instance()->commit();
    ExpTimeMon::instance()->commit();

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( DataPortalException $e ) { report_error( $e->toHtml()); }
  catch( LogBookException    $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( RegDBException      $e ) { report_error( $e->toHtml()); }
  catch( Exception           $e ) { report_error( "{$e}" );      }
  
?>
