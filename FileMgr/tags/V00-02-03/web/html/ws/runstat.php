<?php

require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use LogBook\LogBook ;
use LusiTime\LusiTime ;
use FileMgr\FileMgrIrodsWs ;

function report_error ($msg) {
    $xmlstr =<<<HERE
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<error>{$msg}</error>
HERE;
    $error = new SimpleXMLElement($xmlstr) ;
    Header('Content-type: text/xml') ;
    print($error->asXML()) ;
    exit ;
}
try {

    # Parse parameters of the script

    if (!array_key_exists('exp_name', $_GET)) report_error("missing parameter 'exp_name'") ;
    $exp_name = trim($_GET['exp_name']) ;

    LogBook::instance()->begin() ;
    $logbook_experiment = LogBook::instance()->find_experiment_by_name($exp_name) ;
    if (!$logbook_experiment) report_error("no such experiment '{$exp_name}'") ;

    $regdb_experiment = $logbook_experiment->regdb_experiment() ;

    if (!array_key_exists('run', $_GET)) report_error("missing parameter 'run'") ;
    $run_num = intval($_GET['run']) ;
    if (!$run_num) report_error("illegal value of parameter 'run'") ;

    $run = $logbook_experiment->find_run_by_num($run_num) ;
    if (!$run) report_error("no such run {$run_num} for the experiment {$exp_name}") ;

    $type = 'xtc' ;

    // File availability in IRODS will be calculated later. Right now
    // we're just building a catalog of all known files for the run
    // as they're reported by the DAQ system.

    $file2available = array() ;
    foreach ($regdb_experiment->files($run_num) as $f) {
        $xtc_name = "{$f->base_name()}.{$type}" ;
        $file2available[$xtc_name] = false ;
    }

    $xmlstr =<<<HERE
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<run_summary></run_summary>
HERE;
    $digest_xml = new SimpleXMLElement($xmlstr) ;

    $digest_xml->addChild('subject', 'Run Info for SPADE.pdsf') ;
    $digest_xml->addChild('run',     $run_num) ;
    $digest_xml->addChild('ended',   $run->end_time() ? 1 : 0) ;

    // Harvest the files matching the requested criteria

    $pattern = '/^e'.$regdb_experiment->id().'-r(\d+)-/' ;

    $irods_files = array() ;
    FileMgrIrodsWs::files($irods_files, '/psdm-zone/psdm/'.$regdb_experiment->instrument()->name().'/'.$regdb_experiment->name().'/'.$type) ;
    foreach ($irods_files as $f) {

        if ($f->type === 'collection') continue ;
        if ($f->resource !== 'lustre-resc') continue ;

        preg_match($pattern, $f->name, $matches) ;
        $f_run = intval($matches[1]) ;

        if ($run_num !== $f_run) continue ;

        $file2available[$f->name] = true ;
    }

    // Report the result

    $files_xml = $digest_xml->addChild('files') ;

    foreach ($file2available as $name => $available) {
 
        $file_xml = $files_xml->addChild('file') ;
        $file_xml->addChild('name',      $name) ;
        $file_xml->addChild('available', $available ? 1 : 0) ;
    }

    Header('Content-type: text/xml') ;
    print($digest_xml->asXML()) ;

} catch (Exception $e) { report_error($e) ; }

?>