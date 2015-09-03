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

    $run = 0 ;
    if (array_key_exists('run', $_GET)) {
      $run = intval($_GET['run']) ;
      if (!$run) report_error("illegal value of parameter 'run'") ;
    }
    $after = LusiTime::parse('2009-09-01 00:00:00') ;
    if (array_key_exists('after', $_GET)) {
      $after = LusiTime::parse($_GET['after']) ;
      if (!$after) report_error("illegal value of parameter 'after'") ;
    }
    $before = LusiTime::now() ;
    if (array_key_exists('before', $_GET)) {
      $before = LusiTime::parse($_GET['before']) ;
      if (!$before) report_error("illegal value of parameter 'before'") ;
    }
    $max_files = 0 ;
    if (array_key_exists('max', $_GET)) {
      $max_files = intval($_GET['max']) ;
      if (!$max_files) report_error("illegal value of parameter 'max'") ;
    }

    $name2stop_time = array() ;
    foreach ($regdb_experiment->data_migration2ana_files() as $f) {
        if ('XTC' !== strtoupper($f->type())) continue ;
        $name2stop_time[$f->name()] = $f->stop_time() ;
    }

    $xmlstr =<<<HERE
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<digest></digest>
HERE;
    $digest = new SimpleXMLElement($xmlstr) ;

    $digest->addChild('subject', 'Placed Entries for SPADE.pdsf') ;
    $digest->addChild('after',   $after ->toStringShortISO()) ;
    $digest->addChild('before',  $before->toStringShortISO()) ;

    // Harvest the files matchingthe requested criteria

    $pattern = '/^e'.$regdb_experiment->id().'-r(\d+)-/' ;
    $num_files = 0 ;
    $type = 'xtc' ;

    $files = array() ;
    FileMgrIrodsWs::files($this_type_files, '/psdm-zone/psdm/'.$regdb_experiment->instrument()->name().'/'.$regdb_experiment->name().'/'.$type) ;
    foreach ($this_type_files as $f) {

        if ($f->type === 'collection') continue ;
        if ($f->resource !== 'lustre-resc') continue ;

        $ctime = new LusiTime($f->ctime) ;

        //$mtime = array_key_exists($f->name, $name2stop_time) ? $name2stop_time[$f->name] : null ;
        //if (!$mtime) continue ;                 // file may have not been migrated
        $mtime = $ctime ;
        if ($mtime->less($after)) continue ;
        if ($before->less($mtime)) continue ;

        preg_match($pattern, $f->name, $matches) ;
        $f_run = intval($matches[1]) ;

        if ($run && $run !== $f_run) continue ;

        $f->ctime = $ctime ;
        $f->mtime = $mtime ;
        $f->run   = $f_run ;

        array_push($files, $f) ;
    }

    // Sort the files from oldest to newest

    usort($files, function ($a, $b) {       
        return $a->mtime->less($b->mtime) ? -1 : ($b->mtime->less($a->mtime) ? 1 : 0) ;
    }) ;

    // Report the result taking into consideration a possible restriction on a total
    // number of files reported by the service.

    $issued = $digest->addChild('issued') ;

    foreach ($files as $f) {

        if ($max_files && $num_files++ >= $max_files) break ;
 
        $change = $issued->addChild('change') ;
        $change->addChild('item',      $f->name) ;
        $change->addChild('time',      $f->mtime->toStringShortISO()) ;
        $change->addChild('type',      strtoupper($type)) ;
        $change->addChild('path',      $f->path) ;
        $change->addChild('ctime',     $f->ctime->toStringShortISO()) ;
        $change->addChild('checksum',  $f->checksum) ;
        $change->addChild('run',       $f->run) ;
    }

    Header('Content-type: text/xml') ;
    print($digest->asXML()) ;

} catch (Exception $e) { report_error($e) ; }

?>