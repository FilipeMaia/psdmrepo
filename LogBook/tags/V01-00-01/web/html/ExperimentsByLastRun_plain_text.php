<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTimeException;


try {

    LogBook::instance()->begin();

    $time2experiment = array();
    foreach (LogBook::instance()->experiments() as $e) {
        if ($e->is_facility()) continue;
        $last_run = $e->find_last_run();
        if (is_null($last_run)) continue;
        $begin_time = $last_run->begin_time();
        $time2experiment[intval($begin_time->sec)] = $e;
    }
    $timestamps = array_keys($time2experiment);
    sort($timestamps);
//    echo '<pre>'.print_r($time2experiment,true).'</pre>';
    foreach($timestamps as $t) {
        $e = $time2experiment[$t];
        $i = $e->regdb_experiment()->instrument();
        echo "<br>".$i->name()." ".$e->name()." ".$t;
    }

    LogBook::instance()->commit();

} catch (LogBookException  $e) { report_error( $e->toHtml()); }
  catch (LusiTimeException $e) { report_error( $e->toHtml()); }

?>
