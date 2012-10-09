<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

function report_error( $msg ) {
    print '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg;

    exit;
}

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

try {

    LogBook::instance()->begin();

    $now = LusiTime::now();
    $now_year = $now->year();
    $experiment_by_year_month = array();
    for ($y = 2009; $y <= $now_year; $y++) {
        $experiment_by_year_month[$y] = array();
        $begin_month = $y == 2009      ? 8 : 1;
        $end_month   = $y == $now_year ? $now->month() : 12;
        for ($m = $begin_month; $m <= $end_month; $m++) {
            $experiment_by_year_month[$y][$m] = array();
        }
    }
    foreach (LogBook::instance()->experiments() as $e) {
        if ($e->is_facility()) continue;
        $last_run = $e->find_last_run();
        if (is_null($last_run)) continue;
        $begin_time = $last_run->begin_time();
        array_push ($experiment_by_year_month[$begin_time->year()][$begin_time->month()], $e->name());
    }
    print <<<HERE
<h3>Experiments groupped by a month when they took their last run</h3>
<div style="padding-left:10px;">

HERE;
    foreach (array_reverse(array_keys($experiment_by_year_month)) as $y) {
        print <<<HERE
  <h4 style="background-color:#c0c0c0;">{$y}</h4>
  <div style="padding-left:10px; text-align:right;">

HERE;
        foreach (array_reverse(array_keys($experiment_by_year_month[$y])) as $m) {
            $month = $month2name[$m];
            print "<div style=\"float:left; font-weight:bold; width:80px;\">{$month}:</div>";
            foreach ($experiment_by_year_month[$y][$m] as $name) {
                print " <div style=\"float:left; width:80px;\">{$name}</div>";
            }
            print "<div style=\"clear:both;\"></div>";
        }
        print "</div>";
    }
    print <<<HERE
</div>

HERE;
    LogBook::instance()->commit();

} catch (LogBookException  $e) { report_error( $e->toHtml()); }
  catch (LusiTimeException $e) { report_error( $e->toHtml()); }

?>
