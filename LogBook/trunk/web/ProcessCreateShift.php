<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new shift
 * in the database.
 */
if(isset($_POST['leader']))
    $leader = $_POST['leader'];
else
    die( "no valid shift leader account" );

if(isset($_POST['experiment_name']))
    $experiment_name = $_POST['experiment_name'];
else
    die( "no valid experiment name" );

if(isset($_POST['begin_time'])) {
    $begin_time = LogBookTime::parse($_POST['begin_time']);
    if(is_null($begin_time))
        die("begin time has invalid format");
} else
    die( "no begin time for shift" );

if(isset($_POST['end_time'])) {
    $end_time = $_POST['end_time'];
    if($end_time=='')
        $end_time=null;
    else {
        $end_time = LogBookTime::parse($_POST['end_time']);
        if(is_null($end_time))
            die("end time has invalid format");
    }
} else
    die( "no end time for shift" );

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("failed to find the experiment" );

    $shift = $experiment->create_shift( $leader, $begin_time, $end_time );

} catch( LogBookException $e ) {
    print $e->toHtml();
    return;
}?>
<!--
The page for reporting the information about all shifts of the experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Newely created shift</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>All shifts of an experiment</h1>
        <h2><?php echo $experiment->name(); ?></h2>
        <?php
        LogBookTestTable::Shift()->show( $experiment->shifts());
        ?>
    </body>
</html>