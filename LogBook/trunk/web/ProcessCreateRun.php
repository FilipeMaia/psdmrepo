<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new run
 * in the database.
 */

//print_r($_POST);

if(isset($_POST['num']))
    $num = $_POST['num'];
else
    $num = 0;

if(isset($_POST['experiment_name']))
    $experiment_name = $_POST['experiment_name'];
else
    die( "no valid experiment name" );

if(isset($_POST['begin_time'])) {
    $begin_time = LogBookTime::parse($_POST['begin_time']);
    if(is_null($begin_time))
        die("begin time has invalid format");
} else
    die( "no begin time for run" );

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
    die( "no end time for run" );

$logbook = new LogBook();
$experiment = $logbook->find_experiment_by_name( $experiment_name )
    or die("failed to find the experiment" );
$run = $experiment->create_run( $num, $begin_time, $end_time )
    or die("failed to create the run" );
?>
<!--
The page for reporting the information about all runs of the experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Newely created run</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>All runs of an experiment</h1>
        <h2><?php echo $experiment->name(); ?></h2>
        <?php
        LogBookTestTable::Run()->show( $experiment->runs());
        ?>
    </body>
</html>
