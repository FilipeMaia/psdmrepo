<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new shift
 * in the database.
 */
if( isset( $_POST['leader'] )) {
    $leader = trim( $_POST['leader'] );
    if( $leader == '' )
        die( "shift leader's name can't be empty" );
} else
    die( "no valid shift leader account" );

if( isset( $_POST['experiment_name'] )) {
    $experiment_name = $_POST['experiment_name'];
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

if( isset( $_POST['begin_time'] )) {
    $begin_time = LogBookTime::parse( trim( $_POST['begin_time'] ));
    if( is_null( $begin_time ))
        die( "begin time has invalid format" );
} else
    die( "no begin time for shift" );

if( isset( $_POST['end_time'] )) {
    $str = trim( $_POST['end_time'] );
    if( $str == '' ) $end_time=null;
    else {
        $end_time = LogBookTime::parse( $str );
        if( is_null( $end_time ))
            die( "end time has invalid format" );
    }
} else
    die( "no end time for shift" );

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("failed to find the experiment" );

    $shift = $experiment->create_shift( $leader, $begin_time, $end_time );
?>
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
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}