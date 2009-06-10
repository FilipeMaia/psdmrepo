<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for closing a shift
 * in the database.
 */
if( isset( $_POST['experiment_name'])) {
    $experiment_name = trim( $_POST['experiment_name'] );
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

if( isset( $_POST['end_time'])) {
    $end_time = LusiTime::parse( trim( $_POST['end_time'] ));
    if( is_null( $end_time ))
        die( "end time has invalid format" );
} else
    die( "no end time for shift" );

/* Proceed to the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("no such experiment" );

    // Find the last shift (if any). Make sure it's not closed yet.
    //
    $shift = $experiment->find_last_shift();
    if( !is_null( $shift )) {
        if( !is_null( $shift->end_time()))
            die( "last shift is already closed" );
        $shift->close( $end_time );
    }
?>
<!--
The page for reporting the information about all runs of the experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>All shifts</title>
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
?>