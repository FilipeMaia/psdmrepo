<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for listing shifts of an
 * experiment.
 */
if( isset( $_POST['experiment_name'] )) {
    $experiment_name = trim( $_POST['experiment_name'] );
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

/* Proceed to the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("no such experiment" );
?>
<!--
The page for reporting the information about all shifts of the experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Shifts of experiment : <?php echo $experiment->name(); ?></title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <h1>All shifts of experiment : <?php echo $experiment->name(); ?></h1>
        <?php
        $shifts = $experiment->shifts();
        foreach( $shifts as $s ) {
            LogBookTestTable::Shift( "table_4" )->show( array( $s ));
            LogBookTestTable::ShiftCrew( "table_6" )->show( $s->crew());
            echo( '<br>' );
        }
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>