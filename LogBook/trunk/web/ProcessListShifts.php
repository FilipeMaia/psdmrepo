<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for listing shifts of an
 * experiment.
 */
if(isset($_POST['experiment_name'])) {
    $experiment_name = trim( $_POST['experiment_name'] );
    if( strlen( $experiment_name ) == 0 )
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
        <!------------------------------>
        <h1>All shifts of experiment : <?php echo $experiment->name(); ?></h1>
        <?php
        LogBookTestTable::Shift( "table_4")->show( $experiment->shifts());
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>