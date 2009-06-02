<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for listing runs of an
 * experiment.
 */
if(isset($_POST['experiment_name']))
    $experiment_name = $_POST['experiment_name'];
else
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
The page for reporting the information about all runs of the experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Runs of experiment: <?php echo $experiment->name(); ?></title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>All runs of experiment: <?php echo $experiment->name(); ?></h1>
        <?php
        LogBookTestTable::Run( "table_4")->show( $experiment->runs());
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>