<!--
The page for reporting the information about all registered experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Registered experiments</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>Registered experiments</h1>
        <?php
        require_once('LogBook.inc.php');
        try {
            $logbook = new LogBook();
            $logbook->begin();
            LogBookTestTable::Experiment()->show( $logbook->experiments());
            $logbook->commit();
        } catch( LogBookException $e ) {
            print $e->toHtml();
        }
        ?>
    </body>
</html>