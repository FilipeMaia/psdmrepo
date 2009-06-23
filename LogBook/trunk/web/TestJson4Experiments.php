<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for listing all experiments of an
 * instrument.
 */

$instrument = 'AMO';

try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiments = $logbook->experiments();
    print <<< HERE
{
  "ResultSet": {
    "Result": [

HERE;

    foreach( $experiments as $e ) {
      echo <<< HERE
      { 'id': '{$e->id()}', 'name': '{$e->name()}', 'begin_time': '{$e->begin_time()}',  'end_time': '{$e->end_time()}' }

HERE;
    }
    print <<< HERE
    ]
  }
}

HERE;

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>
