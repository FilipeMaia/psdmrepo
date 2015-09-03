<?php

/**
 * This service will return the plot for the specified identifier.
 * 
 * Parameters:
 * 
 *   <id>
 */
require_once 'sysmon/sysmon.inc.php' ;

use SysMon\SysMon ;

$id = intval($_GET['id']) ;

SysMon::instance()->begin() ;
$plot = SysMon::instance()->find_psanamon_plot_by_id($id) ;
if (!$plot) die("no such plot found for ID {$id}") ;

header("Content-type: {$plot->type()}") ;
header('Cache-Control: no-cache, must-revalidate') ;    // HTTP/1.1
header('Expires: Sat, 26 Jul 1997 05:00:00 GMT') ;      // Date in the past

echo $plot->data() ;

?>