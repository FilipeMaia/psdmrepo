<?php

/**
  * Search equipment on behalf of a Web service and return an array of found items
  *
  * See a list of parameters in a description of the utility function.
  * 
  * @see  \Irep\IrepUtils::find_equipment
  */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {
    return \Irep\IrepUtils::equipment2array(\Irep\IrepUtils::find_equipment($SVC)) ;
}) ;

?>
