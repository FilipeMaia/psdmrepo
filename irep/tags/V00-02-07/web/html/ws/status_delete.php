<?php

/**
 * This service will delete the specified status or a sub-status and return an updated dictionary.
 * 
 * Parameters:
 * 
 *   <scope> <id>
 * 
 * Where:
 * 
 *   <scope> := { status | status2 }
 *   <id> is an identifier or a status or a sub-status depending on the scope
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->has_dict_priv() or
        $SVC->abort('your account not authorized for the operation') ;

    $scope = $SVC->required_str('scope') ;
    $id    = $SVC->required_int('id') ;

    switch ($scope) {
        case 'status'  :

            $status = $SVC->irep()->find_status_by_id ($id) ;
            if (is_null($status))
                $SVC->abort("no status for ID: {$id}") ;

            if ($status->is_locked())
                $SVC->abort("the status with ID: {$id} is lockeed and it can't be deleted by a user") ;

            $num_equipment = count ($SVC->irep()->search_equipment_by_status ($status->name())) ;
            if ($num_equipment)
                $SVC->abort (
                    "{$num_equipment} equipment(s) registered with status {$status->name()}.".
                    " Please, migrate them to a different status first.") ;

            $SVC->irep()->delete_status_by_id ($id) ;

            break ;

        case 'status2' :

            $status2 = $SVC->irep()->find_status2_by_id ($id) ;
            if (is_null($status2))
                $SVC->abort("no sub-status for ID: {$id}") ;

            if ($status2->is_locked())
                $SVC->abort("the sub-status with ID: {$id} is lockeed and it can't be deleted by a user") ;

            $num_equipment = count ($SVC->irep()->search_equipment_by_status ($status2->status()->name(), $status2->name())) ;
            if ($num_equipment)
                $SVC->abort (
                    "{$num_equipment} equipment(s) registered with status {$status2->status()->name()}::{$status2->name()}.".
                    " Please, migrate them to a different status first.") ;

            $SVC->irep()->delete_status2_by_id($id) ;

            break ;

        default:
            $SVC->abort("unsupported scope of the operation: {$scope}") ;
    }
    $SVC->finish(\Irep\IrepUtils::statuses2array($SVC->irep())) ;
}) ;

?>
