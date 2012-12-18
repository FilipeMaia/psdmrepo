<?php

/**
 * This service will a dictionary of manufactures/models.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $statuses = array () ;
    foreach ($SVC->irep()->statuses() as $status) {
        $statuses2 = array () ;
        foreach($status->statuses2() as $status2) {
            array_push (
                $statuses2 ,
                array (
                    'id'   => $status2->id() ,
                    'name' => $status2->name()
                )
            ) ;
        }
        array_push (
            $statuses ,
            array (
                'id'      => $status->id() ,
                'name'    => $status->name() ,
                'status2' => $statuses2
            )
        ) ;
    }
    $all_manufacturers = array () ;
    $all_models = array () ;
    foreach ($SVC->irep()->manufacturers() as $manufacturer) {
        $models = array () ;
        foreach($manufacturer->models() as $model) {
            array_push (
                $models ,
                array (
                    'id'   => $model->id() ,
                    'name' => $model->name()
                )
            ) ;
            array_push (
                $all_models ,
                array (
                    'id'           => $model->id() ,
                    'name'         => $model->name() ,
                    'manufacturer' => array (
                        'id'   => $manufacturer->id() ,
                        'name' => $manufacturer->name()
                    )
                )
            ) ;
        }
        array_push (
            $all_manufacturers ,
            array (
                'id'    => $manufacturer->id() ,
                'name'  => $manufacturer->name() ,
                'model' => $models
            )
        ) ;
    }

    $locations = array () ;
    foreach($SVC->irep()->locations() as $location) {
        $rooms = array () ;
        foreach ($location->rooms() as $room) {
            array_push (
                $rooms ,
                array (
                    'id'   => $room->id() ,
                    'name' => $room->name()
                )
            ) ;
        }
        array_push (
            $locations ,
            array (
                'id'   => $location->id() ,
                'name' => $location->name() ,
                'room' => $rooms
            )
        ) ;
    }
    $tags = $SVC->irep()->known_equipment_tags() ;

    $SVC->finish(
        array (
            'option' => array (
                'status'       => $statuses , 
                'manufacturer' => $all_manufacturers ,
                'model'        => $all_models ,
                'location'     => $locations ,
                'custodian'    => $SVC->irep()->known_custodians() ,
                'tag'          => $tags
            )
        )
    ) ;
}) ;

?>
