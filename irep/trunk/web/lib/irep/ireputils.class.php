<?php

namespace Irep;

require_once( 'irep.inc.php' );

/**
 * Class IrepUtils is a utility class accomodating a collection of
 * functions used by Web services.
 *
 * @author gapon
 */
class IrepUtils {

    /**
     * Search equipment on behalf of a Web service an array of found items
     * 
     * The search can be exact if a specific identifier which will
     * uniquely identify the equipment item such as: equipment id, or a SLAC id
     * or a Property Control number. Otherwise the search will be conducted based on
     * partial values of equipment properties. If no paramaters are provided then
     * an empty list will be returned.
     *
     * Parameters:
     * 
     *   Exact search parameters:
     *
     *      <equipment_id> || <slacid> || || <slacid_range_id> || <pc> || <status_id> || <status2_id>
     *
     *   Full text search (logical OR in relevant properties)
     * 
     *      <text2search>
     *
     *   Partial search parameters (logical AND for all):
     *
     *      [<status> [<status2>]]
     *      [<manufacturer> || <manufacturer_id>]
     *      [<model>        || <model_id>]
     *      [<serial>]
     *      [<location> || <location_id>]
     *      [<room>     || <room_id>]
     *      [<custodian>]
     *      [<tag>]
     *      [<description>]
     *      [<notes>]
     * 
     * @param type $SVC
     * @return array
     */
    public static function find_equipment ($SVC) {

        // Check for exact search parameters and trigger the search if
        // any is found.

        $equipment_id = $SVC->optional_int('equipment_id', null) ;
        if ($equipment_id) {
            $equipment = $SVC->irep()->find_equipment_by_id($equipment_id) ;
            if (is_null($equipment)) { $SVC->abort("no equipment for id: {$equipment_id}") ; }
            return array($equipment) ;
        }

        $slacid = $SVC->optional_int('slacid', null) ;
        if ($slacid) {
            $equipment = $SVC->irep()->find_equipment_by_slacid($slacid) ;
            if (is_null($equipment)) { $SVC->abort("no equipment for SLACid: {$slacid}") ; }
            return array($equipment) ;
        }

        $slacid_range_id = $SVC->optional_int('slacid_range_id', null) ;
        if ($slacid_range_id) {
            return $SVC->irep()->find_equipment_by_slacid_range($slacid_range_id) ;
        }

        $pc = $SVC->optional_str('pc', '') ;
        if ($pc != '') {
            $equipment = $SVC->irep()->find_equipment_by_pc($pc) ;
            if (is_null($equipment)) { $SVC->abort("no equipment for Property Control (PC) number: {$pc}") ; }
            return array($equipment) ;
        }

        $status_id = $SVC->optional_int('status_id', null) ;
        if ($status_id) {
            return $SVC->irep()->find_equipment_by_status_id($status_id) ;
        }
        $status2_id = $SVC->optional_int('status2_id', null) ;
        if ($status2_id) {
            return $SVC->irep()->find_equipment_by_status2_id($status2_id) ;
        }

        // Check for a presence of the full text search parameter and trigger the search if
        // the one is found.

        $text2search = $SVC->optional_str('text2search', '') ;
        if ($text2search != '') {
            return $SVC->irep()->find_equipment_by_any($text2search) ;
        }

        // Harvest optional parameters of the partial search. Note that for some of
        // those parameters we have alternatives such as identifiers.

        $status            = $SVC->optional_str('status', '') ;
        $status2           = $SVC->optional_str('status2', '') ;
        $manufacturer_id   = $SVC->optional_int('manufacturer_id', 0) ;
        $manufacturer_name = $SVC->optional_str('manufacturer', '') ;
        if ($manufacturer_id) {
            if ($manufacturer_name === '') {
                $manufacturer = $SVC->irep()->find_manufacturer_by_id($manufacturer_id) ;
                if (!$manufacturer) {
                    $SVC->abort("no manufacturer found for id: {$manufacturer_id}") ;
                }
                $manufacturer_name = $manufacturer->name() ;
            } else {
                $SVC->abort("conflicting parameters for a manufacturer") ;
            }
        }
        $model_id   = $SVC->optional_int('model_id', 0) ;
        $model_name = $SVC->optional_str('model', '') ;
        if ($model_id) {
            if ($model_name == '') {
                $model = $SVC->irep()->find_model_by_id($model_id) ;
                if (!$model) {
                    $SVC->abort("no model found for id: {$model_id}") ;
                }
                $model_name = $model->name() ;
            } else {
                $SVC->abort("conflicting parameters for a model") ;
            }
        }
        $serial        = $SVC->optional_str('serial', '') ;
        $location_id   = $SVC->optional_int('location_id', 0) ;
        $location_name = $SVC->optional_str('location', '') ;
        if ($location_id) {
            if ($location_name == '') {
                $location = $SVC->irep()->find_location_by_id($location_id) ;
                if (!$location) {
                    $SVC->abort("no location found for id: {$location_id}") ;
                }
                $location_name = $location->name() ;
            } else {
                $SVC->abort("conflicting parameters for a location") ;
            }
        }
        $room_id   = $SVC->optional_int('room_id', 0) ;
        $room_name = $SVC->optional_str('room', '') ;
        if ($room_id) {
            if ($room_name == '') {
                $room = $SVC->irep()->find_room_by_id($room_id) ;
                if (!$room) {
                    $SVC->abort("no room found for id: {$room_id}") ;
                }
                $room_name = $room->name() ;
            } else {
                $SVC->abort("conflicting parameters for a room") ;
            }
        }
        $custodian   = $SVC->optional_str('custodian', '') ;
        $tag         = $SVC->optional_str('tag', '') ;
        $description = $SVC->optional_str('description', '') ;
        $notes       = $SVC->optional_str('notes', '') ;

        return  $SVC->irep()->search_equipment (
                $status ,
                $status2 ,
                $manufacturer_name ,
                $model_name ,
                $serial ,
                $location_name ,
                $custodian ,
                $tag ,
                $description ,
                $notes
            ) ;
    }

    public static function event2array($e) {
        return array (
            'scope'          => $e->scope(),
            'scope_id'       => $e->scope_id(),
            'event_uid'      => $e->event_uid(),
            'event'          => $e->event(),
            'comments'       => $e->comments(),
            'event_time_sec' => $e->event_time()->to64(),
            'event_time'     => $e->event_time()->toStringShort()
        );
    }


    /**
     * Return an array representation of a dictionary of manufactures and
     * related models. The array is suitable for exporting by Web services.
     *
     * @param Irep $irep
     */
    public static function manufacturers2array ($irep) {
        $manufacturers = array () ;
        foreach ($irep->manufacturers() as $manufacturer) {
            $models = array () ;
            foreach ($manufacturer->models() as $model) {
                $attachment = $model->default_attachment() ;
                array_push (
                    $models ,
                    array (
                        'id'                 => $model->id() ,
                        'name'               => $model->name() ,
                        'description'        => $model->description() ,
                        'created_time'       => $model->created_time()->toStringShort() ,
                        'created_time_sec'   => $model->created_time()->sec ,
                        'created_uid'        => $model->created_uid() ,
                        'default_attachment' => is_null($attachment) ?
                            array (
                                'is_available'        => 0) :
                            array (
                                'is_available'        => 1 ,
                                'id'                  => $attachment->id() ,
                                'name'                => $attachment->name() ,
                                'document_type'       => $attachment->document_type() ,
                                'document_size_bytes' => $attachment->document_size() ,
                                'create_time'         => $attachment->create_time()->toStringShort() ,
                                'create_uid'          => $attachment->create_uid() ,
                                'rank'                => $attachment->rank())
                    )
                ) ;
            }
            array_push (
                $manufacturers ,
                array (
                    'id'               => $manufacturer->id() ,
                    'name'             => $manufacturer->name() ,
                    'description'      => $manufacturer->description() ,
                    'created_time'     => $manufacturer->created_time()->toStringShort() ,
                    'created_time_sec' => $manufacturer->created_time()->sec ,
                    'created_uid'      => $manufacturer->created_uid() ,
                    'model'            => $models
                )
            ) ;
        }
        return array ('manufacturer' => $manufacturers) ;
    }

    /**
     * Return an array representation of a dictionary of statuses and
     * related sub-statuses. The array is suitable for exporting by Web services.
     *
     * @param Irep $irep
     * @return array()
     */
    public static function statuses2array ($irep) {
        $statuses = array () ;
        foreach ($irep->statuses() as $status) {
            $statuses2 = array () ;
            foreach ($status->statuses2() as $status2)
                array_push (
                    $statuses2 ,
                    array (
                        'id'               => $status2->id() ,
                        'name'             => $status2->name() ,
                        'is_locked'        => $status2->is_locked() ? 1 : 0 ,
                        'created_time'     => $status2->created_time()->toStringShort() ,
                        'created_time_sec' => $status2->created_time()->sec ,
                        'created_uid'      => $status2->created_uid()

                    )
                ) ;
            array_push (
                $statuses ,
                array (
                    'id'               => $status->id() ,
                    'name'             => $status->name() ,
                    'is_locked'        => $status->is_locked() ? 1 : 0 ,
                    'created_time'     => $status->created_time()->toStringShort() ,
                    'created_time_sec' => $status->created_time()->sec ,
                    'created_uid'      => $status->created_uid() ,
                    'status2'          => $statuses2
                )
            ) ;
        }
        return array ('cable_status' => $statuses) ;
    }

    /**
     * Return an array representation of a dictionary of locations.
     * The array is suitable for exporting by Web services.
     *
     * @param Irep $irep
     */
    public static function locations2array ($irep) {
        $locations = array () ;
        foreach ($irep->locations() as $location) {
            $rooms = array () ;
            foreach ($location->rooms() as $room) {
                array_push (
                    $rooms ,
                    array (
                        'id'               => $room->id() ,
                        'name'             => $room->name() ,
                        'created_time'     => $room->created_time()->toStringShort() ,
                        'created_time_sec' => $room->created_time()->sec ,
                        'created_uid'      => $room->created_uid()
                    )
                ) ;
            }
            array_push (
                $locations ,
                array (
                    'id'               => $location->id() ,
                    'name'             => $location->name() ,
                    'created_time'     => $location->created_time()->toStringShort() ,
                    'created_time_sec' => $location->created_time()->sec ,
                    'created_uid'      => $location->created_uid() ,
                    'room'             => $rooms
                )
            ) ;
        }
        return array ('location' => $locations) ;
    }

    /**
     * Return an array representation of equipment.
     * The array is suitable for exporting by Web services.
     *
     * @param IrepEquipment $equipment
     * @return array
     */
    public static function equipment2array ($equipment_list) {
        $equipment = array() ;

        foreach ($equipment_list as $e) {
            $last_history_event = $e->last_history_event() ;

            $attachments = array () ;
            foreach ($e->attachments() as $a)
                array_push (
                    $attachments ,
                        array (
                            'id'                  => $a->id() ,
                            'name'                => $a->name() ,
                            'document_type'       => $a->document_type() ,
                            'document_size_bytes' => $a->document_size() ,
                            'create_time'         => $a->create_time()->toStringShort() ,
                            'create_uid'          => $a->create_uid())) ;

            $tags = array () ;
            foreach ($e->tags() as $t)
                array_push (
                    $tags ,
                        array (
                            'id'          => $t->id() ,
                            'name'        => $t->name() ,
                            'create_time' => $t->create_time()->toStringShort() ,
                            'create_uid'  => $t->create_uid())) ;

            $parent = $e->parent() ;

            $children = array() ;
            foreach ($e->children() as $child) {
                array_push (
                    $children, array (
                        'manufacturer'      => $child->manufacturer() ,
                        'model'             => $child->model() ,
                        'serial'            => $child->serial() ,
                        'slacid'            => $child->slacid() ,
                        'pc'                => $child->pc())) ;
            }
            array_push (
                $equipment ,
                array (
                    'id'                => $e->id() ,
                    'parent'            => array (
                        'id'                => $parent ? $parent->id()           : 0 ,
                        'manufacturer'      => $parent ? $parent->manufacturer() : '' ,
                        'model'             => $parent ? $parent->model()        : '' ,
                        'serial'            => $parent ? $parent->serial()       : '' ,
                        'slacid'            => $parent ? $parent->slacid()       : '' ,
                        'pc'                => $parent ? $parent->pc()           : '') ,
                    'children'          => $children ,
                    'status'            => $e->status() ,
                    'status2'           => $e->status2() ,
                    'manufacturer'      => $e->manufacturer() ,
                    'model'             => $e->model() ,
                    'serial'            => $e->serial() ,
                    'description'       => $e->description() ,
                    'slacid'            => $e->slacid() ,
                    'pc'                => $e->pc() ,
                    'location'          => $e->location() ,
                    'room'              => $e->room() ,
                    'rack'              => $e->rack() ,
                    'elevation'         => $e->elevation() ,
                    'custodian'         => $e->custodian() ,
                    'modified_time'     => $last_history_event->event_time()->toStringShort() ,
                    'modified_time_sec' => $last_history_event->event_time()->sec ,
                    'modified_uid'      => $last_history_event->event_uid() ,
                    'attachment'        => $attachments ,
                    'tag'               => $tags
                )
            ) ;
        }
        return array ('equipment' => $equipment) ;

    }
    public static function equipment_history2array ($equipment) {
        $history = array () ;
        foreach ($equipment->history() as $e)
            array_push (
                $history ,
                array (
                    'event_time' => $e->event_time()->toStringShort() ,
                    'event_time_sec' => $e->event_time()->sec ,
                    'event_uid' => $e->event_uid() ,
                    'event' => $e->event() ,
                    'comments' => $e->comments())) ;
        return $history ;
    }

    /**
     *
     * Return an array representation of a list of known users. The array
     * is suitable for exporting by Web services.
     *
     * @param array of IrepUser $users
     * @return array 
     */
    public static function access2array($users) {
        $result = array();
        foreach( $users as $u ) {
            if( !array_key_exists( $u->role(), $result ))
                $result[$u->role()] = array();
            array_push(
                $result[$u->role()],
                array (
                    'uid'               => $u->uid(),
                    'role'              => $u->role(),
                    'name'              => $u->name(),
                    'added_time'        => $u->added_time()->toStringShort(),
                    'added_uid'         => $u->added_uid(),
                    'last_active_time'  => $u->last_active_time() == '' ? '' : $u->last_active_time()->toStringShort(),
                    'privilege'         => array(
                        'dict_priv'     => $u->has_dict_priv() ? 1 : 0
                    )
                )
            );
        }
        return $result;
    }

    /**
     * Harvest notification info from the database and return an array of
     * data ready to be serialized into a JSON object and be sent to a Web client.
     *
     * @param type $irep
     * @return array of objects ready to be seriealized into JSON
     */
    public static function notifications2array($irep) {

        $access2array = IrepUtils::access2array($irep->users());

        $notifications2array = array();
        $event_types         = array();

        foreach( $irep->notify_event_types() as $e ) {

            $recipient_type = $e->recipient();

            if( !array_key_exists($recipient_type, $notifications2array))
                $notifications2array[$recipient_type] = array();

            if( !array_key_exists($recipient_type,$event_types))
                $event_types[$recipient_type] = array();

            array_push(
                $event_types[$recipient_type],
                array(
                    'name'        => $e->name(),
                    'description' => $e->description()
                )
            );
        }
        $schedule = $irep->notify_schedule();

        foreach( $irep->notifications() as $notify ) {

            $uid            = $notify->uid();
            $event_type     = $notify->event_type(); 
            $recipient_type = $event_type->recipient();

            if( !array_key_exists($uid, $notifications2array[$recipient_type]))
                 $notifications2array[$recipient_type][$uid] = array(
                     'uid' => $uid
                 );

            $notifications2array[$recipient_type][$uid][$event_type->name()] = $notify->enabled();
        }

        $pending = array();
        foreach( $irep->notify_queue() as $entry ) {
            $event_type = $entry->event_type();
            $event = array(
                'id'                     => $entry->id(),
                'event_type_id'          => $event_type->id(),
                'event_type_name'        => $event_type->name(),
                'event_type_description' => $event_type->description(),
                'event_time'             => $entry->event_time()->toStringShort(),
                'event_time_64'          => $entry->event_time()->to64(),
                'originator_uid'         => $entry->originator_uid(),
                'recipient_uid'          => $entry->recipient_uid(),
                'recipient_role'         => $event_type->recipient_role_name(),
                'scope'                  => $event_type->scope()
            );
            $extra = $entry->extra();
            switch($event_type->scope()) {
                case 'EQUIPMENT':
                    $event['equipment_id'] = is_null($extra) ? '0' : $extra['project_id'];
                    break;
            }
            array_push($pending, $event);
        }
        return array(
            'access'      => $access2array,
            'event_types' => $event_types,
            'schedule'    => $schedule,
            'notify'      => $notifications2array,
            'pending'     => $pending );
    }
}
?>
