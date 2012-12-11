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
                        'url'                => $model->url() ,
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
                    'url'              => $manufacturer->url() ,
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
            array_push (
                $equipment ,
                array (
                    'id'                => $e->id() ,
                    'status'            => $e->status() ,
                    'status2'           => $e->status2() ,
                    'manufacturer'      => $e->manufacturer() ,
                    'model'             => $e->model() ,
                    'serial'            => $e->serial() ,
                    'description'       => $e->description() ,
                    'attachment'        => $attachments ,
                    'slacid'            => $e->slacid() ,
                    'pc'                => $e->pc() ,
                    'location'          => $e->location() ,
                    'custodian'         => $e->custodian() ,
                    'modified_time'     => $last_history_event->event_time()->toStringShort() ,
                    'modified_time_sec' => $last_history_event->event_time()->sec ,
                    'modified_uid'      => $last_history_event->event_uid()
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
