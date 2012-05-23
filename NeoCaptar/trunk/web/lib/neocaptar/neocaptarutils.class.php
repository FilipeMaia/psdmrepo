<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarUtils is a utility class accomodating a collection of
 * functions used by Web services.
 *
 * @author gapon
 */
class NeoCaptarUtils {
    /**
     * Return an array representation of a project. The array is suitable for
     * exporting by Web services.
     *
     * @param NeoCaptarProject $p project
     * @return array
     */
    public static function project2array($p) {
        return array (
            'id'               => $p->id(),
            'owner'            => $p->owner(),
            'title'            => $p->title(),
            'job'              => $p->job(),
            'description'      => $p->description(),
            'created_sec'      => $p->created_time()->sec,
            'created'          => $p->created_time()->toStringDay(),
            'due_sec'          => $p->due_time()->sec,
            'due'              => $p->due_time()->toStringDay(),
            'modified_sec'     => $p->modified_time()->sec,
            'modified'         => $p->modified_time()->toStringShort(),
            'status'           => $p->status()
        );
    }
        
    /**
     * Return an array representation of a cable. The array is suitable for
     * exporting by Web services.
     *
     * [ Note that the project identifier is an optional parameter. If none
     *   is provided then each cable will be asked for its owner project's
     *   identifier. ]
     *
     * @param NeoCaptarCable $c cable
     * @param integer $project_id an optional identifier of a project
     * @return array
     */
    public static function cable2array($c,$project_id=null) {
        $last_event = $c->history_last_entry();
        return array (
            'id'           => $c->id(),
            'project_id'   => $c->project()->id(),
            'project_title'=> $c->project()->title(),
            'status'       => $c->status(),
            'job'          => $c->project()->job(),
            'cable'        => $c->cable(),
            'device'           => $c->device(),
            'device_location'  => $c->device_location(),
            'device_region'    => $c->device_region(),
            'device_component' => $c->device_component(),
            'device_counter'   => $c->device_counter(),
            'device_suffix'    => $c->device_suffix(),
            'func'         => $c->func(),
            'cable_type'   => $c->cable_type(),
            'length'       => $c->length(),
            'routing'      => $c->routing(),
            'origin'       => array (
                'name'     => $c->origin_name(),
                'loc'      => $c->origin_loc(),
                'rack'     => $c->origin_rack(),
                'ele'      => $c->origin_ele(),
                'side'     => $c->origin_side(),
                'slot'     => $c->origin_slot(),
                'conn'     => $c->origin_conn(),
                'pinlist'  => $c->origin_pinlist(),
                'station'  => $c->origin_station(),
                'conntype' => $c->origin_conntype(),
                'instr'    => $c->origin_instr()
            ),
            'destination'  => array (
                'name'     => $c->destination_name(),
                'loc'      => $c->destination_loc(),
                'rack'     => $c->destination_rack(),
                'ele'      => $c->destination_ele(),
                'side'     => $c->destination_side(),
                'slot'     => $c->destination_slot(),
                'conn'     => $c->destination_conn(),
                'pinlist'  => $c->destination_pinlist(),
                'station'  => $c->destination_station(),
                'conntype' => $c->destination_conntype(),
                'instr'    => $c->destination_instr()
            ),
            'proj'         => array (
                'id'       => is_null($project_id) ? $c->project()->id() : $project_id
            ),
            'modified'     => is_null($last_event) ?
                              array(
                                  'time'    => '',
                                  'time_64' => 0,
                                  'uid'     => ''
                              ) :
                              array(
                                  'time'    => $last_event->event_time()->toStringShort(),
                                  'time_64' => $last_event->event_time()->to64(),
                                  'uid'     => $last_event->event_uid()
                              )
        );
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
     * Return an array representation of a cable number allocation. The array
     * is suitable for exporting by Web services.
     *
     * @param NeoCaptarCableNumberAlloc $c
     * @return array 
     */
    public static function cablenumber2array($c) {
        return array (
            'id'                     => $c->id(),
            'location'               => $c->location(),
            'prefix'                 => $c->prefix(),
            'first'                  => $c->first(),
            'last'                   => $c->last(),
            'num_in_use'             => $c->num_in_use(),
            'num_available'          => $c->num_available(),
            'next_available'         => is_null($c->next_available        ()) ? '' : $c->next_available        (),
            'recently_allocated'     => is_null($c->recently_allocated    ()) ? '' : $c->recently_allocated    (),
            'recently_allocated_name'=> is_null($c->recently_allocated    ()) ? '' : sprintf("%2s%05d",$c->prefix(),$c->recently_allocated()),
            'recent_allocation_time' => is_null($c->recent_allocation_time()) ? '' : $c->recent_allocation_time()->toStringShort(),
            'recent_allocation_uid'  => is_null($c->recent_allocation_uid ()) ? '' : $c->recent_allocation_uid ()
        );
    }

    /**
     * Return an array representation of a job number allocation. The array
     * is suitable for exporting by Web services.
     *
     * @param NeoCaptarJobNumberAlloc $c
     * @return array 
     */
    public static function jobnumber2array($c) {
        return array (
            'id'                           => $c->id(),
            'owner'                        => $c->owner(),
            'prefix'                       => $c->prefix(),
            'first'                        => $c->first(),
            'last'                         => $c->last(),
            'num_in_use'                   => $c->num_in_use(),
            'num_available'                => $c->num_available(),
            'next_available'               => is_null($c->next_available        ()) ? '' : $c->next_available              (),
            'recently_allocated'           => is_null($c->recently_allocated    ()) ? '' : $c->recently_allocated          (),
            'recently_allocated_name'      => is_null($c->recently_allocated    ()) ? '' : sprintf("%3s%03d",$c->prefix(),$c->recently_allocated()),
            'recent_allocation_time'       => is_null($c->recent_allocation_time()) ? '' : $c->recent_allocation_time      ()->toStringShort(),
            'recent_allocation_uid'        => is_null($c->recent_allocation_uid ()) ? '' : $c->recent_allocation_uid       ()
        );
    }
    /**
     *
     * Return an array representation of a job number. The array
     * is suitable for exporting by Web services.
     *
     * @param NeoCaptarJobNumberAlloc $c
     * @return array 
     */
    public static function jobnumber_allocation2array($c) {
        return array (
            'jobnumber_id'                 => $c->jobnumber_id(),
            'jobnumber_name'               => $c->jobnumber_name(),
            'owner'                        => $c->allocation()->owner(),
            'num_cables'                   => $c->num_cables(),
            'allocated_time'               => $c->allocated_time()->toStringShort(),
            'project_title'                => $c->allocation()->neocaptar()->find_project_by_id($c->project_id())->title()
        );
    }

    /**
     *
     * Return an array representation of a list of known users. The array
     * is suitable for exporting by Web services.
     *
     * @param array of NeoCaptarUser $users
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
                    'last_active_time'  => $u->last_active_time() == '' ? '' : $u->last_active_time()->toStringShort()
                )
            );
        }
        return $result;
    }

    /**
     * Harvest notification info from the database and return an array of
     * data ready to be serialized into a JSON object and be sent to a Web client.
     *
     * @param type $neocaptar
     * @return array of objects ready to be seriealized into JSON
     */
    public static function notifications2array($neocaptar) {

        $access2array = NeoCaptarUtils::access2array($neocaptar->users());

        $notifications2array = array();
        $event_types         = array();

        foreach( $neocaptar->notify_event_types() as $e ) {

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
        $schedule = $neocaptar->notify_schedule();

        foreach( $neocaptar->notifications() as $notify ) {

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
        foreach( $neocaptar->notify_queue() as $entry ) {
            array_push(
                $pending,
                array(
                    'id'              => $entry->id(),
                    'event_type_id'   => $entry->event_type()->id(),
                    'event_type_name' => $entry->event_type()->name(),
                    'event_time'      => $entry->event_time()->toStringShort(),
                    'event_time_64'   => $entry->event_time()->to64(),
                    'originator_uid'  => $entry->originator_uid(),
                    'recipient_uid'   => $entry->recipient_uid()
                )
            );
        }
        return array(
            'access'      => $access2array,
            'event_types' => $event_types,
            'schedule'    => $schedule,
            'notify'      => $notifications2array,
            'pending'     => $pending );
    }

    /**
     * Report (print) a JSON object with optional result.
     * 
     * @param array $result an optional array of results (key/value pairs)
     */
    public static function report_success($result=null) {
        print
        	'{ "status": '.json_encode("success").
            ', "updated": '.json_encode( LusiTime::now()->toStringShort());
        if( !is_null($result))
            foreach( $result as $k => $v)
                print ','.json_encode($k).':'.json_encode($v);
        print
       		'}';
    }

    /**
     * Package the error message into a JSON object and return the one back
     * to a caller. The script's execution will end at this point.
     *
     * @param string $msg a message to be returned to a caller
     */
    public static function report_error($msg) {
        print '{"status": '.json_encode('error').', "message": '.json_encode( $msg ).'}';
        exit;
    }

    /**
     * A generic parameter handler function. It will scan the specified source
     * array for a parameter/value pair and return the value (if found). Errors
     * will be reported as JSON objects.
     *
     * @param array $source - a dictionary where to look for the parameter
     * @param string $name - a name of the parameter
     * @param booleans $required - true if the parameter is required
     * @param boolean $allow_empty - true if the parameter is allowed to be empty
     * @return string 
     */
    public static function get_param($source, $name, $required, $allow_empty) {
        if(!isset($source[$name])) {
            if($required) NeoCaptarUtils::report_error('missing parameter: '.$name);
            return null;
        }
        $param = trim($source[$name]);
        if(!$allow_empty && $param == '') NeoCaptarUtils::report_error('empty value of parameter: '.$name);
        return $param;
    }
    public static function get_param_GET($name, $required=true, $allow_empty=false) {
        return NeoCaptarUtils::get_param($_GET, $name, $required, $allow_empty);
    }
    public static function get_param_POST($name, $required=true, $allow_empty=false) {
        return NeoCaptarUtils::get_param($_POST, $name, $required, $allow_empty);
    }

    public static function get_param_GET_time($name, $required=true, $allow_empty=false, $source_has_hours=false) {
        $str = NeoCaptarUtils::get_param_GET($name, $required, $allow_empty);
        return is_null($str) || ($str == '') ? null : LusiTime::parse($str.($source_has_hours ? '' : ' 00:00:00'));
    }
}
?>
