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
     * Return an array representation a dictionary of cables and connectors. The array is
     * suitable for exporting by Web services.
     *
     * @param Irep $irep 
     */
    public static function dict_types2array($irep) {
        $types = array(
            'cable'     => array(),
            'connector' => array()
        );
        foreach( $irep->dict_cables() as $cable ) {

            $connectors = array();
            foreach( $cable->connectors() as $connector ) {
                $connectors[$connector->name()] = array(
                    'id'            => $connector->id(),
                    'documentation' => $connector->documentation(),
                    'created_time'  => $connector->created_time()->toStringShort(),
                    'created_uid'   => $connector->created_uid()
                );
            }
            $types['cable'][$cable->name()] = array(
                'id'            => $cable->id(),
                'documentation' => $cable->documentation(),
                'created_time'  => $cable->created_time()->toStringShort(),
                'created_uid'   => $cable->created_uid(),
                'connector'     => $connectors
            );
        }
        foreach( $irep->dict_connectors() as $connector ) {

            $cables = array();
            foreach( $connector->cables() as $cable ) {
                $cables[$cable->name()] = array(
                    'id'            => $cable->id(),
                    'documentation' => $cable->documentation(),
                    'created_time'  => $cable->created_time()->toStringShort(),
                    'created_uid'   => $cable->created_uid()
                );
            }
            $types['connector'][$connector->name()] = array(
                'id'            => $connector->id(),
                'documentation' => $connector->documentation(),
                'created_time'  => $connector->created_time()->toStringShort(),
                'created_uid'   => $connector->created_uid(),
                'cable'         => $cables
            );
        }
        return $types;
    }

    /**
     * Return an array representation a dictionary of pinlists. The array is
     * suitable for exporting by Web services.
     *
     * @param Irep $irep 
     */
    public static function dict_pinlists2array($irep) {
        $pinlists = array();
        foreach( $irep->dict_pinlists() as $pinlist ) {
            $pinlists[$pinlist->name()] = array(
                'id'                    => $pinlist->id(),
                'documentation'         => $pinlist->documentation(),
                'created_time'          => $pinlist->created_time()->toStringShort(),
                'created_uid'           => $pinlist->created_uid(),
                'cable'                 => $pinlist->cable(),
                'origin_connector'      => $pinlist->origin_connector(),
                'destination_connector' => $pinlist->destination_connector()
            );
        }
        return $pinlists;
    }

    /**
     * Return an array representation a dictionary of locations. The array is
     * suitable for exporting by Web services.
     *
     * @param Irep $irep 
     */
    public static function dict_locations2array($irep) {
        $locations = array();
        foreach( $irep->dict_locations() as $location ) {

            $racks = array();
            foreach( $location->racks() as $rack ) {

                $racks[$rack->name()] = array(
                    'id'           => $rack->id(),
                    'created_time' => $rack->created_time()->toStringShort(),
                    'created_uid'  => $rack->created_uid()
                );
            }
            $locations[$location->name()] = array(
                'id'           => $location->id(),
                'created_time' => $location->created_time()->toStringShort(),
                'created_uid'  => $location->created_uid(),
                'rack'         => $racks
            );
        }
        return $locations;
    }

    /**
     * Return an array representation a dictionary of device name components.
     * The array is suitable for exporting by Web services.
     *
     * @param Irep $irep
     */
    public static function dict_devices2array($irep) {
        $locations = array();
        foreach( $irep->dict_device_locations() as $location ) {

            $regions = array();
            foreach( $location->regions() as $region ) {

                $components = array();
                foreach( $region->components() as $component ) {
                    $components[$component->name()] = array(
                        'id'            => $component->id(),
                        'created_time'  => $component->created_time()->toStringShort(),
                        'created_uid'   => $component->created_uid()
                    );
                }
                $regions[$region->name()] = array(
                    'id'           => $region->id(),
                    'created_time' => $region->created_time()->toStringShort(),
                    'created_uid'  => $region->created_uid(),
                    'component'    => $components
                );
            }
            $locations[$location->name()] = array(
                'id'           => $location->id(),
                'created_time' => $location->created_time()->toStringShort(),
                'created_uid'  => $location->created_uid(),
                'region'       => $regions
            );
        }
        return $locations;
    }

    /**
     * Return an array representation a dictionary of routings. The array is
     * suitable for exporting by Web services.
     *
     * @param Irep $irep 
     */
    public static function dict_routings2array($irep) {
        $routings = array();
        foreach( $irep->dict_routings() as $routing ) {
            $routings[$routing->name()] = array(
                'id'           => $routing->id(),
                'created_time' => $routing->created_time()->toStringShort(),
                'created_uid'  => $routing->created_uid()
            );
        }
        return $routings;
    }

    /**
     * Return an array representation a dictionary of instructions. The array is
     * suitable for exporting by Web services.
     *
     * @param Irep $irep 
     */
    public static function dict_instrs2array($irep) {
        $instrs = array();
        foreach( $irep->dict_instrs() as $instr ) {
            $instrs[$instr->name()] = array(
                'id'           => $instr->id(),
                'created_time' => $instr->created_time()->toStringShort(),
                'created_uid'  => $instr->created_uid()
            );
        }
        return $instrs;
    }

    /**
     * Return an array representation of a project. The array is suitable for
     * exporting by Web services.
     *
     * @param IrepProject $p project
     * @return array
     */
    public static function project2array($p) {
        return array (
            'id'               => $p->id(),
            'owner'            => $p->owner(),
            'comanager'        => $p->comanagers(),
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
     * @param IrepCable $c cable
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
            'revision'     => $c->revision(),
            'description'  => $c->description(),
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
     * Return an array representation of a cable number prefixes.
     * The array is suitable for exporting by Web services.
     *
     * @param Irep $irep
     * @return array 
     */
    public static function cablenumber_prefixes2array($irep) {
        $result = array();
        foreach( $irep->cablenumber_prefixes() as $p )
            array_push(
                $result,
                array (
                    'name'     => $p->name(),
                    'location' => $p->locations(),
                    'range'    => $p->ranges()));
        return $result;
    }

    /**
     * Return an array representation of a cable number ranges associated
     * with a prefix. The array is suitable for exporting by Web services.
     *
     * @param IrepCableNumberPrefix $prefix
     * @return array
     */
    public static function cablenumber_ranges2array($prefix) {
        return $prefix->ranges();
    }

    /**
     * Return an array representing orphan cables.
     *
     * @param Irep $irep
     * @return type 
     */
    public static function cablenumber_orphant2array($irep) {
        return $irep->find_orphant_cables();
    }

    /**
     * Return an array representing reserved cables.
     *
     * @param Irep $irep
     * @return type 
     */
    public static function cablenumber_reserved2array($irep) {
        return $irep->find_reserved_cables();
    }
    /**
     * Return an array representation of a job number allocation. The array
     * is suitable for exporting by Web services.
     *
     * @param IrepJobNumberAlloc $c
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
     * @param IrepJobNumberAlloc $c
     * @return array 
     */
    public static function jobnumber_allocation2array($c) {
        return array (
            'jobnumber_id'                 => $c->jobnumber_id(),
            'jobnumber_name'               => $c->jobnumber_name(),
            'owner'                        => $c->allocation()->owner(),
            'num_cables'                   => $c->num_cables(),
            'allocated_time'               => $c->allocated_time()->toStringShort(),
            'project_title'                => $c->allocation()->irep()->find_project_by_id($c->project_id())->title()
        );
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
                case 'PROJECT':
                    $event['project_id'] = is_null($extra) ? '0' : $extra['project_id'];
                    break;
                case 'CABLE':
                    $event['cable_id'] =  is_null($extra) ? '0' : $extra['cable_id'];
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
