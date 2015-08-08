<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \AuthDB\AuthDB ;
use \FileMgr\DbConnection ;
use \LusiTime\LusiTime ;

/**
 * Class Irep encapsulates operations with the Inventory and Repairs database
 *
 * @author gapon
 */
class Irep extends DbConnection {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $instance = null ;

    /**
     * Singleton to simplify certain operations.
     *
     * @return Irep
     */
    public static function instance () {
        if (is_null(Irep::$instance)) Irep::$instance =
            new Irep (
                IREP_DEFAULT_HOST,
                IREP_DEFAULT_USER,
                IREP_DEFAULT_PASSWORD,
                IREP_DEFAULT_DATABASE) ;
        return Irep::$instance ;
    }

    /**
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     *
     * @param string $host
     * @param string $user
     * @param string $password
     * @param string $database
     */
    public function __construct ($host, $user, $password, $database) {
        parent::__construct ($host, $user, $password, $database) ;
    }

    /* ------------------
     *   SLACid numbers
     * ------------------
     */
    public function slacid_ranges () {
        $list = array () ;
        $result = $this->query("SELECT * FROM {$this->database}.slacid_range ORDER BY first");
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            $range = array (
                'id'          => intval($attr['id']) ,
                'first'       => intval($attr['first']) ,
                'last'        => intval($attr['last']) ,
                'description' =>   trim($attr['description'])) ;
            $range['available'] = $this->find_available_slacid($range) ;
            array_push($list, $range) ;
        }
        return $list ;
    }
    public function find_slacid_range_for ($slacid) {
        $slacid = intval($slacid) ;
        $result = $this->query("SELECT * FROM {$this->database}.slacid_range WHERE first <= {$slacid} AND {$slacid} <= last") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1)
            throw new IrepException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt.") ;
        $attr = mysql_fetch_array($result, MYSQL_ASSOC) ; 
        $range = array (
            'id'          => intval($attr['id'] ),
            'first'       => intval($attr['first']) ,
            'last'        => intval($attr['last']) ,
            'description' =>   trim($attr['description'])) ;
        $range['available'] = count($this->find_available_slacid($range)) ;
        return $range ;
    }
    public function find_slacid_range ($id) {
        $id = intval($id) ;
        $result = $this->query("SELECT * FROM {$this->database}.slacid_range WHERE id={$id}") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1)
            throw new IrepException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt.") ;
        $attr = mysql_fetch_array($result, MYSQL_ASSOC) ; 
        $range = array (
            'id'          => intval($attr['id'] ),
            'first'       => intval($attr['first']) ,
            'last'        => intval($attr['last']) ,
            'description' =>   trim($attr['description'])) ;
        $range['available'] = count($this->find_available_slacid($range)) ;
        return $range ;
    }
    private function find_available_slacid ($range) {
        $list      = array() ;
        $range_id  = $range['id'] ;   
        $first     = $range['first' ];   
        $last      = $range['last'] ;
        $allocated = array () ;
        $result = $this->query("SELECT slacid FROM {$this->database}.slacid_allocated WHERE range_id={$range_id} AND {$first} <= slacid AND slacid <= {$last} ORDER BY slacid") ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            array_push($allocated, intval($attr['slacid'])) ;
        }
        return array_diff(range($first, $last), $allocated) ;
    }
    public function find_slacid ($slacid) {
        $slacid = intval($slacid) ;
        return $this->find_slacid_("slacid={$slacid}") ;
    }
    public function find_slacid_for ($equipment_id) {
        $equipment_id = intval($equipment_id) ;
        return $this->find_slacid_("equipment_id={$equipment_id}") ;
    }
    private function find_slacid_ ($condition) {
        $sql = "SELECT * FROM {$this->database}.slacid_allocated WHERE {$condition}" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
        return array (
            'range_id'         => intval($attr['range_id']) ,
            'slacid'           => intval($attr['slacid']) ,
            'equipment_id'     => intval($attr['equipment_id']) ,
            'allocated_time'   => LusiTime::from64(intval($attr['allocated_time'])) ,
            'allocated_by_uid' => trim($attr['allocated_by_uid'])
        ) ;
    }
    public function add_slacid_range ($first, $last, $description) {
        $description_escaped = $this->escape_string(trim($description)) ;
        $sql = "INSERT INTO {$this->database}.slacid_range VALUES (NULL,{$first},{$last},'{$description_escaped}')" ;
        $this->query($sql) ;
    }
    public function update_slacid_range ($range_id, $first, $last, $description) {
        $description_escaped = $this->escape_string(trim($description)) ;
        $sql = "UPDATE {$this->database}.slacid_range SET first={$first}, last={$last}, description='{$description_escaped}' WHERE id={$range_id}" ;
        $this->query($sql) ;
    }
    public function delete_slacid_range ($range_id) {
        $this->query("DELETE FROM {$this->database}.slacid_range WHERE id={$range_id}") ;
    }
    private function allocate_slacid ($slacid, $equipment_id, $uid) {
        $slacid       = intval($slacid);
        $equipment_id = intval($equipment_id);

        // First check if a number has already been allocated for this
        // equipment in one of the ranges.
        //
        if (!is_null($this->find_slacid_for($equipment_id)))
            throw new IrepException (
        		__METHOD__, "SLACid number is already in use.") ;

        // Check if the number falls into one of the known ranges
        //
        foreach ($this->slacid_ranges() as $range) {
            if (($range['first' ] <= $slacid) && ($slacid <= $range['last'])) {
                $range_id          = $range['id'] ;   
                $allocated_time_64 = LusiTime::now()->to64() ;
                $this->query("INSERT {$this->database}.slacid_allocated VALUES ({$range_id},{$slacid},{$equipment_id},{$allocated_time_64},'{$uid}')") ;
                return ;
            }
        }
        throw new IrepException(__METHOD__, "SLAC ID {$slacid} is beyond of any supported range") ;
    }
    private function free_slacid ($slacid) {
        $slacid = intval($slacid);
        $this->query("DELETE FROM {$this->database}.slacid_allocated WHERE slacid={$slacid}") ;
    }

    /* -------------------
     *   Users and roles
     * -------------------
     */
    public function is_other () {
        $user = $this->current_user() ;
        return !is_null($user) && $user->is_other();
    }
    public function is_administrator () {
        $user = $this->current_user() ;
        return !is_null($user) && $user->is_administrator( ) ;
    }
    public function can_edit_inventory () {
        $user = $this->current_user() ;
        return !is_null($user) && ($user->is_administrator() || $user->is_editor()) ;
    }
    public function has_dict_priv () {
        $user = $this->current_user() ;
        return !is_null($user) && $user->has_dict_priv() ;
    }
    public function current_user () {
        return $this->find_user_by_uid(AuthDB::instance()->authName()) ;
    }
    public function users () {
        $list = array () ;
        $result = $this->query("SELECT * FROM {$this->database}.user ORDER BY uid,role") ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepUser (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function find_user_by_uid ($uid) {
        $uid_escaped = $this->escape_string(trim($uid)) ;
        $result = $this->query("SELECT * FROM {$this->database}.user WHERE uid='{$uid_escaped}'" );
        $nrows = mysql_numrows( $result ) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." ) ;
        return new IrepUser (
            $this ,
            mysql_fetch_array($result, MYSQL_ASSOC)) ;
    }
    public function add_user ($uid, $name, $role) {
        $uid_escaped = $this->escape_string(trim($uid)) ;
        $role_uc = strtoupper(trim($role)) ;
        $name_escaped = $this->escape_string(trim($name)) ;
        $added_time = LusiTime::now()->to64() ;
        $added_uid_escaped = $this->escape_string(trim(AuthDB::instance()->authName())) ;
        $sql = "INSERT INTO {$this->database}.user VALUES('{$uid_escaped}','{$role_uc}','{$name_escaped}',{$added_time},'{$added_uid_escaped}',NULL)" ;
        $this->query($sql) ;
        return $this->find_user_by_uid($uid) ;
    }
    public function delete_user ($uid) {
        $uid_escaped = $this->escape_string(trim($uid)) ;
        $this->query("DELETE FROM {$this->database}.user WHERE uid='{$uid_escaped}'") ;
        $this->query("DELETE FROM {$this->database}.notify WHERE uid='{$uid_escaped}'") ;
    }
    public function update_current_user_activity () {
        $user = $this->current_user() ;
        if(is_null($user)) return ;
        $uid_escaped = $this->escape_string(trim($user->uid())) ;
        $current_time = LusiTime::now()->to64() ;
        $sql = "UPDATE {$this->database}.user SET last_active_time={$current_time} WHERE uid='{$uid_escaped}'" ;
        $this->query($sql) ;
    }
    public function known_custodians () {
        $list = array () ;
        $result = $this->query("SELECT DISTINCT custodian FROM {$this->database}.equipment ORDER BY custodian") ;
        for ($i = 0, $nrows = mysql_numrows( $result ) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $custodian = trim($attr['custodian']) ;
            if ($custodian == '') continue ;
            array_push ($list, $custodian) ;
        }
        return $list ;
    }

    /* -----------------
     *   Notifications
     * -----------------
     */
    public function notify_event_types ($recipient=null) {
        $list = array () ;
        $recipient_condition = is_null($recipient) ? '' : "WHERE recipient='{$this->escape_string(trim($recipient))}'" ;
        $result = $this->query("SELECT * FROM {$this->database}.notify_event_type {$recipient_condition} ORDER BY recipient,scope,id") ;
        for ($i = 0, $nrows = mysql_numrows( $result ) ; $i < $nrows ; $i++)
            array_push (
                $list,
                new IrepNotifyEventType (
                    $this,
                    mysql_fetch_array($result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function find_notify_event_type_by_id ($id) {
        $result = $this->query("SELECT * FROM {$this->database}.notify_event_type WHERE id={$id}") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt.") ;
        return new IrepNotifyEventType(
                $this,
                mysql_fetch_array($result, MYSQL_ASSOC)) ;
    }
    public function find_notify_event_type ($recipient,$name) {
        $recipient_escaped = $this->escape_string(trim($recipient)) ;
        $name_escaped      = $this->escape_string(trim($name)) ;
        $result = $this->query("SELECT * FROM {$this->database}.notify_event_type WHERE recipient='{$recipient_escaped}' and name='{$name_escaped}'") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt.") ;
        return new IrepNotifyEventType (
                $this,
                mysql_fetch_array($result, MYSQL_ASSOC)) ;
    }
    public function notify_schedule () {
        $dictionary = array () ;
        $result = $this->query("SELECT * FROM {$this->database}.notify_schedule") ;
        for ($i = 0, $nrows = mysql_numrows( $result ) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $dictionary[$attr['recipient']] = $attr['mode'] ;
        }
        return $dictionary ;
    }
    public function notifications ($uid=null) {
        $list = array () ;
        $uid_condition = is_null($uid) ? '' : "WHERE uid='{$this->escape_string(trim($uid))}'" ;
        $result = $this->query("SELECT * FROM {$this->database}.notify {$uid_condition}") ;
        for ($i = 0, $nrows = mysql_numrows( $result ) ; $i < $nrows ; $i++)
            array_push (
                $list,
                new IrepNotify (
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC ))) ;
        return $list ;
    }
    public function find_notification_by_id ($id) {
        return $this->find_notification_by_("id={$id}") ;
    }
    public function find_notification($uid, $event_type_id) {
        $uid_escaped  = $this->escape_string(trim($uid)) ;
        return $this->find_notification_by_("uid='{$uid_escaped}' AND event_type_id={$event_type_id}") ;
    }
    public function find_notification_by_($condition) {
        $result = $this->query("SELECT * FROM {$this->database}.notify WHERE {$condition}") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt.") ;
        return new IrepNotify (
                $this,
                mysql_fetch_array($result, MYSQL_ASSOC)) ;
    }
    public function notify_queue () {
        $list = array () ;
        $result = $this->query("SELECT * FROM {$this->database}.notify_queue ORDER BY event_time") ;
        for ($i = 0, $nrows = mysql_numrows( $result ) ; $i < $nrows ; $i++)
            array_push (
                $list,
                new IrepNotifyQueuedEntry (
                    $this,
                    mysql_fetch_array($result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function find_notify_queue_entry_by_id ($id) {
        return $this->find_notify_queue_entry_by_("id={$id}") ;
    }
    public function find_notify_queue_entry_by_ ($condition) {
        $result = $this->query("SELECT * FROM {$this->database}.notify_queue WHERE {$condition}") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt.") ;
        return new IrepNotifyQueuedEntry (
            $this,
            mysql_fetch_array($result, MYSQL_ASSOC)) ;
    }
    public function add_notification ($uid, $event_type_id, $enabled) {
        $uid_escaped  = $this->escape_string(trim($uid)) ;
        $enabled_flag = $enabled ? 'ON' : 'OFF' ;
        $this->query("INSERT INTO {$this->database}.notify VALUES(NULL,'{$uid_escaped}',{$event_type_id},'{$enabled_flag}')") ;
        return $this->find_notification_by_('id IN (SELECT LAST_INSERT_ID())') ;       
    }
    public function update_notification ($id, $enabled) {
        $enabled_flag = $enabled ? 'ON' : 'OFF' ;
        $this->query("UPDATE {$this->database}.notify SET enabled='{$enabled_flag}' WHERE id={$id}") ;
        return $this->find_notification_by_id($id) ;       
    }
    public function update_notification_schedule ($recipient, $policy) {
        $recipient_escaped = $this->escape_string(trim($recipient)) ;
        $policy_escaped    = $this->escape_string(trim($policy)) ;
        $this->query("UPDATE {$this->database}.notify_schedule SET mode='{$policy_escaped}' WHERE recipient='{$recipient_escaped}'") ;
    }
    
    /* ----------------------------
     *   Manufacturers and models
     * ----------------------------
     */
    public function manufacturers () {
        $list = array () ;
        $result = $this->query("SELECT * FROM {$this->database}.dict_manufacturer ORDER BY name") ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepManufacturer (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function add_manufacturer ($name, $description='') {
        $name_escaped = $this->escape_string(trim($name)) ;
        $description_escaped = $this->escape_string(trim($description)) ;
        $created_time = LusiTime::now()->to64() ;
        $created_uid_escaped = $this->escape_string(trim(AuthDB::instance()->authName())) ;
        $sql = "INSERT INTO {$this->database}.dict_manufacturer VALUES(NULL,'{$name_escaped}','{$description_escaped}',{$created_time},'{$created_uid_escaped}')" ;
        $this->query($sql) ;
        return $this->find_manufacturer_by_('id=(SELECT LAST_INSERT_ID())') ;
    }
    public function find_manufacturer_by_name ($name) {
        $name_escaped = $this->escape_string(trim($name)) ;
        return $this->find_manufacturer_by_("name='{$name_escaped}'") ;
    }
    public function find_manufacturer_by_id ($id) {
        $id = intval($id) ;
        return $this->find_manufacturer_by_("id={$id}") ;
    }
    private function find_manufacturer_by_ ($condition='') {
        $conditions_opt = $condition ? " WHERE {$condition}" : '' ;
        $sql = "SELECT * FROM {$this->database}.dict_manufacturer {$conditions_opt}" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepManufacturer (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }
    public function delete_manufacturer ($id) {
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->database}.dict_manufacturer WHERE id={$id}" ;
        $this->query($sql) ;
    }
    public function find_model_by_id ($id) {
        $id = intval($id) ;
        $models = $this->find_models_by_("id={$id}") ;
        switch (count($models)) {
            case 0: return null ;
            case 1: return $models[0] ;
        }
        throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt.") ;
    }
    public function find_models_by_ ($condition) {
        $list = array () ;
        $result = $this->query("SELECT * FROM {$this->database}.dict_model WHERE {$condition}") ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC) ;
            array_push (
                $list,
                new IrepModel (
                    $this->find_manufacturer_by_id($attr['manufacturer_id']) ,
                    $attr)) ;
        }
        return $list ;
    }
    public function delete_model_by_id ($id) {
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->database}.dict_model WHERE id={$id}" ;
        $this->query($sql) ;
    }

    /* ---------------------
     *   Locations & rooms
     * ---------------------
     */
    public function locations () {
        $list = array () ;
        $result = $this->query("SELECT * FROM {$this->database}.dict_location ORDER BY name") ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepLocation (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function add_location ($name) {
        $name_escaped = $this->escape_string(trim($name)) ;
        $created_time = LusiTime::now()->to64() ;
        $created_uid_escaped = $this->escape_string(trim(AuthDB::instance()->authName())) ;
        $sql = "INSERT INTO {$this->database}.dict_location VALUES(NULL,'{$name_escaped}',{$created_time},'{$created_uid_escaped}')" ;
        $this->query($sql) ;
        return $this->find_location_by_('id=(SELECT LAST_INSERT_ID())') ;
    }
    public function find_location_by_name ($name) {
        $name_escaped = $this->escape_string(trim($name)) ;
        return $this->find_location_by_("name='{$name_escaped}'") ;
    }
    public function find_location_by_id ($id) {
        $id = intval($id) ;
        return $this->find_location_by_("id={$id}") ;
    }
    private function find_location_by_ ($condition='') {
        $conditions_opt = $condition ? " WHERE {$condition}" : '' ;
        $sql = "SELECT * FROM {$this->database}.dict_location {$conditions_opt}" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepLocation (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }
    public function delete_location ($id) {
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->database}.dict_location WHERE id={$id}" ;
        $this->query($sql) ;
    }
    public function find_room_by_id ($id) {
        $id = intval($id) ;
        return $this->find_room_by_("id={$id}") ;
    }
    private function find_room_by_ ($condition='') {
        $conditions_opt = $condition ? " WHERE {$condition}" : '' ;
        $sql = "SELECT * FROM {$this->database}.dict_room {$conditions_opt}" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC) ;
        return new IrepRoom (
            $this->find_location_by_id($attr['location_id']) ,
            $attr) ;
    }
    public function delete_room ($id) {
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->database}.dict_room WHERE id={$id}" ;
        $this->query($sql) ;
    }

    /* -----------------------------
     *   Statuses and sub-statuses
     * -----------------------------
     */
    public function statuses () {
        $list = array () ;
        $result = $this->query("SELECT * FROM {$this->database}.dict_status ORDER BY name") ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepStatus (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function add_status ($name) {
        $name_escaped = $this->escape_string(trim($name)) ;
        $created_time = LusiTime::now()->to64() ;
        $created_uid_escaped = $this->escape_string(trim(AuthDB::instance()->authName())) ;
        $sql = "INSERT INTO {$this->database}.dict_status VALUES(NULL,'{$name_escaped}','NO',{$created_time},'{$created_uid_escaped}')" ;
        $this->query($sql) ;
        return $this->find_status_by_('id=(SELECT LAST_INSERT_ID())') ;
    }
    public function find_status_by_name ($name) {
        $name_escaped = $this->escape_string(trim($name)) ;
        return $this->find_status_by_("name='{$name_escaped}'") ;
    }
    public function find_status_by_id ($id) {
        $id = intval($id) ;
        return $this->find_status_by_("id={$id}") ;
    }
    private function find_status_by_ ($condition='') {
        $conditions_opt = $condition ? " WHERE {$condition}" : '' ;
        $sql = "SELECT * FROM {$this->database}.dict_status {$conditions_opt}" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepStatus (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }
    public function delete_status_by_id ($id) {
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->database}.dict_status WHERE id={$id}" ;
        $this->query($sql) ;
    }
    public function find_status2_by_id ($id) {
        $id = intval($id) ;
        $sql = "SELECT * FROM {$this->database}.dict_status2 WHERE id={$id}" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (0 == $nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC) ;
        return new IrepStatus2 (
            $this->find_status_by_id($attr['status_id']) ,
            $attr) ;
    }
    public function delete_status2_by_id ($id) {
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->database}.dict_status2 WHERE id={$id}" ;
        $this->query($sql) ;
    }

    /* -------------
     *   Equipment
     * -------------
     */
    public function add_equipment ($manufacturer, $model, $serial, $description, $pc, $slacid, $location, $room, $rack, $elevation, $custodian, $parent=null) {

        // Step I: register new equipment to get its identifier. We will need the one
        //         later in order to associate it with the SLAC ID in the SLAC ID registry.

        $parent_id = is_null($parent) ? 'NULL' : $parent->id() ;
        $slacid = intval($slacid) ;
        if ($this->find_slacid($slacid))
            throw new IrepException (
                __METHOD__, "SLAC ID {$slacid} is in use for some other equipment") ;

        $status               = 'Unknown' ;
        $status2              = '' ;
        $manufacturer_escaped = $this->escape_string(trim($manufacturer)) ;
        $model_escaped        = $this->escape_string(trim($model)) ;
        $serial_escaped       = $this->escape_string(trim($serial)) ;
        $description_escaped  = $this->escape_string(trim($description)) ;
        $pc_escaped           = $this->escape_string(trim($pc)) ;
        $location_escaped     = $this->escape_string(trim($location)) ;
        $room_escaped         = $this->escape_string(trim($room)) ;
        $rack_escaped         = $this->escape_string(trim($rack)) ;
        $elevation_escaped    = $this->escape_string(trim($elevation)) ;
        $custodian_escaped    = $this->escape_string(trim($custodian)) ;
        $this->query(<<<HERE
INSERT INTO {$this->database}.equipment VALUES(
    NULL ,
    {$parent_id} ,
    '{$status}' ,
    '{$status2}' ,
    '{$manufacturer_escaped}' ,
    '{$model_escaped}' ,
    '{$serial_escaped}' ,
    '{$description_escaped}' ,
    {$slacid} ,
    '{$pc_escaped}' ,
    '{$location_escaped}' ,
    '{$room_escaped}' ,
    '{$rack_escaped}' ,
    '{$elevation_escaped}' ,
    '{$custodian_escaped}'
)
HERE
        ) ;

        // Step II: put the SLACid number into the registry

        $equipment = $this->find_equipment_by_('id=(SELECT LAST_INSERT_ID())') ;
        $this->allocate_slacid (
            $slacid ,
            $equipment->id() ,
            AuthDB::instance()->authName()
        ) ;
        $this->add_history_event (
            $equipment->id() ,
            'Registered' ,
            array('New equipment entry created in the database')
        ) ;
        return $equipment ;
    }
    public function delete_equipment ($id) {
        $id = intval($id) ;
        $sql = "DELETE FROM {$this->database}.equipment WHERE id={$id}" ;
        $this->query($sql) ;
    }
    public function find_equipment_by_id ($id) {
        $id = intval($id) ;
        return $this->find_equipment_by_("id={$id}") ;
    }
    public function find_equipment_by_pc ($pc) {
        $pc_escaped = $this->escape_string(trim($pc)) ;
        return $this->find_equipment_by_("pc='{$pc_escaped}'") ;
    }
    public function find_equipment_by_slacid($id) {
        $id = intval($id) ;
        return $this->find_equipment_by_("slacid={$id}") ;
    }
    public function find_equipment_by_slacid_range ($id) {
        $range = $this->find_slacid_range($id) ;
        if (!$range)
            throw new IrepException (
                __METHOD__, "no SLACid range exists for ID: {$id}") ;
        $first = $range['first'] ;
        $last  = $range['last'] ;
        return $this->find_equipment_many_by_("((slacid >= {$first}) AND (slacid <= {$last}))") ;
    }
    public function find_equipment_by_status_id ($id) {
        $status = $this->find_status_by_id($id) ;
        if (is_null($status))
            throw new IrepException (
                __METHOD__, "no status exists for ID: {$id}") ;
        return $this->find_equipment_many_by_("status='{$status->name()}'") ;
    }
    public function find_equipment_by_status2_id ($id) {
        $status2 = $this->find_status2_by_id($id) ;
        if (is_null($status2))
            throw new IrepException (
                __METHOD__, "no sub-status exists for ID: {$id}") ;
        return $this->find_equipment_many_by_("(status='{$status2->status()->name()}' AND status2='{$status2->name()}')") ;
    }
    public function search_equipment_by_many ($criteria) {
        $status       = array_key_exists('status',       $criteria) ? $criteria['status']       : '' ;
        $status2      = array_key_exists('status2',      $criteria) ? $criteria['status2']      : '' ;
        $manufacturer = array_key_exists('manufacturer', $criteria) ? $criteria['manufacturer'] : '' ;
        $model        = array_key_exists('model',        $criteria) ? $criteria['model']        : '' ;
        $serial       = array_key_exists('serial',       $criteria) ? $criteria['serial']       : '' ;
        $location     = array_key_exists('location',     $criteria) ? $criteria['location']     : '' ;
        $custodian    = array_key_exists('custodian',    $criteria) ? $criteria['custodian']    : '' ;
        $tag_name     = array_key_exists('tag_name',     $criteria) ? $criteria['tag_name']     : '' ;
        $description  = array_key_exists('description',  $criteria) ? $criteria['description']  : '' ;
        $notes        = array_key_exists('notes',        $criteria) ? $criteria['notes']        : '' ;
        return $this->search_equipment (
            $status ,
            $status2 ,
            $manufacturer ,
            $model ,
            $serial ,
            $location ,
            $custodian ,
            $tag_name ,
            $description ,
            $notes) ;
    }
    public function search_equipment (
        $status ,
        $status2 ,
        $manufacturer ,
        $model ,
        $serial ,
        $location ,
        $custodian ,
        $tag_name ,
        $description ,
        $notes) {

        // First search based on the properties of an equipment.

        $conditions_opt = '' ;
        $status = trim($status) ;
        if ($status != '') {
            $status_escaped = $this->escape_string($status) ;
            $conditions_opt .= ($conditions_opt == '' ? '' : ' AND ')." (status LIKE '%{$status_escaped}%') " ;
        }
        $status2 = trim($status2) ;
        if ($status2 != '') {
            $status2_escaped = $this->escape_string($status2) ;
            $conditions_opt .= ($conditions_opt == '' ? '' : ' AND ')." (status2 LIKE '%{$status2_escaped}%') " ;
        }
        $manufacturer = trim($manufacturer) ;
        if ($manufacturer != '') {
            $manufacturer_escaped = $this->escape_string($manufacturer) ;
            $conditions_opt .= ($conditions_opt == '' ? '' : ' AND ')." (manufacturer LIKE '%{$manufacturer_escaped}%') " ;
        }
        $model = trim($model) ;
        if ($model != '') {
            $model_escaped = $this->escape_string($model) ;
            $conditions_opt .= ($conditions_opt == '' ? '' : ' AND ')." (model LIKE '%{$model_escaped}%') " ;
        }
        $serial = trim($serial) ;
        if ($serial != '') {
            $serial_escaped = $this->escape_string($serial) ;
            $conditions_opt .= ($conditions_opt == '' ? '' : ' AND ')." (serial LIKE '%{$serial_escaped}%') " ;
        }
        $location = trim($location) ;
        if ($location != '') {
            $location_escaped = $this->escape_string($location) ;
            $conditions_opt .= ($conditions_opt == '' ? '' : ' AND ')." (location LIKE '%{$location_escaped}%') " ;
        }
        $custodian = trim($custodian) ;
        if ($custodian != '') {
            $custodian_escaped = $this->escape_string($custodian) ;
            $conditions_opt .= ($conditions_opt == '' ? '' : ' AND ')." (custodian LIKE '%{$custodian_escaped}%') " ;
        }
        $notes = trim($notes) ;
        if ($notes != '') {
            $notes_escaped = $this->escape_string($notes) ;
            $conditions_opt .= ($conditions_opt == '' ? '' : ' AND ')." (description LIKE '%{$notes_escaped}%') " ;
        }        
        $result_prev = $conditions_opt == '' ? array() : $this->find_equipment_many_by_($conditions_opt) ;

        // Apply the model description filter (if any)

        $description = trim($description) ;
        if ($description != '') {

            $result = array() ;

            // Find all models which match the specified description requirement
            // and put them into a 2-level dictionary.

            $description_escaped = $this->escape_string($description) ;
            $manuf_model = array() ;
            foreach ($this->find_models_by_("description LIKE '%{$description_escaped}%'") as $idx => $m) {

                $model_name = $m->name() ;
                $manuf_name = $m->manufacturer()->name() ;
                if (!array_key_exists($manuf_name, $manuf_model))              $manuf_model[$manuf_name]              = array() ;
                if (!array_key_exists($model_name, $manuf_model[$manuf_name])) $manuf_model[$manuf_name][$model_name] = true ;
            }
            if (!empty($manuf_model)) {

                // Now we have two scenarios: if theere we no specific conditiosn for the previous
                // search then we should make our search puerly based on the model description
                // pattern. Otherwise (the second scenario) we would have to merge results of
                // the previous search against the models matching the speified description.

                if ($conditions_opt == '') {
                    
                    // Scenario I: make the search based on the manufactures & models

                    foreach ($manuf_model as $manuf_name => $models) {
                        $manuf_name_escaped = $this->escape_string($manuf_name) ;
                        foreach ($models as $model_name => $val) {
                            $model_name_escaped = $this->escape_string($model_name) ;
                            foreach ($this->find_equipment_many_by_("(manufacturer='{$manuf_name_escaped}' AND model='{$model_name_escaped}')" ) as $equip) {
                                array_push($result, $equip) ;
                            }
                        }
                    }

                } else {

                    // Scenario II: filter results of the previous search

                    foreach ($result_prev as $idx => $equip) {
                        $manuf_name = $equip->manufacturer() ;
                        $model_name = $equip->model() ;
                        if (!array_key_exists($manuf_name, $manuf_model))             continue ;
                        if (!array_key_exists($model_name, $manuf_model[$manuf_name])) continue ;
                        array_push($result, $equip) ;
                    }
                }
            }
            $result_prev = $result ;
        }

        // Apply the tag filter (if any)

        $tag_name = trim($tag_name) ;
        if ($tag_name != '') {

            $result = array() ;

            $tags = $this->find_equipment_tag_by_name($tag_name) ;

            if (!empty($tags)) {

                // Now we have two scenarios: if theere we no specific conditiosn for the previous
                // search then we should make our search puerly based on the equipment items matched
                // the tag. Otherwise (the second scenario) we would have to merge results of
                // the previous search against the equipment identifiers matching the speified tag.

                if (($conditions_opt == '') && ($description == '')) {
                    
                    // Scenario I: make the search based on the tags only

                    foreach ($tags as $tag) {
                        array_push($result, $tag->equipment()) ;
                    }

                } else {

                    // Scenario II: filter results of the previous search
                    //
                    // TODO: The algorithm is highly inefficient due to O(x**2). Look for possible
                    //       ways to optimize it.

                    foreach ($result_prev as $equip) {
                        foreach ($tags as $tag) {
                            if ($equip->id() == $tag->equipment()->id()) {
                                array_push($result, $equip) ;
                            }
                        }
                    }
                }
            }
            $result_prev = $result ;
        }
        return $result_prev ;
    }
    public function search_equipment_by_status ($status, $status2=null) {
        $status_escaped = $this->escape_string(trim($status)) ;
        if ($status_escaped == '')
            throw new IrepException (
                __METHOD__, "the status parameter can't be empty") ;
        $conditions_opt = "status='{$status_escaped}'" ;
        if (!is_null($status2)) {
            $status2_escaped = $this->escape_string(trim($status2)) ;
            $conditions_opt .= " AND status2='{$status2_escaped}'" ;
        }
        return $this->find_equipment_many_by_($conditions_opt) ;
    }

    public function find_equipment_by_any ($text2search) {
        $text2search_escaped = $this->escape_string(trim($text2search)) ;
        if ($text2search_escaped == '')
            throw new IrepException (
                __METHOD__, "the <text2search> parameter can't be empty") ;

        // Make two separate searches and merge results into a single list
        // which won't have any duplicates (based on equipment identifiers).

        $result = array() ;

        $partial_result = array (
            $this->search_equipment_by_many(array('description' => $text2search)) ,
            $this->search_equipment_by_many(array('notes'       => $text2search))
        ) ;
        
        $ids = array() ;
        for ($i = 0; $i < count($partial_result); $i++) {
            foreach ($partial_result[$i] as $e) {
                if (array_key_exists($e->id(), $ids)) continue ;
                $ids[$id] = $id ;
                array_push ($result, $e) ;
            }
        }
        return $result ;
    }
    
    private function find_equipment_by_ ($conditions_opt='') {
        $result = $this->query("SELECT * FROM {$this->database}.equipment ".($conditions_opt == '' ? '' : " WHERE {$conditions_opt}")) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepEquipment (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }
    public function find_equipment_many_by_ ($conditions_opt='') {
        $list = array () ;
        $sql = "SELECT * FROM {$this->database}.equipment ".($conditions_opt == '' ? '' : " WHERE {$conditions_opt}") ;
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++)
            array_push (
                $list,
                new IrepEquipment (
                    $this ,
                    mysql_fetch_array( $result, MYSQL_ASSOC))) ;
        return $list ;
    }
    public function update_equipment ($id, $properties2update, $comment) {
        $equipment = $this->find_equipment_by_id($id) ;
        if (is_null($equipment))
            throw new IrepException (
                __METHOD__, "unknown equipment identifier: {$id}") ;
        $comments = array () ;
        if ($comment != '') array_push($comments, "User comments: {$comment}") ;
        foreach ($properties2update as $property => $new_value) {
            $old_value = $equipment->property($property) ;
            array_push($comments, "{$property}: {$old_value} -> {$new_value}") ;
        }
        $equipment->update($properties2update) ;
        $this->add_history_event($id, 'Modified', $comments) ;
        return $this->find_equipment_by_id($id) ;
    }
    public function add_history_event ($equipment_id, $event_text, $comments=array()) {
        $equipment_id  = intval($equipment_id) ;
        $event_time_64 = LusiTime::now()->to64() ;
        $event_uid     = $this->escape_string(trim(AuthDB::instance()->authName())) ;
        $event_text_escaped = $this->escape_string(trim($event_text)) ;
        $this->query("INSERT INTO {$this->database}.equipment_history VALUES(NULL,{$equipment_id},{$event_time_64},'{$event_uid}','{$event_text_escaped}')") ;
        $event = $this->find_history_event_by_('id=(SELECT LAST_INSERT_ID())') ;
        foreach ($comments as $comment) {
            $comment_escaped = $this->escape_string(trim($comment)) ;
            $this->query("INSERT INTO {$this->database}.equipment_history_comments VALUES({$event->id()},'{$comment_escaped}')") ;
        }
        return $event ;
    }
    public function last_history_event ($equipment_id) {
        $equipment_id  = intval($equipment_id) ;
        return $this->find_history_event_by_("(equipment_id={$equipment_id})", "ORDER BY event_time DESC LIMIT 1") ;
    }
    public function find_history_event_by_ ($conditions_opt='', $sort_opt='') {
        $sql = "SELECT * FROM {$this->database}.equipment_history ".($conditions_opt == '' ? '' : " WHERE {$conditions_opt} {$sort_opt}") ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows ($result) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        return new IrepEquipmentHistoryEvent (
            $this ,
            mysql_fetch_array( $result, MYSQL_ASSOC)) ;
    }

    /* ---------------
     *   Attachments
     * ---------------
     */
    public function find_equipment_attachment_by_id ($id) {
        $id = intval($id) ;
        $sql = "SELECT * FROM {$this->database}.equipment_attachment WHERE id={$id}" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC) ;
        return new IrepEquipmentAttachment (
            $this->find_equipment_by_id($attr['equipment_id']) ,
            $attr) ;
    }

    public function find_model_attachment_by_id ($id) {
        $id = intval($id) ;
        $sql = "SELECT * FROM {$this->database}.dict_model_attachment WHERE id={$id}" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows( $result ) ;
        if (!$nrows) return null ;
        if (1 != $nrows)
            throw new IrepException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}") ;
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC) ;
        return new IrepModelAttachment (
            $this->find_model_by_id($attr['model_id']) ,
            $attr) ;
    }

    /* --------
     *   Tags
     * --------
     */
    public function known_equipment_tags () {
        $list = array () ;
        $sql = "SELECT DISTINCT name FROM {$this->database}.equipment_tag ORDER BY name" ;
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC) ;
            array_push (
                $list,
                $attr['name']) ;
        }
        return $list ;
    }
    public function find_equipment_tag_by_name ($name) {
        $list = array () ;
        $name_escaped = $this->escape_string(trim($name)) ;
        $sql = "SELECT * FROM {$this->database}.equipment_tag WHERE name='{$name_escaped}' ORDER BY equipment_id" ;
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC) ;
            array_push (
                $list,
                new IrepEquipmentTag (
                    $this->find_equipment_by_id($attr['equipment_id']) ,
                    $attr)) ;
        }
        return $list ;
    }
}

?>
