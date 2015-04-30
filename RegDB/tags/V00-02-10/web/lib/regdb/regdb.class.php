<?php

namespace RegDB;

require_once( 'regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

class RegDB {

    private static $instance = null;

    /**
     * Return an instance of the object initialzied with default version
     * of parameters.
     */
    public static function instance() {
        if( is_null(RegDB::$instance)) RegDB::$instance =
            new RegDB(
                REGDB_DEFAULT_HOST,
                REGDB_DEFAULT_USER,
                REGDB_DEFAULT_PASSWORD,
                REGDB_DEFAULT_DATABASE,
                REGDB_DEFAULT_LDAP_HOST,
                REGDB_DEFAULT_LDAP_USER,
                REGDB_DEFAULT_LDAP_PASSWD
            );
        return RegDB::$instance;
    }

    /* Data members
     */
    private $connection;

    /* Constructor
     *
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     */
    public function __construct ($host, $user, $password, $database, $ldap_host, $ldap_user, $ldap_passwd) {

        $this->connection =
            new RegDBConnection (
                $host,
                $user,
                $password,
                $database,
                $ldap_host,
                $ldap_user,
                $ldap_passwd );
        }

    /*
     * ==========================
     *   TRANSACTION MANAGEMENT
     * ==========================
     */
    public function begin () {
        $this->connection->begin (); }

    public function commit () {
        $this->connection->commit (); }

    public function rollback () {
        $this->connection->rollback (); }

    /* ===============
     *   EXPERIMENTS
     * ===============
     */
    public function experiment_names () {

        $list = array();

        $result = $this->connection->query (
            "SELECT name FROM {$this->connection->database}.experiment ORDER BY name" );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                mysql_result( $result, $i ));

        return $list;
    }

    public function experiments () {
        return $this->find_experiments_by_(); }

    public function experiments_for_instrument ( $instrument_name ) {

        $instrument = $this->find_instrument_by_name( $instrument_name );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "no such instrument in the database" );

        return $this->find_experiments_by_ ( 'instr_id='.$instrument->id() );
    }

    public function find_experiment ( $instrument_name, $experiment_name ) {

        $instrument = $this->find_instrument_by_name( $instrument_name );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "no such instrument in the database" );

        $experiments = $this->find_experiments_by_ ( 'instr_id='.$instrument->id()." AND name='{$experiment_name}'" );
        if( count( $experiments ) == 0 ) return null;
        if( count( $experiments ) == 1 ) return $experiments[0];
        throw new RegDBException (
            __METHOD__,
            "inconsistent results returned fromthe database. The database may be corrupted." );
    }

    public function find_experiment_by_unique_name ( $experiment_name ) {

        $experiments = $this->find_experiments_by_ ( "name='{$experiment_name}'" );
        if( count( $experiments ) == 0 ) return null;
        if( count( $experiments ) == 1 ) return $experiments[0];
        throw new RegDBException (
            __METHOD__,
            "too many experiments found in the database. The experiment name is not unique." );
    }

    private function find_experiments_by_ ( $condition='' ) {

        $list = array();

        $extra_condition = $condition == '' ? '' : ' WHERE '.$condition;
        $result = $this->connection->query (
            "SELECT * FROM {$this->connection->database}.experiment ".$extra_condition. ' ORDER by begin_time DESC' );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                new RegDBExperiment (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function find_experiment_by_id ( $id ) {
        return $this->find_experiment_by_( 'id='.$id) ; }

    public function find_experiment_by_name ( $name ) {
        return $this->find_experiment_by_( "name='".$name."'") ; }

    public function find_last_experiment () {
        return $this->find_experiment_by_( 'begin_time=(SELECT MAX(begin_time) FROM experiment)') ; }

    private function find_experiment_by_ ( $condition ) {

        $result = $this->connection->query(
            "SELECT * FROM {$this->connection->database}.experiment WHERE ".$condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new RegDBExperiment(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function register_experiment (
        $experiment_name, $instrument_name, $description,
        $registration_time, $begin_time, $end_time,
        $posix_gid, $leader, $contact ) {

        /* Verify parameters
         */
        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new RegDBException (
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        $trim_experiment_name = trim( $experiment_name );
        if( $trim_experiment_name == '' )
            throw new RegDBException (
                __METHOD__,
                "the experiment name can't be empty string" );

        /* Find instrument.
         */
        $instrument = $this->find_instrument_by_name( $instrument_name );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "no such instrument" );

        /* Make sure the group exists and the group leader is among its
         * members.
         */
        $trim_leader = trim( $leader );
        if( $trim_leader == '' )
            throw new RegDBException (
                __METHOD__,
                "the leader name can't be empty string" );

        $trim_posix_gid = trim( $posix_gid );
        if( $trim_posix_gid == '' )
            throw new RegDBException (
                __METHOD__,
                "the POSIX group name can't be empty string" );
/*
 * TODO: Disable this temporarily untill a more sophisticated algorithm for
 * browsing LDAP associations (groups/users) is implemented.
 *
        if( !$this->is_member_of_posix_group ( $trim_posix_gid, $trim_leader ))
            throw new RegDBException (
                __METHOD__,
                "the proposed leader isn't a member of the POSIX group" );
*/
         /* Proceed with the operation.
          */
        $this->connection->query (
            "INSERT INTO {$this->connection->database}.experiment VALUES(NULL,'".$this->connection->escape_string( $trim_experiment_name ).
            "','".$this->connection->escape_string( $description ).
            "',".$instrument->id().
            ",".$registration_time->to64().
            ",".$begin_time->to64().
            ",".$end_time->to64().
            ",'".$this->connection->escape_string( $trim_leader ).
            "','".$this->connection->escape_string( trim( $contact )).
            "','".$this->connection->escape_string( $trim_posix_gid )."')" );

        $experiment = $this->find_experiment_by_id( '(SELECT LAST_INSERT_ID())' );
        if( !$experiment )
            throw new RegDBException (
                __METHOD__,
                "fatal internal error" );

        /* Create the run numbers generator
         */
        $this->connection->query (
            "CREATE TABLE {$this->connection->database}.run_{$experiment->id()} ".
            "(num int(11) NOT NULL auto_increment, ".
            " request_time bigint(20) unsigned NOT NULL,".
            " PRIMARY KEY (num)".
            ")" );

        return $experiment;
    }

    public function delete_experiment_by_id ( $id ) {

        $experiment = $this->find_experiment_by_id( $id );
        if( !$experiment )
            throw new RegDBException (
                __METHOD__,
                "no such experiment" );

        /* Proceed with the operation.
         */
        $this->connection->query ( "DELETE FROM {$this->connection->database}.experiment_param WHERE exper_id=".$id );
        $this->connection->query ( "DELETE FROM {$this->connection->database}.experiment WHERE id=".$id );
        $this->connection->query ( "DROP TABLE IF EXISTS {$this->connection->database}.run_".$id );
    }

    /* ===============
     *   INSTRUMENTS
     * ===============
     */
    public function instrument_names () {

        $list = array();

        $result = $this->connection->query (
            "SELECT name FROM {$this->connection->database}.instrument " );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                mysql_result( $result, $i ));

        return $list;
    }

    public function instruments ( $condition='' ) {

        $list = array();

        $result = $this->connection->query (
            "SELECT * FROM {$this->connection->database}.instrument ".$condition );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                new RegDBInstrument (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function find_instrument_by_id ( $id ) {
        return $this->find_instrument_by_( 'id='.$id) ; }

    public function find_instrument_by_name ( $name ) {
        return $this->find_instrument_by_( "name='".$name."'") ; }

    private function find_instrument_by_ ( $condition ) {

        $result = $this->connection->query(
            "SELECT * FROM {$this->connection->database}.instrument WHERE ".$condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 )
            return new RegDBInstrument (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        throw new RegDBException(
            __METHOD__,
            "unexpected size of result set returned by the query" );
    }

    public function register_instrument ( $instrument_name, $description ) {

        /* Verify parameters
         */
        $trim_instrument_name = trim( $instrument_name );
        if( $trim_instrument_name == '' )
            throw new RegDBException (
                __METHOD__,
                "the instrument name can't be empty string" );

         /* Proceed with the operation.
          */
        $this->connection->query (
            "INSERT INTO {$this->connection->database}.instrument VALUES(NULL,'".$this->connection->escape_string( $trim_instrument_name ).
            "','".$this->connection->escape_string( $description )."')" );

        $instrument = $this->find_instrument_by_id( '(SELECT LAST_INSERT_ID())' );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "fatal internal error" );
        return $instrument;
    }

    public function delete_instrument_by_id ( $id ) {

        $instrument = $this->find_instrument_by_id( $id );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "no such instrument" );

        /* Find and delete the connected experiments first.
         */
        $experiments = $this->experiments_for_instrument( $instrument->name());
        foreach( $experiments as $e )
            $this->delete_experiment_by_id( $e->id());

        /* Proceed to the instrument.
         */
        $this->connection->query ( "DELETE FROM {$this->connection->database}.instrument_param WHERE instr_id=".$id );
        $this->connection->query ( "DELETE FROM {$this->connection->database}.instrument WHERE id=".$id );
    }

    /* =====================
     *   EXPERIMENT SWITCH
     * =====================
     */
    public function switch_experiment( $experiment_name, $station, $requestor_uid, $notifications=null ) {

        $experiment = $this->find_experiment_by_unique_name( $experiment_name );
        if( !$experiment )
            throw new RegDBException (
                __METHOD__,
                "no such experiment: {$experiment_name}" );

        $num_stations = $experiment->instrument()->find_param_by_name( 'num_stations' );
        if( is_null($num_stations))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not properly configured in the database, experiment: {$experiment_name}" );

        if( $station >= intval($num_stations->value()))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not configured to take data, experiment: {$experiment_name}" );

        $current_time = LusiTime::now();

        $this->connection->query (
            "INSERT INTO expswitch VALUES (NULL,{$experiment->id()},{$station},{$current_time->to64()},'{$requestor_uid}')"
        );
        if( !is_null($notifications) && count( $notifications ) > 0 ) {
            $sql = '';
            foreach( $notifications as $n ) {
                $uid = trim( $n['uid'] );
                $gecos = $this->connection->escape_string( trim( $n['gecos'] ));
                $email = $this->connection->escape_string( trim( $n['email'] ));
                $rank = trim( $n['rank'] );
                $notified = $n['notified'] ? 'YES' : 'NO';
                if( $sql != '' ) $sql .= ',';
                $sql .=  "(LAST_INSERT_ID(),'{$uid}','{$gecos}','{$email}','{$rank}','{$notified}')";
            }
            $sql = 'INSERT INTO expswitch_notify VALUES '.$sql;
            $this->connection->query( $sql );
        }
    }

    /* Find the info for the last experiment switched using this interface. Return null
     * if none was found.
     */
    public function last_experiment_switch( $instrument_name, $station=0 ) {

        $instrument = $this->find_instrument_by_name($instrument_name);

        $num_stations = $instrument->find_param_by_name( 'num_stations' );
        if( is_null($num_stations))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not properly configured in the database, instrument: {$instrument_name}" );

        if( $station >= intval($num_stations->value()))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not configured to take data, instrument: {$instrument_name}" );

        $sql =<<<HERE
SELECT * FROM {$this->connection->database}.expswitch
WHERE exper_id IN ( SELECT e.id FROM {$this->connection->database}.experiment `e`,
                                     {$this->connection->database}.instrument `i`
                    WHERE e.instr_id=i.id
                    AND i.name='{$instrument_name}' )
AND station={$station}
ORDER BY switch_time DESC LIMIT 1
HERE;
        $result = $this->connection->query( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            return $attr;
        }
        throw new RegDBException(
            __METHOD__,
            "unexpected size of result set returned by the query" );
    }

    /**
     * 
     * @param String $instrument_name
     * @param Number $station
     * @param LusiTime $before
     * @return Array
     * @throws RegDBException
     */
    public function last_experiment_switch_before( $instrument_name, $station, $before ) {

        $instrument = $this->find_instrument_by_name($instrument_name);

        $num_stations = $instrument->find_param_by_name( 'num_stations' );
        if( is_null($num_stations))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not properly configured in the database, instrument: {$instrument_name}" );

        if( $station >= intval($num_stations->value()))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not configured to take data, instrument: {$instrument_name}" );

        $sql =<<<HERE
SELECT * FROM {$this->connection->database}.expswitch
WHERE exper_id IN ( SELECT e.id FROM {$this->connection->database}.experiment `e`,
                                     {$this->connection->database}.instrument `i`
                    WHERE e.instr_id=i.id
                    AND i.name='{$instrument_name}' )
AND station={$station}
AND switch_time < {$before->to64()}
ORDER BY switch_time DESC LIMIT 1
HERE;
        $result = $this->connection->query( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            return $attr;
        }
        throw new RegDBException(
            __METHOD__,
            "unexpected size of result set returned by the query" );
    }

    /**
     * Return true if the specified experiment is active (current) at any DAQ station
     * of the corresponding instrument.
     *
     * @param integer $exper_id
     * @return boolean
     * @throws RegDBException
     */
    public function is_active_experiment($exper_id) {
        $experiment = $this->find_experiment_by_id($exper_id);
        if( !$experiment )
            throw new RegDBException (
                __METHOD__,
                "no such experiment ID: {$exper_id}" );

        $num_stations = $experiment->instrument()->find_param_by_name( 'num_stations' );
        if( is_null($num_stations))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not properly configured in the database, experiment ID: {$exper_id}" );

        if( !intval($num_stations->value()))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not configured to take data, experiment ID: {$exper_id}" );
                
        for( $station = 0; $station < intval($num_stations->value()); $station++ ) {
            $last_switch = $this->last_experiment_switch( $experiment->instrument()->name(), $station );
            if( !is_null($last_switch) && ($exper_id == $last_switch['exper_id'])) return true;
        }
        return false;
    }

    /* Return a history for all known experiment switches for the specified instrument.
     */
    public function experiment_switches( $instrument_name ) {

        $sql =<<<HERE
SELECT * FROM {$this->connection->database}.expswitch
WHERE exper_id IN ( SELECT e.id FROM {$this->connection->database}.experiment `e`,
                                     {$this->connection->database}.instrument `i`
                    WHERE e.instr_id=i.id
                    AND i.name='{$instrument_name}' )
ORDER BY switch_time DESC
HERE;
        $result = $this->connection->query( $sql );
        $nrows = mysql_numrows( $result );
        $list = array();
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                mysql_fetch_array( $result, MYSQL_ASSOC ));
        }
        return $list;
    }

    /* =========
     *   FILES
     * =========
     */

    /**
     * Return an iterator of files reported by the DAQ system as "open".
     * 
     * @param string $instr_name
     * @param \LusiTime\LusiTime $begin_time
     * @param \LusiTime\LusiTime $end_time
     * @param boelean $reverse_order
     * @param boelean $order_by_time
     * @return \RegDB\RegDBFileItr
     */
    public function files_itr (
        $instr_name    = null ,
        $begin_time    = null ,
        $end_time      = null ,
        $reverse_order = true ,
        $order_by_time = true) {

        $select_opt = '' ;
        if (!is_null($instr_name) ||
            !is_null($begin_time) ||
            !is_null($end_time)) {
  
            if (!is_null($instr_name)) {
                $instr = $this->find_instrument_by_name($instr_name) ;
                if (is_null($instr))
                    throw new RegDBException (
                        __METHOD__ ,
                        "no such instrument: '{$instr_name}'") ;
                $sql_subquery = "SELECT id FROM {$this->connection->database}.experiment WHERE instr_id={$instr->id()}" ;
                $select_opt .= ($select_opt == '' ? ' WHERE' : ' AND')." exper_id IN ({$sql_subquery})" ;
            }
            if (!is_null($begin_time)) $select_opt .= ($select_opt == '' ? ' WHERE' : ' AND')." open >= {$begin_time->to64()}" ;
            if (!is_null($end_time))   $select_opt .= ($select_opt == '' ? ' WHERE' : ' AND')." open <  {$end_time->to64()}" ;
        }
        $order = $reverse_order ? 'DESC' : '' ;
        $order_by_opt = $order_by_time ?
            "open {$order}" :
            "run {$order}, stream {$order}, chunk {$order}" ;

        $sql = "SELECT * FROM {$this->connection->database}.file {$select_opt} ORDER BY {$order_by_opt}" ;
                        
        return new RegDBFileItr($this, $this->connection, $sql) ;
    }

    /* ====================
     *   GROUPS AND USERS
     * ====================
     */
    public function posix_groups ( $all_groups=true ) {
        return $this->connection->posix_groups( $all_groups ); }

    public function is_known_posix_group ( $name ) {
        return $this->connection->is_known_posix_group( $name ); }

    public function is_member_of_posix_group ( $group, $uid ) {
        return $this->connection->is_member_of_posix_group( $group, $uid ); }

    public function posix_group_members ( $name, $and_as_primary_group=true ) {
        return $this->connection->posix_group_members( $name, $and_as_primary_group ); }

    public function user_accounts ( $user='*' ) {
        return $this->connection->user_accounts( $user ); }

    public function find_user_accounts ( $uid_or_gecos_pattern, $scope ) {
        return $this->connection->find_user_accounts( $uid_or_gecos_pattern, $scope ); }

    public function find_user_account ( $user ) {
        return $this->connection->find_user_account( $user ); }

    public function add_user_to_posix_group ( $user_name, $group_name ) {
        $this->connection->add_user_to_posix_group( $user_name, $group_name ); }

    public function remove_user_from_posix_group ( $user_name, $group_name ) {
        $this->connection->remove_user_from_posix_group( $user_name, $group_name ); }

    /* Return an associative array of experiment groups whose names
     * follow the pattern:
     * 
     *   iiipppyy
     *
     * Where:
     *
     *   'iii' - is TLA for an instrument name
     *   'ppp' - proposal number for a year when the experiment is conducted
     *   'yy'  - last two digits for the year of the experiemnt
     *
     * Non standard group names:
     *
     *   In addition to the above explained rule the method can be modified
     *   to retuns non-standard group names. See details in th eimplementation
     *   of the method. These group names should follow the following
     *   convention:
     *
     *     iiis..yy
     *
     * Where:
     *
     *   'iii' - is TLA for an instrument name
     *   's..' - instrument/experiment specific designation for an experiment
     *   'yy'  - last two digits for the year of the experiemnt
     *
     * Parameters:
     *
     *   'instr' - optional name of an instrument to narrow the search. If not
     *             present then all instruments will be assumed.
     */
    public function experiment_specific_groups( $instr=null ) {
        $groups = array();
        $instr_names = array();
        if( is_null( $instr )) $instr_names = $this->instrument_names();
        else array_push( $instr_names, $instr );
        foreach( $instr_names as $i ) {
            foreach( $this->experiments_for_instrument( $i ) as $exper ) {
                $g = $exper->name();
                if(( 1 == preg_match( '/^[a-z]{3}[0-9]{5}$/', $g )) ||
                   ( 1 == preg_match( '/^[a-z]{3}[a-z][0-9]{4}$/', $g )) ||
                   ( 1 == preg_match( '/^[a-z]{3}daq[0-9]{2}$/', $g )) ||
                   ( 1 == preg_match( '/^dia[a-z]{3}[0-9]{2}$/', $g ))) $groups[$g] = True;
            }
        }

        /* Add known POSIX groups for special experiments which aren't following
         * the standard naming convention:
         *
         *   <instr><proposal><year>
         */

        /* In-house commissionning, in-house, etc. for the year of 2010.
         */
        if( is_null( $instr ) || ( $instr == 'AMO' )) {
            $groups['ps-amo'] = True;
            $groups['ps-amo-sci'] = True;
            $groups['ps-amo-elog'] = True;
            $groups['amoopr'] = True;
        }

        /* SXR commissionning, in-house, etc. experiments for the year of 2010.
         */
        if( is_null( $instr ) || ( $instr == 'SXR' )) {
            $groups['sxrrsx10'] = True;
            $groups['sxrsse10'] = True;
            $groups['sxrlje10'] = True;
            $groups['ps-sxr'] = True;
            $groups['ps-sxr-sci'] = True;
            $groups['ps-sxr-elog'] = True;
            $groups['sxropr'] = True;
        }

        /* Groups for which there is no entry in RegDB but which we still want
         * to treat as experiment specific groups.
         */

        /* XPP commissionning, in-house, etc. experiments for the year of 2010.
         */
        if( is_null( $instr ) || ( $instr == 'XPP' )) {
            $groups['xpp80610'] = True;
            $groups['ps-xpp'] = True;
            $groups['ps-xpp-sci'] = True;
            $groups['ps-xpp-elog'] = True;
            $groups['xppopr'] = True;
            $groups['xppcom12'] = True;
            $groups['xppcom13'] = True;
            $groups['xpptst14'] = True;
        }
        /* CXI commissionning, in-house, etc. experiments for the year of 2010.
         */
        if( is_null( $instr ) || ( $instr == 'CXI' )) {
            $groups['ps-cxi'] = True;
            $groups['ps-cxi-sci'] = True;
            $groups['ps-cxi-elog'] = True;
            $groups['ps-cxi-geom'] = True;
            $groups['cxiopr'] = True;
        }
        /* MEC commissionning, in-house, etc. experiments for the year of 2010.
         */
        if( is_null( $instr ) || ( $instr == 'MEC' )) {
            $groups['ps-mec'] = True;
            $groups['ps-mec-sci'] = True;
            $groups['ps-mec-elog'] = True;
            $groups['mecopr'] = True;
        }

        /* XCS commissionning, in-house, etc. experiments for the year of 2010.
         */
        if( is_null( $instr ) || ( $instr == 'XCS' )) {
            $groups['ps-xcs'] = True;
            $groups['ps-xcs-sci'] = True;
            $groups['ps-xcs-elog'] = True;
            $groups['xcsopr'] = True;
            $groups['xcscom12'] = True;
        }

        /* Mobile rack experiments
         */
        if( is_null( $instr ) || ( $instr == 'MOB' )) {
            $groups['ps-mob'] = True;
        }
        
        /* External (user) experiments
         */
        if( is_null( $instr ) || ( $instr == 'USR' )) {
            $groups['ps-usr'] = True;
            $groups['ps-usr-sci'] = True;
        }

        /* Facility e-logs.
         */
        if( is_null( $instr ) || ( $instr == 'NEH' )) {
            $groups['ps-las'] = True;
        }

        /* Add groups which aren't really experiment or instrument specific.
         */
        //$groups['ps-data'] = True;

        return $groups;
    }
}

/* =======================
 * UNIT TEST FOR THE CLASS
 * =======================
 *

require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

try {
    RegDB::instance()->begin();

    $name              = "Exp-A";
    $instrument_name   = "CXI";
    $description       = "Experiment description goes here";
    $registration_time = LusiTime::now();
    $begin_time        = LusiTime::parse( "2009-07-01 09:00:01-0700" );
    $end_time          = LusiTime::parse( "2009-09-01 09:00:01-0700" );
    $posix_gid         = "ps-data";
    $leader            = "gapon";
    $contact           = "Phone: SLAC x5095, E-Mail: gapon@slac.stanford.edu";

    $experiment = RegDB::instance()->register_experiment (
        $name,
        $instrument_name,
        $description,
        $registration_time, 
        $begin_time,
        $end_time,
        $posix_gid,
        $leader,
        $contact );

    print_r( $experiment );

    RegDB::instance()->commit();

} catch ( RegDBException $e ) { print( $e->toHtml()); }

*/
?>
