<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'regdb/regdb.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \AuthDB\AuthDB ;
use \FileMgr\DbConnection ;
use \LusiTime\LusiTime ;
use \RegDB\RegDB ;

/**
 * Class ShiftMgr encapsulates operations with the PCDS Shift Management database
 *
 * @author gapon
 */
class ShiftMgr extends DbConnection {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $instance = null ;

    /**
     * Singleton to simplify certain operations.
     *
     * @return ShiftMgr
     */
    public static function instance () {
        if (is_null(ShiftMgr::$instance)) ShiftMgr::$instance =
            new ShiftMgr (
                SHIFTMGR_DEFAULT_HOST,
                SHIFTMGR_DEFAULT_USER,
                SHIFTMGR_DEFAULT_PASSWORD,
                SHIFTMGR_DEFAULT_DATABASE) ;
        return ShiftMgr::$instance ;
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

    /**
     * Return True if the currently logged user is allowed to manage shifts for the instrument.
     *
     * @param string $instrument - an instrument name
     * @return boolean
     */
    public function is_manager ($instrument_name) {

        $this->assert_instrument($instrument_name) ;

        $instrument_name = strtoupper(trim($instrument_name)) ;

        AuthDb::instance()->begin() ;
        return AuthDB::instance()->hasRole (
            AuthDB::instance()->authName() ,    // UID of the currennt logged user
            null ,                              // accross all experiments of the instrument
            "ShiftMgr" ,                        // the application name 
            "Manage_{$instrument_name}"         // the role name encodes the instrument name
        ) ;
    }

    /**
     * Return a list of shifts for all or the specified instrument.
     * 
     * Shifts are represented as objects of class \ShiftMgr\ShiftMgrShift.
     *
     * @param string $instrument - an optional instrument name
     * @return array - shifts found in the database
     */
    public function shifts ($instrument_name='') {

        $instrument_option = '' ;
        if ($instrument_name) {
            $this->assert_instrument($instrument_name) ;
            $instrument_option = " WHERE instrument='".$this->escape_string(strtoupper(trim($instrument_name)))."'" ;
        }
        $sql = "SELECT * FROM {$this->database}.shift {$instrument_option} ORDER BY instrument, id DESC, begin_time DESC" ;

        $shifts = array () ;
        $result = $this->query($sql) ;

        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            array_push (
                $shifts ,
                new ShiftMgrShift ($this, mysql_fetch_array ($result, MYSQL_ASSOC))
            ) ;
        }
        return $shifts ;
    }

    /**
     * Return the lastest shift of the specified instrument or null if no such shift exists.
     *
     * @param string $instrument - the instrument name
     * @return null|\ShiftMgr\ShiftMgrShift - the last shift
     * @throws \ShiftMgr\ShiftMgrException
     */
    public function last_shift ($instrument_name) {

        $this->assert_instrument($instrument_name) ;

        $instrument_escaped = $this->escape_string(strtoupper(trim($instrument_name))) ;

        // This statement will find all shift of the specified instrument
        // and leave the one with the highest begin time if any exists.

        $sql = "SELECT * FROM {$this->database}.shift WHERE instrument='{$instrument_escaped}' ORDER BY id DESC, begin_time DESC LIMIT 1" ;

        $result = $this->query($sql) ;
        $nrows = mysql_numrows($result) ;
        switch ($nrows) {
            case 0 : return null ;
            case 1 : return new ShiftMgrShift ($this, mysql_fetch_array ($result, MYSQL_ASSOC)) ;
        }
        throw new ShiftMgrException (
            __class__.'::'.__METHOD__ ,
            'inconsistent result returned from the database. Wrong schema?'
        ) ;
    }

    /**
     * Begin the new shift at the specified instrument.
     *
     * Note, that this function will also close the last shift (if any) if it's
     * still open. The shift will be closed where the new one will begin.
     *
     * @param string $instrument
     * @return ShiftMgr\ShiftMgrShift
     */
    public function new_shift ($instrument_name) {

        $this->assert_instrument($instrument_name) ;

        $begin_time_sec = LusiTime::now()->sec ;

        $last_shift = $this->last_shift($instrument_name) ;
        if (!is_null($last_shift) && is_null($last_shift->end_time())) $last_shift->close() ;

        $instrument_name_escaped = $this->escape_string(strtoupper(trim($instrument_name))) ;
        $sql = "INSERT INTO {$this->database}.shift VALUES (NULL,'{$instrument_name_escaped}',{$begin_time_sec},NULL)" ;

        $this->query($sql) ;
        
        return $this->last_shift($instrument_name) ;
    }

    /**
     * Throw an exception if the specified name doesn't correspond to any known instrument.
     *
     * @param string $name - instrument name
     * @throws ShiftMgrException
     */
    private function assert_instrument ($name) {
        
        // The implementation of the method will check if the specified
        // name matches any known instrument in the Experiment Registry Database.
        // This will may require to begin a separate transaction.

        RegDB::instance()->begin() ;
        if (is_null(RegDB::instance()->find_instrument_by_name($name)))
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                "no such instrument found in the Experiment Registry Database: '{$name}'") ;
    }
}

?>
