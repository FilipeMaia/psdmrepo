<?php

namespace ShiftMgr;

require_once 'shiftmgr.inc.php';
require_once 'authdb/authdb.inc.php';
require_once 'regdb/regdb.inc.php';
require_once 'filemgr/filemgr.inc.php';
require_once 'lusitime/lusitime.inc.php';

use \AuthDB\AuthDB;
use \FileMgr\DbConnection;
use \LusiTime\LusiTime;
use \RegDB\RegDB;

/**
 * Class ShiftMgr encapsulates operations with the PCDS Shift Management database
 *
 * @author gapon
 */
class ShiftMgr extends DbConnection {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $instance = null;

    /**
     * Singleton to simplify certain operations.
     *
     * @return ShiftMgr
     */
    public static function instance() {
        if (is_null(ShiftMgr::$instance)) ShiftMgr::$instance =
            new ShiftMgr (
                SHIFTMGR_DEFAULT_HOST,
                SHIFTMGR_DEFAULT_USER,
                SHIFTMGR_DEFAULT_PASSWORD,
                SHIFTMGR_DEFAULT_DATABASE);
        return ShiftMgr::$instance;
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
    public function __construct($host, $user, $password, $database) {
        parent::__construct($host, $user, $password, $database);
    }

    /**
     * Throw an exception if the specified name doesn't correspond to any known hutch.
     *
     * @param string $name - hutch name
     * @throws ShiftMgrException
     */
    private function assert_hutch($name) {
        
        // The implementation of the method will check if the specified
        // name matches any known hutch in the Experiment Registry Database.
        // This will may require to begin a separate transaction.

        RegDB::instance()->begin();
        if (is_null(RegDB::instance()->find_instrument_by_name($name)))
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                "no such hutch found in the Experiment Registry Database: '{$name}'");
    }

    /**
     * Return True if the currently logged user is allowed to manage shifts for the hutch.
     *
     * @param string $hutch - a hutch name
     * @return boolean
     */
    public function is_manager($hutch) {
      return true;
      /*
        $this->assert_hutch($hutch);
        $hutch = strtoupper(trim($hutch));
        AuthDb::instance()->begin();
        return AuthDB::instance()->hasRole (
            AuthDB::instance()->authName() ,    // UID of the current logged user
            null ,                              // across all experiments of the hutch
            "ShiftMgr" ,                        // the application name 
            "Manage_{$hutch}"                   // the role name encodes the hutch name
        );
      */
    }

    /**
     * Return current logged-in user.
     */
    public function current_user() {
        AuthDb::instance()->begin();
        return AuthDB::instance()->authName();
    }

    public function get_stopper_out() {
      return 5 * 3600; // XXX fetch from PV
    }

    public function get_door_open() {
      return 3 * 3600; // XXX fetch from PV
    }

    public function get_total_shots() {
      return 60 * $this->get_stopper_out();
    }

    private static $hutches = array("SXR", "AMO", "CXI", "MEC", "XCS", "XPP");
    private static $areas = array("FEL", "Beamline", "Controls", "DAQ", "Laser", "Hutch/Hall", "Other");
    private static $uses = array("Tuning", "Alignment", "Data Taking", "Access", "Other", "Total");

    public function get_all_hutches() {
      return ShiftMgr::$hutches;
    }

    public function get_permitted_hutches() {
      $permitted_hutches = array();
      foreach (ShiftMgr::$hutches as $hutch) {
        if ($this->is_manager($hutch)) {
          array_push($permitted_hutches, $hutch);
        }
      }
      return $permitted_hutches;
    }

    public function get_uses() {
      return ShiftMgr::$uses;
    }

    public function get_areas() {
      return ShiftMgr::$areas;
    }

    /**
     * Return a list of shifts for specified hutch,
     * or if no hutch was specified, then for all hutches.
     * Returned shifts are "shallow" and do not contain area_evaluations, etc.
     */
    public function get_shifts($hutch = '', $earliest_start_time = 0, $latest_start_time = 0) {
      $where_clause = "WHERE start_time >= {$earliest_start_time}";
      if ($latest_start_time) {
        $where_clause .= " AND start_time <= {$latest_start_time}";
      }
      if ($hutch) {
        $this->assert_hutch($hutch);
        $where_clause .= " AND hutch='".$this->escape_string(strtoupper(trim($hutch)))."'";
      }
      $sql = "SELECT * FROM {$this->database}.shift {$where_clause} ORDER BY start_time, hutch DESC";
      $result = $this->query($sql);
      $nrows = mysql_numrows($result);

      $shifts = array();
      for ($i = 0; $i < $nrows; $i++) {
        array_push($shifts, new ShiftMgrShift($this, mysql_fetch_array($result, MYSQL_ASSOC)));
      }
      return $shifts;
    }

    // Returns last modified time for shift with id.
    public function get_shift_last_modified_time($id) {
      $sql = "SELECT last_modified_time FROM {$this->database}.shift where id={$id}"; // XXX should use prepared statements
      $result = $this->query($sql);
      $nrows = mysql_numrows($result);
      if ($nrows < 1) {
        throw new ShiftMgrException (
            __class__.'::'.__METHOD__ ,
            'Could not find shift with id ' . $id
        );
      }
      $shift = mysql_fetch_array($result, MYSQL_ASSOC);
      $last_modified_time = $shift["last_modified_time"];
      return $last_modified_time;
    }

    // Returns fully populated shift (with id, area evaluations, and time uses) for this id.
    public function get_shift($id) {
      $sql = "SELECT * FROM {$this->database}.shift where id={$id}"; // XXX should use prepared statements
      $result = $this->query($sql);
      $nrows = mysql_numrows($result);
      if ($nrows < 1) {
        throw new ShiftMgrException (
            __class__.'::'.__METHOD__ ,
            'Could not find shift with id ' . $id
        );
      }
      $shift = mysql_fetch_array($result, MYSQL_ASSOC);

      // Fetch all area_evaluation rows for this shift.
      $sql = "SELECT * FROM {$this->database}.area_evaluation WHERE id={$id}";
      $result = $this->query($sql);
      $nrows = mysql_numrows($result);
      $area_evaluations = array();
      for ($i = 0; $i < $nrows; $i++) {
        $area_evaluation = mysql_fetch_array($result, MYSQL_ASSOC);
        $area = $area_evaluation["area"];
        $area_evaluations[$area] = $area_evaluation;
      }
      $shift["area_evaluation"] = $area_evaluations;

      // Fetch all time_use rows for this shift.
      $sql = "SELECT * FROM {$this->database}.time_use WHERE id={$id}";
      $result = $this->query($sql);
      $nrows = mysql_numrows($result);
      $time_uses = array();
      for ($i = 0; $i < $nrows; $i++) {
        $time_use = mysql_fetch_array($result, MYSQL_ASSOC);
        $use = $time_use["use_name"];
        $time_uses[$use] = $time_use;
      }
      $shift["time_use"] = $time_uses;

      // Return fully populated shift.
      return $shift;
    }

    // Create a new shift.
    // Returns a fully populated shift (with id, area_evaluation map, etc.)
    public function create_shift($hutch, $start_time, $end_time) {
      $this->assert_hutch($hutch);
      $hutch_escaped = $this->escape_string(strtoupper(trim($hutch)));

      // Calculate values not passed as parameters
      $username = $this->current_user();
      $stopper_out = $this->get_stopper_out();
      $door_open = $this->get_door_open();
      $total_shots = $this->get_total_shots();
      $last_modified_time = LusiTime::now()->sec;

      // Create the shift
      $sql = "INSERT INTO {$this->database}.shift";
      $sql .= " VALUES(NULL,{$last_modified_time},'{$username}','{$hutch_escaped}',{$start_time},{$end_time},{$stopper_out},{$door_open},{$total_shots},'')";
      $this->query($sql);

      // Fetch id from newly created shift
      $sql = "SELECT id FROM {$this->database}.shift WHERE hutch='{$hutch_escaped}' AND start_time='{$start_time}' ORDER BY id DESC";
      $result = $this->query($sql);
      $nrows = mysql_numrows($result);
      if ($nrows < 1) {
        throw new ShiftMgrException (
            __class__.'::'.__METHOD__ ,
            'Could not find the shift we just created!'
        );
        return null;
      }
      $shallow_created_shift = mysql_fetch_array($result, MYSQL_ASSOC);
      $id = $shallow_created_shift["id"];

      // Create a row for each area
      foreach (ShiftMgr::$areas as $area) {
        $sql = "INSERT INTO {$this->database}.area_evaluation VALUES ({$id},'{$area}',1,0,'')";
        $this->query($sql);
      }

      // Create a row for each use
      foreach (ShiftMgr::$uses as $use) {
        $sql = "INSERT INTO {$this->database}.time_use VALUES ({$id},'{$use}',0,'')";
        $this->query($sql);
      }

      // Now fetch and return complete shift.
      return $this->get_shift($id);
    }

    // Do a "shallow" update of the shift. Don't update area evaluations, etc.
    // Return new last modified time.
    public function update_shift($id, $hutch, $start_time, $end_time, $other_notes) {
      $other_notes = $this->escape_string($other_notes);

      // Calculate values not passed as parameters
      $stopper_out = $this->get_stopper_out();
      $door_open = $this->get_door_open();
      $total_shots = $this->get_total_shots();
      $last_modified_time = LusiTime::now()->sec;

      // Do the update
      $sql = "UPDATE {$this->database}.shift";
      $sql .= " SET hutch='{$hutch}'";
      $sql .= ", start_time={$start_time}";
      $sql .= ", end_time={$end_time}";
      $sql .= ", last_modified_time={$last_modified_time}";
      $sql .= ", stopper_out={$stopper_out}";
      $sql .= ", door_open={$door_open}";
      $sql .= ", total_shots={$total_shots}";
      $sql .= ", other_notes='{$other_notes}'";
      $sql .= " WHERE id={$id}";
      $this->query($sql);

      // And return new last modified time.
      return $last_modified_time;
    }

    public function get_area_evaluation($id, $area) {
      $sql = "SELECT * FROM {$this->database}.area_evaluation WHERE id={$id} AND area='{$area}'";
      $result = $this->query($sql);
      $nrows = mysql_numrows($result);
      if ($nrows < 1) {
        return null;
      }
      return mysql_fetch_array($result, MYSQL_ASSOC);
    }

    public function update_area_evaluation($id, $area, $ok, $downtime, $comment) {
      $comment = $this->escape_string($comment);

      // These are the fields to update
      $sql = "UPDATE {$this->database}.area_evaluation SET";
      $sql .= " ok='{$ok}',";
      $sql .= " downtime={$downtime},";
      $sql .= "comment='{$comment}'";

      // Specify the record we want to update
      $sql .= " WHERE id={$id}";
      $sql .= " AND area='{$area}'";
      $this->query($sql);

      // And update shift so that others know to reload.
      $last_modified_time = LusiTime::now()->sec;
      $sql = "UPDATE {$this->database}.shift SET last_modified_time={$last_modified_time} WHERE id={$id}";
      $this->query($sql);

      // Return last modified time so that caller can update shift.
      return $last_modified_time;
    }

    public function update_time_use($id, $use, $use_time, $comment) {
      $comment = $this->escape_string($comment);

      // These are the fields to update
      $sql = "UPDATE {$this->database}.time_use SET";
      $sql .= " use_time={$use_time},";
      $sql .= "comment='{$comment}'";

      // Specify the record we want to update
      $sql .= " WHERE id={$id}";
      $sql .= " AND use_name='{$use}'";
      $this->query($sql);

      // And update shift so that others know to reload.
      $last_modified_time = LusiTime::now()->sec;
      $sql = "UPDATE {$this->database}.shift SET last_modified_time={$last_modified_time} WHERE id={$id}";
      $this->query($sql);

      // Return last modified time so that caller can update shift.
      return $last_modified_time;
    }
}

?>
