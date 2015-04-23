<?php

namespace DataPortal;

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use AuthDB\AuthDB;
use LusiTime\LusiTime;
use FileMgr\DbConnection;
use \stdClass;

/**
 * Class Config encapsulates operations with the logging database
 */
class Config extends DbConnection {

    /* ====================
     *   STATIC INTERFACE
     * ====================
     */

    private static $instance = null;

    /**
     * Singleton to simplify certain operations.
     *
     * @return unknown_type
     */
    public static function instance() {
        if (is_null(Config::$instance))
            Config::$instance =
                new Config(
                    PORTAL_DEFAULT_HOST,
                    PORTAL_DEFAULT_USER,
                    PORTAL_DEFAULT_PASSWORD,
                    PORTAL_DEFAULT_DATABASE);
        return Config::$instance;
    }

    /**
     * Constructor
     *
     * @param string $host
     * @param string $user
     * @param string $password
     * @param string $database 
     */
    public function __construct($host, $user, $password, $database) {
        parent::__construct($host, $user, $password, $database);
    }



    public function set_policy_param($storage, $attr, $value = '') {
        $storage_escaped = $this->escape_string(trim($storage));
        $attr_escaped = $this->escape_string(trim($attr));
        $sql = null;
        if (trim($value) == '') {
            if (!is_null($this->get_policy_param($storage, $attr)))
                $sql = "DELETE FROM {$this->database}.storage_policy WHERE storage='{$storage_escaped}' AND attr='{$attr_escaped}'";
        } else {
            $value_escaped = $this->escape_string(strtolower(trim($value)));
            $sql = is_null($this->get_policy_param($storage, $attr)) ?
                "INSERT INTO {$this->database}.storage_policy VALUES ('{$storage_escaped}','{$attr_escaped}','{$value_escaped}')" :
                "UPDATE {$this->database}.storage_policy SET value='{$value_escaped}' WHERE storage='{$storage_escaped}' AND attr='{$attr_escaped}'" ;
        }
        $this->query($sql);
    }

    public function get_policy_param($storage, $attr) {
        $storage_escaped = $this->escape_string(trim($storage));
        $attr_escaped = $this->escape_string(trim($attr));
        $sql = "SELECT value FROM {$this->database}.storage_policy WHERE storage='{$storage_escaped}' AND attr='{$attr_escaped}'";
        $result = $this->query($sql);
        $nrows = mysql_numrows($result);
        if (!$nrows) return null;
        if ($nrows != 1) throw new DataPortalException(__METHOD__, "duplicate entries for query {$sql}). Database can be corrupted.");
        $row = mysql_fetch_array($result, MYSQL_ASSOC);
        return trim($row['value']);
    }

    /* Add a persistent record for a file restore request which is passed as
     * a dictionary of:
     * 
     *   'exper_id'
     *   'runnum'
     *   'file_type'
     *   'irods_filepath'
     *   'irods_src_resource'
     *   'irods_dst_resource'
     * 
     * The methood will return an extended object as explain in the corresponding
     * find_* method:
     * 
     * @see Config::find_file_restore_request
     */

    public function add_file_restore_request($request) {
        $exper_id = intval($request['exper_id']);
        $runnum = intval($request['runnum']);
        $file_type = $this->escape_string(strtolower(trim($request['file_type'])));
        $irods_filepath = $this->escape_string(trim($request['irods_filepath']));
        $irods_src_resource = $this->escape_string(trim($request['irods_src_resource']));
        $irods_dst_resource = $this->escape_string(trim($request['irods_dst_resource']));
        $requested_time = LusiTime::now();
        $requested_uid = $this->escape_string(trim(AuthDB::instance()->authName()));
        $sql = "INSERT INTO {$this->database}.file_restore_requests VALUES ({$exper_id},{$runnum},'{$file_type}','{$irods_filepath}','{$irods_src_resource}','{$irods_dst_resource}',{$requested_time->to64()},'{$requested_uid}','RECEIVED')";
        $this->query($sql);
        return $this->find_file_restore_request($request);
    }

    /* Find a persistent record for a file restore request which is passed as
     * a dictionary of:
     * 
     *   'exper_id'
     *   'runnum'
     *   'file_type'
     *   'irods_filepath'
     *   'irods_src_resource'
     *   'irods_dst_resource'
     * 
     * If such request found then return an extended dictionary:
     *
     *   'exper_id'
     *   'runnum'
     *   'file_type'
     *   'irods_filepath'
     *   'irods_src_resource'
     *   'irods_dst_resource'
     *   'requested_time'
     *   'requested_uid'
     *
     * Otherwise return null.
     */

    public function find_file_restore_request($request) {
        $exper_id           = intval($request['exper_id']);
        $runnum             = intval($request['runnum']);
        $file_type          = $this->escape_string(strtolower(trim($request['file_type'])));
        $irods_filepath     = $this->escape_string(trim($request['irods_filepath']));
        $irods_src_resource = $this->escape_string(trim($request['irods_src_resource']));
        $irods_dst_resource = $this->escape_string(trim($request['irods_dst_resource']));
        $sql                = "SELECT * FROM {$this->database}.file_restore_requests WHERE exper_id={$exper_id} AND runnum={$runnum} AND file_type='{$file_type}' AND irods_filepath='{$irods_filepath}' AND irods_src_resource='{$irods_src_resource}' AND irods_dst_resource='{$irods_dst_resource}'";

        $result = $this->query($sql);
        $nrows  = mysql_numrows($result);
        if (!$nrows) return null;
        if ($nrows != 1) throw new DataPortalException(__METHOD__, "duplicate entries for query {$sql}). Database can be corrupted.");

        $row = mysql_fetch_array($result, MYSQL_ASSOC);
        $request['requested_time'] = LusiTime::from64($row['requested_time']);
        $request['requested_uid' ] = trim($row['requested_uid']);
        $request['status'        ] = trim($row['status']);

        return $request;
    }

    public function file_restore_requests($exper_id=null, $runnum=null, $file_type=null) {
        $list = array();

        $opt = '';
        if ($exper_id)  $opr .= " WHERE exper_id={$exper_id}";
        if ($runnum)    $opt .= ( $opt ? ' AND' : ' WHERE' ) . " runnum={$runnum}";
        if ($file_type) $opt .= ( $opt ? ' AND' : ' WHERE' ) . " file_type='" . $this->escape_string(strtolower(trim($file_type))) . "'";

        $result = $this->query("SELECT * FROM {$this->database}.file_restore_requests {$opt} ORDER BY requested_time DESC, exper_id, runnum, file_type");

        for ($i = 0, $nrows = mysql_numrows($result); $i < $nrows; $i++) {
            $row = mysql_fetch_array($result, MYSQL_ASSOC);
            $row['exper_id'      ] = intval($row['exper_id']);
            $row['runnum'        ] = intval($row['runnum']);
            $row['file_type'     ] = trim($row['file_type']);
            $row['irods_filepath'] = trim($row['irods_filepath']);
            $row['requested_time'] = LusiTime::from64($row['requested_time']);
            $row['requested_uid' ] = trim($row['requested_uid']);
            $row['status'        ] = trim($row['status']);
            array_push($list, $row);
        }
        return $list;
    }

    /**
     * Delete the specified file restore request (if any found in teh database)
     *
     * @param type $request 
     */
    public function delete_file_restore_request($request) {
        $exper_id           = intval($request['exper_id']);
        $runnum             = intval($request['runnum']);
        $file_type          = $this->escape_string(trim($request['file_type']));
        $irods_filepath     = $this->escape_string(trim($request['irods_filepath']));
        $irods_src_resource = $this->escape_string(trim($request['irods_src_resource']));
        $irods_dst_resource = $this->escape_string(trim($request['irods_dst_resource']));
        $sql                = "DELETE FROM {$this->database}.file_restore_requests WHERE exper_id={$exper_id} AND runnum={$runnum} AND file_type='{$file_type}' AND irods_filepath='{$irods_filepath}' AND irods_src_resource='{$irods_src_resource}' AND irods_dst_resource='{$irods_dst_resource}'";
        $this->query($sql);
    }

    /* Register a file in the MEDIUM-TERM storage with parameters found in
     * a dictionary of:
     * 
     *   'exper_id'
     *   'runnum'
     *   'file_type'
     *   'irods_filepath'
     *   'irods_resource'
     *   'irods_size'
     * 
     * The methood will return an extended object as explain in the corresponding
     * find_* method:
     * 
     * @see Config::find_medium_store_file
     */

    public function add_medium_store_file($request) {
        $exper_id        = intval($request['exper_id']);
        $runnum          = intval($request['runnum']);
        $file_type       = $this->escape_string(strtoupper(trim($request['file_type'])));
        $irods_filepath  = $this->escape_string(trim($request['irods_filepath']));
        $irods_resource  = $this->escape_string(trim($request['irods_resource']));
        $irods_size      = intval($request['irods_size']);
        $registered_time = LusiTime::now();
        $registered_uid  = $this->escape_string(trim(AuthDB::instance()->authName()));
        $sql             = "INSERT INTO {$this->database}.medium_term_storage VALUES ({$exper_id},{$runnum},'{$file_type}','{$irods_filepath}','{$irods_resource}',{$irods_size},{$registered_time->to64()},'{$registered_uid}')";
        $this->query($sql);

        return $this->find_medium_store_file($request);
    }

    /**
     * Remove the specified file from the MEDIUM-STORE registry
     *
     * @param type $exper_id
     * @param type $runnum
     * @param type $file_type
     * @param type $irods_filepath 
     */
    public function remove_medium_store_file($exper_id, $runnum, $file_type, $irods_filepath) {
        $exper_id       = intval($exper_id);
        $runnum         = intval($runnum);
        $file_type      = $this->escape_string(strtoupper(trim($file_type)));
        $irods_filepath = $this->escape_string(trim($irods_filepath));
        $sql            = "DELETE FROM {$this->database}.medium_term_storage WHERE exper_id={$exper_id} AND runnum={$runnum} AND file_type='{$file_type}' AND irods_filepath='{$irods_filepath}'";
        $this->query($sql);
    }

    /* Find a file entry in the MEDIUM-TERM storage using a request which is passed as
     * a dictionary of:
     * 
     *   'exper_id'
     *   'runnum'
     *   'file_type'
     *   'irods_filepath'
     *   'irods_resource'
     * 
     * If such request found then return an extended dictionary:
     *
     *   'exper_id'
     *   'runnum'
     *   'file_type'
     *   'irods_filepath'
     *   'irods_resource'
     *   'irods_size'
     *   'registered_time'
     *   'registered_uid'
     *
     * Otherwise return null.
     */

    public function find_medium_store_file($request) {
        $exper_id = intval($request['exper_id']);
        $runnum = intval($request['runnum']);
        $file_type = $this->escape_string(strtoupper(trim($request['file_type'])));
        $irods_filepath = $this->escape_string(trim($request['irods_filepath']));
        $irods_resource = $this->escape_string(trim($request['irods_resource']));
        $sql = "SELECT * FROM {$this->database}.medium_term_storage WHERE exper_id={$exper_id} AND runnum={$runnum} AND file_type='{$file_type}' AND irods_filepath='{$irods_filepath}' AND irods_resource='{$irods_resource}'";
        $result = $this->query($sql);
        $nrows = mysql_numrows($result);
        if (!$nrows) return null;
        if ($nrows != 1) throw new DataPortalException(__METHOD__, "duplicate entries for query {$sql}). Database can be corrupted.");
        $row = mysql_fetch_array($result, MYSQL_ASSOC);
        $request['irods_size'] = intval($row['irods_size']);
        $request['registered_time'] = LusiTime::from64($row['registered_time']);
        $request['registered_uid'] = trim($row['registered_uid']);
        return $request;
    }

    /**
     * Get a list of file entries for the specified experiment.
     * @param integer $exper_id
     * @param string $file_type
     * @return array
     */
    public function medium_store_files($exper_id, $file_type = null) {
        $list = array();
        $exper_id = intval($exper_id);
        $file_type_option = is_null($file_type) ? '' : ", AND file_type='" . $this->escape_string(strtoupper(trim($file_type))) . "'";
        $sql = "SELECT * FROM {$this->database}.medium_term_storage WHERE exper_id={$exper_id} {$file_type_option} ORDER BY runnum, file_type, irods_filepath";
        $result = $this->query($sql);
        for ($i = 0, $nrows = mysql_numrows($result); $i < $nrows; $i++) {
            $row = mysql_fetch_array($result, MYSQL_ASSOC);
            $row['runnum'] = intval($row['runnum']);
            $row['file_type'] = strtoupper(trim($row['file_type']));
            $row['irods_size'] = intval($row['irods_size']);
            $row['registered_time'] = LusiTime::from64($row['registered_time']);
            array_push($list, $row);
        }
        return $list;
    }

    /**
     * Calculated medium term quota usage (GB) for the experiment
     * @param type $exper_id
     * @return integer 
     */
    public function calculate_medium_quota($exper_id) {
        $used_gb = 0;
        $bytes_in_gb = 1024 * 1024 * 1024;
        foreach ($this->medium_store_files($exper_id) as $file) $used_gb += $file['irods_size'];
        return intval($used_gb / $bytes_in_gb);
    }

    /**
     * Create a new cache and return a descriptor of the cache
     *
     * @param string $application
     * @return integer
     */
    public function create_irods_cache($application) {
        $application_escaped = $this->escape_string(trim($application));
        $created_time_64 = LusiTime::now()->to64();
        $sql = "INSERT INTO {$this->database}.irods_cache VALUES (NULL,'{$application_escaped}',{$created_time_64})";
        $this->query($sql);
        return $this->find_irods_cache_by_('id=(SELECT LAST_INSERT_ID())');
    }

    /**
     * Find a single instance of a cache for the specified SQL condition.
     * Return cache descriptor if found or null otherwise.
     *
     * @param string $condition
     * @return a descriptor object with 3 public members: 'id','application','create_time'
     */
    private function find_irods_cache_by_($condition) {
        $sql = "SELECT * FROM {$this->database}.irods_cache WHERE {$condition}";
        $result = $this->query($sql);
        $nrows = mysql_numrows($result);
        if (!$nrows) return null;
        if ($nrows != 1) throw new DataPortalException(__METHOD__, "duplicate entries for query {$sql}). Database can be corrupted.");
        $row = mysql_fetch_array($result, MYSQL_ASSOC);
        $descr = new stdClass();
        $descr->id = intval($row['id']);
        $descr->application = intval($row['application']);
        $descr->create_time = LusiTime::from64($row['create_time']);
        return $descr;
    }

    /**
     * Populate the cache with runs and files using a data structure
     * returned by the Web service operation:
     * 
     *   GET /runs/{instrument}/{experiment}/{type}/{runs}
     *
     * @param integer $cache_id
     * @param array $runs 
     */
    public function add_files_irods_cache($cache_id, $exper_id, $file_type, $runs) {
        $file_type_escaped = $this->escape_string(trim($file_type));
        foreach ($runs as $r) {
            $run_sql = "INSERT INTO {$this->database}.irods_type_run VALUES(NULL,{$cache_id},{$exper_id},'{$file_type_escaped}',{$r->run})";
            $this->query($run_sql);
            $run_descriptor = $this->find_irods_type_run_by_('id=(SELECT LAST_INSERT_ID())');
            $run_id = $run_descriptor->id;
            foreach ($r->files as $f) {
                $type_escaped = $this->escape_string(trim($f->type));
                $name_escaped = $this->escape_string(trim($f->name));
                $url_escaped = $this->escape_string(trim($f->url));
                $file_sql = "INSERT INTO {$this->database}.irods_file VALUES({$run_id},{$r->run},'{$type_escaped}','{$name_escaped}','{$url_escaped}',";
                if ($f->type == 'object') {
                    $checksum = $this->escape_string(trim($f->checksum));
                    $collName = $this->escape_string(trim($f->collName));
                    $datamode = $this->escape_string(trim($f->datamode));
                    $owner = $this->escape_string(trim($f->owner));
                    $path = $this->escape_string(trim($f->path));
                    $replStat = $this->escape_string(trim($f->replStat));
                    $resource = $this->escape_string(trim($f->resource));
                    $file_sql .= "'{$checksum}','{$collName}',{$f->ctime},'{$datamode}',{$f->id},{$f->mtime},'{$owner}','{$path}','{$replStat}',{$f->replica},'{$resource}',{$f->size})";
                } else {
                    $file_sql .= "NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)";
                }
                $this->query($file_sql);
            }
        }
    }

    /**
     * Find a descriptor for a (file_type,runnum) in the irods cache
     *
     * @param string $condition 
     * @return array of 5 keys 'id','cache_id','exper_id','file_type','runnum'
     */
    private function find_irods_type_run_by_($condition) {
        $sql = "SELECT * FROM {$this->database}.irods_type_run WHERE {$condition}";
        $result = $this->query($sql);
        $nrows = mysql_numrows($result);
        if (!$nrows) return null;
        if ($nrows != 1) throw new DataPortalException(__METHOD__, "duplicate entries for query {$sql}). Database can be corrupted.");
        $row = mysql_fetch_array($result, MYSQL_ASSOC);
        $descr = new stdClass();
        $descr->id = intval($row['id']);
        $descr->cache_id = intval($row['cache_id']);
        $descr->exper_id = intval($row['exper_id']);
        $descr->file_type = intval($row['file_type']);
        $descr->runnum = intval($row['runnum']);
        return $descr;
    }

    private function find_irods_type_run_ids($condition) {
        $ids = array();
        $sql = "SELECT id FROM {$this->database}.irods_type_run WHERE {$condition}";
        $result = $this->query($sql);
        for ($i = 0, $nrows = mysql_numrows($result); $i < $nrows; $i++) {
            $row = mysql_fetch_array($result, MYSQL_ASSOC);
            array_push($ids, intval($row['id']));
        }
        return $ids;
    }

    /**
     * Check if the database has a cache for the specified application and if so
     * return cached values for runs and files into a data structure
     * similar to the one returned by the Web service operation:
     * 
     *   GET /runs/{instrument}/{experiment}/{type}/{runs}
     *
     * If no cache exists the function shall return null.
     *
     * @param string $application
     * @param integer $exper_id
     * @param string $file_type
     * @return array 
     */
    public function irods_files_from_recent_cache($application, $exper_id, $file_type) {
        $application_escaped = $this->escape_string(trim($application));
        $cache = $this->find_irods_cache_by_("application='{$application_escaped}' AND create_time IN (SELECT MAX(create_time) FROM {$this->database}.irods_cache WHERE application='{$application_escaped}')");
        if (is_null($cache)) return null;
        return $this->irods_files($cache->id, $exper_id, $file_type);
    }

    /**
     * Return cached values for runs and files into a data structure
     * similar to the one returned by the Web service operation:
     * 
     *   GET /runs/{instrument}/{experiment}/{type}/{runs}
     *
     * @param integer $cache_id
     * @param integer $exper_id
     * @param string $file_type
     * @return array 
     */
    public function irods_files($cache_id, $exper_id, $file_type) {
        $runs = array();
        $file_type_escaped = $this->escape_string(trim($file_type));
        $run_ids = $this->find_irods_type_run_ids("cache_id={$cache_id} AND exper_id={$exper_id} AND file_type='{$file_type_escaped}'");
        $runs_ids_str = '';
        foreach ($run_ids as $id) {
            if ($runs_ids_str != '') $runs_ids_str .= ',';
            $runs_ids_str .= $id;
        }
        if ($runs_ids_str == '') return $runs;  // no runs containg the requested type found for the experiment

        $sql = "SELECT * FROM {$this->database}.irods_file WHERE run_id IN ({$runs_ids_str}) ORDER BY run, name";
        $result = $this->query($sql);
        $current_run = null;
        for ($i = 0, $nrows = mysql_numrows($result); $i < $nrows; $i++) {
            $row = mysql_fetch_array($result, MYSQL_ASSOC);
            $run = intval($row['run']);

            if (!is_null($current_run) && ($run != $current_run->run))
                array_push($runs, $current_run);

            $current_run = new stdClass();
            $current_run->run = $run;
            $current_run->files = array();

            $f = new stdClass();
            $f->type = $row['type'];
            $f->name = $row['name'];
            $f->url = $row['url'];
            if ($f->type == 'object') {
                $f->checksum = $row['checksum'];
                $f->collName = $row['collName'];
                $f->ctime = $row['ctime'];
                $f->datamode = $row['datamode'];
                $f->id = $row['id'];
                $f->mtime = $row['mtime'];
                $f->owner = $row['owner'];
                $f->path = $row['path'];
                $f->replStat = $row['replStat'];
                $f->replica = $row['replica'];
                $f->resource = $row['resource'];
                $f->size = $row['size'];
            }
            array_push($current_run->files, $f);
        }
        if (!is_null($current_run))
            array_push($runs, $current_run);

        return $runs;
    }

    public function do_notify($address, $subject, $body, $application) {
        $tmpfname = tempnam("/tmp", "webportal");
        $handle = fopen($tmpfname, "w");
        fwrite($handle, $body);
        fclose($handle);

        shell_exec("cat {$tmpfname} | mail -s '{$subject}' {$address} -- -F '{$application}'");

        // Delete the file only after piping its contents to the mailer command.
        // Otherwise its contents will be lost before we use it.
        //
        unlink($tmpfname);
    }

    /**
     * Find and return an application parameter for the current user.
     * The method will return null if no such parameter is known to the database.
     *
     * Note that an array returned by the operation will contain 3 representations
     * for the update time of teh parameter:
     * 
     *   update_time     - human-readable text (seconds resolution)
     *   update_time_sec - a 32-bit number of seconds since UNIX Epoch
     *   update_time_64  - a 64-bit number of nano-seconds since UNIX Epoch
     *
     * @param string $application
     * @param string $scope
     * @param string $parameter
     * @return array
     * @throws DataPortalException
     */
    public function find_application_parameter ($application, $scope, $parameter) {
        $uid = AuthDB::instance()->authName() ;
        $application_escaped = $this->escape_string(trim($application)) ;
        $scope_escaped      = $this->escape_string(trim($scope)) ;
        $parameter_escaped  = $this->escape_string(trim($parameter)) ;
        $sql = "SELECT * FROM {$this->database}.application_config WHERE uid='{$uid}' AND application='{$application_escaped}' AND scope='{$scope_escaped}' AND parameter='{$parameter_escaped}'" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows($result) ;
        if (0 == $nrows) return null ;
        if (1 != $nrows)
            throw new DataPortalException (
                __METHOD__ ,
                "duplicate entries for query {$sql}). Database can be corrupted."
            ) ;
        $row = mysql_fetch_array($result, MYSQL_ASSOC) ;
        $update_time = LusiTime::from64(intval($row['update_time'])) ;
        $row['update_time']     = $update_time->toStringShort() ;
        $row['update_time_sec'] = $update_time->sec ;
        $row['update_time_64']  = $update_time->to64() ;
        return $row ;
    }

    /**
     * Add or update a value of an application parameter in the database. Return
     * the value description back.
     * 
     * @param string $application
     * @param string $scope
     * @param string $parameter
     * @param string $value
     * @return array
     */
    public function save_application_parameter ($application, $scope, $parameter, $value) {
        $uid               = AuthDB::instance()->authName() ;
        $application_escaped = $this->escape_string(trim($application)) ;
        $scope_escaped      = $this->escape_string(trim($scope)) ;
        $parameter_escaped  = $this->escape_string(trim($parameter)) ;
        $value_escaped      = $this->escape_string(trim($value)) ;
        $update_time_64     = LusiTime::now()->to64() ;
        $sql = is_null($this->find_application_parameter($application, $scope, $parameter)) ?
            "INSERT INTO {$this->database}.application_config VALUES('{$uid}','{$application_escaped}','{$scope_escaped}','{$parameter_escaped}','{$value_escaped}',{$update_time_64})" :
            "UPDATE {$this->database}.application_config SET value='{$value_escaped}', update_time={$update_time_64} WHERE uid='{$uid}' AND application='{$application_escaped}' AND scope='{$scope_escaped}' AND parameter='{$parameter_escaped}'" ;
        $this->query($sql) ;
        return $this->find_application_parameter($application, $scope, $parameter) ;
    }
}
?>
