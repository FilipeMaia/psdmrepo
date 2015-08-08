<?php

namespace FileMgr;

require_once( 'filemgr.inc.php' );

use FileMgr\FileMgrException;
use FileMgr\DbConnection;


/* ATTENTION: This limit is required to deal with huge data structures/collections
 * produced by some PHP functions when dealing with irodsws collections. Consider
 * increasing it further down if the interpreter will stop working and if the Web
 * server's log file /var/log/httpd/error_log will say something like:
 *
 *  ..
 *  Allowed memory size of 16777216 bytes exhausted (tried to allocate 26 bytes)
 *  ..
 */
ini_set("memory_limit","256M");

use \stdClass;

define ('ATIME_SHIFT',   100*1000*1000) ;

/*
 * The helper utility class to deal with iRODS Database
 */
class FileMgrIrodsDb extends DbConnection {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $instance = null;

    /**
     * Singleton to simplify certain operations.
     *
     * @return FileMgrIrodsDb
     */
    public static function instance() {
        if( is_null( FileMgrIrodsDb::$instance ))
            FileMgrIrodsDb::$instance =
                new FileMgrIrodsDb (
                    ICAT_DEFAULT_HOST,
                    ICAT_DEFAULT_USER,
                    ICAT_DEFAULT_PASSWORD,
                    ICAT_DEFAULT_DATABASE );
        return FileMgrIrodsDb::$instance;
    }

    /**
     * Constructor
     *
     * @param string $host
     * @param string $user
     * @param string $password
     * @param string $database 
     */
    public function __construct ($host, $user, $password, $database) {
        parent::__construct ( $host, $user, $password, $database );
    }

    /** Return a list of runs and files in a format similar to what's returned by the corresponding Web service
     * 
     * @param string $instrument
     * @param string $experiment
     * @param string $type
     * @param integer $first_run
     * @param integer $last_run
     * @return array
     */
    public function runs ($instrument, $experiment, $type, $first_run=null, $last_run=null) {
     
        if (!is_null($first_run)) {
            $first_run = intval($first_run);
//            if ($first_run == 0)
//                throw new FileMgrException(__METHOD__, 'illegal parameter for the first run');
        }
        if (!is_null($last_run)) {
            $last_run = intval($last_run);
//            if ($last_run == 0)
//                throw new FileMgrException(__METHOD__, 'illegal parameter for the last run');
        }
        if (($first_run && $last_run) && ($first_run > $last_run))
            throw new FileMgrException(__METHOD__, 'illegal parameters: the first run must be equal or strictly less than the last run');

        // Find the collection first
        //
        $collection = $this->find_collection($instrument, $experiment, $type);
        if (is_null($collection)) return array();

        // Now find all files in the collection
        //
        $sql = 'SELECT * FROM r_data_main WHERE coll_id='.$collection['id'].' ORDER BY data_name';

        return $this->file_query2runs ($sql, $collection['name'], $first_run, $last_run);
    }

    public function find_file($instrument, $experiment, $type, $name) {

        $name_escaped = $this->escape_string(trim($name));

        // Find the collection first
        //
        $collection = $this->find_collection($instrument, $experiment, $type);
        if (is_null($collection)) return array();

        // Now find all files in the collection
        //
        $sql = 'SELECT * FROM r_data_main WHERE coll_id='.$collection['id']." AND data_name='{$name_escaped}'";

        return $this->file_query2runs ($sql, $collection['name']);
    }

    /**
     * Return a collection descriptor (name,id) or null if not found.
     *
     * @param string $instrument
     * @param string $experiment
     * @param string $type
     * @return array
     * @throws FileMgrException
     */
    private function find_collection($instrument, $experiment, $type) {
        $instrument_escaped = $this->escape_string(trim($instrument));
        $experiment_escaped = $this->escape_string(trim($experiment));
        $type_escaped       = $this->escape_string(strtolower(trim($type)));
        $coll_name          = "/psdm-zone/psdm/{$instrument_escaped}/{$experiment_escaped}/{$type_escaped}";
        $sql                = "SELECT coll_id FROM r_coll_main WHERE coll_name='{$coll_name}'";
        $result             = $this->query ($sql);
        $nrows              = mysql_numrows($result);
        if (!$nrows) return null;
        if ($nrows != 1) throw new FileMgrException(__METHOD__, "duplicate entries for query {$sql}). Database can be corrupted.");
        $row = mysql_fetch_array($result, MYSQL_ASSOC);
        return array(
            'name' => $coll_name,
            'id'   => intval($row['coll_id']));
    }

    private function file_query2runs ($sql, $collection_name, $first_run=null, $last_run=null) {
        $result = $this->query ($sql);
        $infiles = array();
        for ($i=0, $nrows = mysql_numrows($result); $i<$nrows; $i++) {
            $row = mysql_fetch_array($result, MYSQL_ASSOC);
            $f = new stdClass();
            $f->type     = 'object';
            $f->name     = $row['data_name'];
            $f->url      = '';
            $f->checksum = $row['data_checksum'];
            $f->collName = $collection_name;
            $f->ctime    = $row['create_ts'];
            $f->atime    = $row['data_expiry_ts'] ? $row['data_expiry_ts'] - ATIME_SHIFT : $f->ctime ;
            $f->datamode = $row['data_mode'];
            $f->id       = $row['data_id'];
            $f->mtime    = $row['modify_ts'];
            $f->owner    = $row['data_owner_name'];
            $f->path     = $row['data_path'];
            $f->replStat = $row['data_status'];
            $f->replica  = $row['data_repl_num'];
            $f->resource = $row['resc_name'];
            $f->size     = $row['data_size'];
            array_push($infiles, $f);
        }
        $runs2files = array();
        foreach ($infiles as $f) {
            $run = 0;
            $matches = null;
            $stat = preg_match ('/^.+\-r(\d+)/', $f->name, $matches);
            if (!$stat) {
                if ($stat === false)
                    throw new FileMgrException(
                        __METHOD__,
                        "failed to extract run number from file '{$f->name}' of collection {$f->collName}");
                else
                    continue;
            }
            $run = intval($matches[1]);

            // Skip rans which are not in the specified range
            //
            if ($first_run && ($run < $first_run)) continue;
            if ($last_run  && ($run > $last_run )) continue;

            if (!array_key_exists($run, $runs2files)) $runs2files[$run] = array();
            array_push($runs2files[$run], $f);
        }
        $runs = array();
        foreach ($runs2files as $run => $files) {
            $r = new stdClass();
            $r->run = $run;
            $r->files = $files;
            array_push($runs, $r);
        }
        return $runs;
    }
}

/* =======================
 * UNIT TEST FOR THE CLASS
 * =======================
 *
try {
    $irodsdb = FileMgrIrodsDb::instance();
    $irodsdb->begin();
    $runs = $irodsdb->runs('CXI', 'cxi80410', 'xtc');
    foreach ($runs as $run) {
        print '<pre>'.print_r($run, true).'</pre>';
    }
    $irodsdb->commit();

} catch (Exception        $e) { print $e; }
  catch (FileMgrException $e) { print $e->toHtml(); }
*/
?>
