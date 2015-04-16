<?php

namespace SysMon ;

require_once 'sysmon.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'regdb/regdb.inc.php' ;

use \AuthDB\AuthDB ;
use \FileMgr\DbConnection ;
use \LusiTime\LusiTime ;
use RegDB\RegDB ;

define ('BYTES_IN_KB', 1024.0) ;
define ('BYTES_IN_MB', 1024.0 * 1024.0) ;
define ('BYTES_IN_GB', 1024.0 * 1024.0 * 1024.0) ;
define ('BYTES_IN_TB', 1024.0 * 1024.0 * 1024.0 * 1024) ;

define ('SECONDS_IN_MONTH', 30 * 24 * 3600) ;

/**
 * Class SysMon encapsulates operations with the PCDS System Monitoring database
 *
 * @author gapon
 */
class SysMon extends DbConnection {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $instance = null ;

    /**
     * Singleton to simplify certain operations.
     *
     * @return SysMon
     */
    public static function instance () {
        if (is_null(SysMon::$instance)) SysMon::$instance =
            new SysMon (
                SYSMON_DEFAULT_HOST,
                SYSMON_DEFAULT_USER,
                SYSMON_DEFAULT_PASSWORD,
                SYSMON_DEFAULT_DATABASE) ;
        return SysMon::$instance ;
    }

    public static function autoformat_size ($bytes) {
        if      ($bytes < BYTES_IN_KB) return sprintf(                                        "%d   ", $bytes) ;
        else if ($bytes < BYTES_IN_MB) return sprintf($bytes < 10 * BYTES_IN_KB ? "%.1f KB" : "%d KB", $bytes / BYTES_IN_KB) ;
        else if ($bytes < BYTES_IN_GB) return sprintf($bytes < 10 * BYTES_IN_MB ? "%.1f MB" : "%d MB", $bytes / BYTES_IN_MB );
        else if ($bytes < BYTES_IN_TB) return sprintf($bytes < 10 * BYTES_IN_GB ? "%.1f GB" : "%d GB", $bytes / BYTES_IN_GB) ;
        else                           return sprintf($bytes < 10 * BYTES_IN_TB ? "%.1f TB" : "%d TB", $bytes / BYTES_IN_TB) ;
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

    /* -------------------
     *   Users and roles
     * -------------------
     */
    public function is_administrator () {
        AuthDb::instance()->begin() ;
        return AuthDB::instance()->hasRole(AuthDB::instance()->authName(), null, 'SysMon', 'Admin') ;
    }

    /* ----------------
     *   File Systems
     * ----------------
     */
    public function file_systems ($class=null, $latest_snapshot=true) {
        $list = array () ;
        $class_opt = '' ;
        if (!is_null($class)) $class_opt = "WHERE base_path LIKE '%/".$this->escape_string(strtolower(trim($class)))."'" ;
        $sql = "SELECT * FROM {$this->database}.file_system {$class_opt} ORDER BY base_path, scan_time DESC" ;
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            $base_path = $attr['base_path'] ;
            $scan = array (
                'id' => intval($attr['id']) ,
                'scan_time' => new LusiTime (intval($attr['scan_time']))
            ) ;
            if (array_key_exists($base_path, $list)) {
                if (!$latest_snapshot) array_push ($list[$base_path], $scan) ;
            } else {
                $list[$base_path] = array ($scan) ;
            }
        }
        return $list ;
    }
    public function find_file_system_by_id ($id) {
        $result = $this->query("SELECT * FROM {$this->database}.file_system WHERE id={$id}") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1) throw new SysMonException (
            __class__.'::'.__METHOD__ ,
            'inconsistent result returned from the database. Wrong schema?') ;
        $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
        return array ($attr['base_path'] => array (
            'id' => intval($attr['id']) ,
            'scan_time' => new LusiTime (intval($attr['scan_time']))
        )) ;
    }
    public function file_system_summary ($class=null, $id=null) {
        $entry_types = $this->file_entry_types($class, $id) ;
        $total_size  = $this->file_total_size ($class, $id) ;
        $summary = array (
            'file_systems' => array () ,
            'files'       => array_key_exists('FILE',  $entry_types) ? $entry_types['FILE']  : 0 ,
            'directories' => array_key_exists('DIR',   $entry_types) ? $entry_types['DIR' ]  : 0 ,
            'links'       => array_key_exists('LINK',  $entry_types) ? $entry_types['LINK']  : 0 ,
            'others'      => array_key_exists('OTHER', $entry_types) ? $entry_types['OTHER'] : 0 ,
            'size'        => $total_size ? SysMon::autoformat_size( $total_size ) : ''
        ) ;
        if (is_null($id)) {
            $summary['file_systems'] = $this->file_systems($class) ;
        } else {
            $fs = $this->find_file_system_by_id($id) ;
            if (is_null($fs))
                throw new SysMonException (
                    __class__.'::'.__METHOD__, "no file system for id='{$id}' found in the database") ;
            $summary['file_systems'] = $fs ;
        }
        return $summary ;
    }
    private function file_entry_types ($class=null, $id=null) {
        $list = array() ;
        $sql = "SELECT entry_type, COUNT(*) AS `num_files` FROM {$this->database}.file_catalog GROUP BY entry_type" ;
        if ($id) {
            $sql = "SELECT entry_type, COUNT(*) AS `num_files` FROM {$this->database}.file_catalog WHERE file_system_id={$id} GROUP BY entry_type" ;
        } elseif ($class) {
            $class_opt = "WHERE base_path LIKE '%/".$this->escape_string(strtolower(trim($class)))."'" ;
            $sql_class_subquery = "SELECT id FROM {$this->database}.file_system {$class_opt}" ;
            $sql = "SELECT entry_type, COUNT(*) AS `num_files` FROM {$this->database}.file_catalog WHERE file_system_id IN ({$sql_class_subquery}) GROUP BY entry_type" ;
        }
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            $list[$attr['entry_type']] = intval($attr['num_files']) ;
        }
        return $list ;
    }
    private function file_total_size ($class=null, $id=null) {
        if ($id) {
            $sql = "SELECT SUM(size_bytes) AS `size_bytes` FROM {$this->database}.file_catalog WHERE entry_type='FILE' AND file_system_id={$id}" ;
        } elseif ($class) {
            $class_opt = "WHERE base_path LIKE '%/".$this->escape_string(strtolower(trim($class)))."'" ;
            $sql_class_subquery = "SELECT id FROM {$this->database}.file_system {$class_opt}" ;
            $sql = "SELECT SUM(size_bytes) AS `size_bytes` FROM {$this->database}.file_catalog WHERE entry_type='FILE' AND file_system_id IN ({$sql_class_subquery})" ;
        }
        $result = $this->query($sql) ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1) throw new SysMonException (
            __class__.'::'.__METHOD__ ,
            'inconsistent result returned from the database. Wrong schema?') ;
        $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
        return intval($attr['size_bytes']) ;
    }
    public function file_sizes ($class=null, $id=null) {
        $list_sizes = array (
            array('max_size' =>   '1 KB', 'max_size_bytes' =>   1 * BYTES_IN_KB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' =>  '10 KB', 'max_size_bytes' =>  10 * BYTES_IN_KB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' => '100 KB', 'max_size_bytes' => 100 * BYTES_IN_KB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' =>   '1 MB', 'max_size_bytes' =>   1 * BYTES_IN_MB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' =>  '10 MB', 'max_size_bytes' =>  10 * BYTES_IN_MB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' => '100 MB', 'max_size_bytes' => 100 * BYTES_IN_MB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' =>   '1 GB', 'max_size_bytes' =>   1 * BYTES_IN_GB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' =>  '10 GB', 'max_size_bytes' =>  10 * BYTES_IN_GB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' => '100 GB', 'max_size_bytes' => 100 * BYTES_IN_GB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' =>   '1 TB', 'max_size_bytes' =>   1 * BYTES_IN_TB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('max_size' =>  '10 TB', 'max_size_bytes' =>  10 * BYTES_IN_TB, 'num_files' => 0, 'size_bytes' => 0, 'size' => '')
        ) ;                    
        if ($id) {
            $sql = "SELECT size_bytes,ctime FROM {$this->database}.file_catalog WHERE entry_type='FILE' AND file_system_id={$id}" ;
        } elseif ($class) {
            $class_opt = "WHERE base_path LIKE '%/".$this->escape_string(strtolower(trim($class)))."'" ;
            $sql_class_subquery = "SELECT id FROM {$this->database}.file_system {$class_opt}" ;
            $sql = "SELECT size_bytes,ctime FROM {$this->database}.file_catalog WHERE entry_type='FILE' AND file_system_id IN ({$sql_class_subquery})" ;
        }
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {

            $row = mysql_fetch_row ($result) ;

            $size_bytes = intval($row[0]) ;

            $size_overflow = true ;
            for ($size_range = 0 ; $size_range < count($list_sizes) ; $size_range++) {
                if ($size_bytes < $list_sizes[$size_range]['max_size_bytes']) {
                    $list_sizes[$size_range]['num_files' ] += 1 ;
                    $list_sizes[$size_range]['size_bytes'] += $size_bytes ;
                    $size_overflow = false ;
                    break ;
                }
            }
        }
        for ($size_range = 0 ; $size_range < count($list_sizes) ; $size_range++)
            $list_sizes[$size_range]['size'] = SysMon::autoformat_size($list_sizes[$size_range]['size_bytes']) ;

        return $list_sizes ;
    }
    public function file_ctime ($class=null, $id=null) {
        $now_sec = LusiTime::now()->sec ;
        $list_ctime = array (
            array('ctime_age' =>  '1 month', 'ctime_age_sec' => $now_sec -  1 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>  '2 month', 'ctime_age_sec' => $now_sec -  2 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>  '3 month', 'ctime_age_sec' => $now_sec -  3 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>  '4 month', 'ctime_age_sec' => $now_sec -  4 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>  '5 month', 'ctime_age_sec' => $now_sec -  5 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>  '6 month', 'ctime_age_sec' => $now_sec -  6 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>  '7 month', 'ctime_age_sec' => $now_sec -  7 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>  '8 month', 'ctime_age_sec' => $now_sec -  8 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>  '9 month', 'ctime_age_sec' => $now_sec -  9 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' => '10 month', 'ctime_age_sec' => $now_sec - 10 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' => '11 month', 'ctime_age_sec' => $now_sec - 11 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' => '12 month', 'ctime_age_sec' => $now_sec - 12 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' => '24 month', 'ctime_age_sec' => $now_sec - 24 * SECONDS_IN_MONTH, 'num_files' => 0, 'size_bytes' => 0, 'size' => '') ,
            array('ctime_age' =>    'older', 'ctime_age_sec' => 0,                            'num_files' => 0, 'size_bytes' => 0, 'size' => '')
        ) ;
        if ($id) {
            $sql = "SELECT size_bytes,ctime FROM {$this->database}.file_catalog WHERE entry_type='FILE' AND file_system_id={$id}" ;
        } elseif ($class) {
            $class_opt = "WHERE base_path LIKE '%/".$this->escape_string(strtolower(trim($class)))."'" ;
            $sql_class_subquery = "SELECT id FROM {$this->database}.file_system {$class_opt}" ;
            $sql = "SELECT size_bytes,ctime FROM {$this->database}.file_catalog WHERE entry_type='FILE' AND file_system_id IN ({$sql_class_subquery})" ;
        }
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {

            $row = mysql_fetch_row ($result) ;

            $size_bytes = intval($row[0]) ;
            $ctime      = intval($row[1]) ;

            $ctime_overflow = true ;
            for ($ctime_range = 0 ; $ctime_range < count($list_ctime) ; $ctime_range++) {
                if ($ctime >= $list_ctime[$ctime_range]['ctime_age_sec']) {
                    $list_ctime[$ctime_range]['num_files' ] += 1 ;
                    $list_ctime[$ctime_range]['size_bytes'] += $size_bytes ;
                    $ctime_overflow = false ;
                    break ;
                }
            }
        }
        for ($ctime_range = 0 ; $ctime_range < count($list_ctime) ; $ctime_range++)
            $list_ctime[$ctime_range]['size'] = SysMon::autoformat_size($list_ctime[$ctime_range]['size_bytes']) ;

        return $list_ctime ;
    }
    public function file_extensions ($class=null, $id=null) {
        $list = array() ;
        if ($id) {
            $sql = "SELECT extension, COUNT(*) AS `num_files`, SUM(size_bytes) AS `size_bytes` FROM {$this->database}.file_catalog WHERE entry_type='FILE' AND file_system_id={$id} GROUP BY extension" ;
        } elseif ($class) {
            $class_opt = "WHERE base_path LIKE '%/".$this->escape_string(strtolower(trim($class)))."'" ;
            $sql_class_subquery = "SELECT id FROM {$this->database}.file_system {$class_opt}" ;
            $sql = "SELECT extension, COUNT(*) AS `num_files`, SUM(size_bytes) AS `size_bytes` FROM {$this->database}.file_catalog WHERE entry_type='FILE' AND file_system_id IN ({$sql_class_subquery}) GROUP BY extension" ;
        }
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            $extension = $attr['extension'] ;
            if (!array_key_exists($extension, $list))
                $list[$extension] = array(
                    'num_files'  => 0 ,
                    'size_bytes' => 0
                ) ;
            $list[$extension]['num_files' ] += intval($attr['num_files' ]) ;
            $list[$extension]['size_bytes'] += intval($attr['size_bytes']) ;
        }
        foreach ($list as $extension => $entry)
            $list[$extension]['size'] =
                SysMon::autoformat_size($list[$extension]['size_bytes']) ;

        return $list ;
    }
    public function file_types ($class=null, $id=null) {
        $list = array() ;
        if ($id) {
            $sql = "SELECT t.name as `name`, COUNT(*) AS 'num_files', SUM(c.size_bytes) AS 'size_bytes' FROM {$this->database}.file_type AS `t`, {$this->database}.file_catalog AS `c` WHERE c.entry_type='FILE' AND c.file_system_id={$id} AND t.file_system_id=c.file_system_id AND t.id=c.file_type_id GROUP BY t.name ORDER BY t.name" ;
        } elseif ($class) {
            $class_opt = "WHERE base_path LIKE '%/".$this->escape_string(strtolower(trim($class)))."'" ;
            $sql_class_subquery = "SELECT id FROM {$this->database}.file_system {$class_opt}" ;
            $sql = "SELECT t.name as `name`, COUNT(*) AS 'num_files', SUM(c.size_bytes) AS 'size_bytes' FROM {$this->database}.file_type AS `t`, {$this->database}.file_catalog AS `c` WHERE c.entry_type='FILE' AND c.file_system_id IN ({$sql_class_subquery}) AND t.file_system_id=c.file_system_id AND t.id=c.file_type_id GROUP BY t.name ORDER BY t.name" ;
        }
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            $name = $attr['name'] ;
            if (!array_key_exists($name, $list))
                $list[$name] = array(
                    'num_files'  => 0 ,
                    'size_bytes' => 0 ,
                    'size'       => ''
                ) ;
            $list[$name]['num_files' ] += intval($attr['num_files' ]) ;
            $list[$name]['size_bytes'] += intval($attr['size_bytes']) ;
        }
        foreach ($list as $name => $entry)
            $list[$name]['size'] =
                SysMon::autoformat_size($list[$name]['size_bytes']) ;

        $list2array = array () ;
        foreach ($list as $name => $entry)
            array_push (
                $list2array ,
                array (
                    'name'       => $name ,
                    'num_files'  => $entry['num_files' ] ,
                    'size_bytes' => $entry['size_bytes'] ,
                    'size'       => $entry['size'      ]
                )
            ) ;

        return $list2array ;
    }
    
    /**
     * Return an object representing the specified psanamon plot (if any).
     *
     * @param integer $exper_id
     * @param string $name
     * @return \SysMon\SysMonPsanaMonPlot|null
     * @throws SysMonException
     */
    public function find_psanamon_plot ($exper_id, $name) {

        if (!$exper_id) throw new SysMonException (
            __class__.'::'.__METHOD__ ,
            'experiment ID passed into the method is not valid') ;

        if (is_null($name) || $name === '') throw new SysMonException (
            __class__.'::'.__METHOD__ ,
            'plot name passed into the method is not valid') ;

        $name_esc = $this->escape_string(strtolower(trim($name))) ;

        return $this->find_psanamon_plot_by_("exper_id={$exper_id} and name='{$name_esc}'") ;
    }

    /**
     * Return an object representing the specified psanamon plot (if any).
     * 
     * @param integer $id
     * @return \SysMon\SysMonPsanaMonPlot|null
     * @throws SysMonException
     */
    public function find_psanamon_plot_by_id ($id) {
        if (!$id) throw new SysMonException (
            __class__.'::'.__METHOD__ ,
            'plot ID passed into the method is not valid') ;
        return $this->find_psanamon_plot_by_("id={$id}") ;
    }

    /**
     * Return an object representing the specified psanamon plot (if any).
     * 
     * @param string $cond
     * @return \SysMon\SysMonPsanaMonPlot|null
     * @throws SysMonException
     */
    private function find_psanamon_plot_by_ ($cond) {

        $result = $this->query (
            "SELECT id, exper_id, name, type, descr, LENGTH(data) AS data_size, update_time, update_uid" .
            " FROM {$this->database}.psanamon_plot_m" .
            " WHERE {$cond}") ;

        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1) throw new SysMonException (
            __class__.'::'.__METHOD__ ,
            'inconsistent result returned from the database. Wrong schema?') ;

        return new SysMonPsanaMonPlot (
            $this ,
            mysql_fetch_array ($result, MYSQL_ASSOC)) ;
    }
    

    
    /* --------------------------------
     *   File system usage monitoring
     * --------------------------------
     */
    public function fs_mon_def () {
        $list = array () ;
        $sql = "SELECT * FROM {$this->database}.fs_mon_def ORDER BY 'group', 'name'" ;
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            array_push($list, array (
                'id'    => intval($attr['id']) ,
                'group' => $attr['group'] ,
                'name'  => $attr['name']
            )) ;
        }
        return $list ;
    }
    public function fs_mon_def_by_id ($id) {
        $id = intval($id) ;
        $sql = "SELECT * FROM {$this->database}.fs_mon_def WHERE id={$id}" ;        
        $result = $this->query($sql) ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows != 1) throw new SysMonException (
            __class__.'::'.__METHOD__ ,
            'inconsistent result returned from the database. Wrong schema?') ;
        $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
        return array (
            'id'    => intval($attr['id']) ,
            'group' =>        $attr['group'] ,
            'name'  =>        $attr['name']
        ) ;
    }
    
    public function fs_mon_stat ($id) {
        $id = intval($id) ;
        $list = array () ;
        $sql = "SELECT  insert_time, used, available FROM {$this->database}.fs_mon_stat WHERE fs_id={$id} ORDER BY insert_time DESC" ;
        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            array_push($list, array (
                'insert_time' => new LusiTime (intval($attr['insert_time'])) ,
                'used'        =>               intval($attr['used']) ,
                'available'   =>               intval($attr['available'])
            )) ;
        }
        return $list ;
    }
    public function fs_mon_summary () {
        $list = array () ;
        foreach ($this->fs_mon_def() as $fs) {
            $fs_id = intval($fs['id']) ;
            $sql = "SELECT used, available FROM {$this->database}.fs_mon_stat WHERE fs_id={$fs_id} ORDER BY insert_time DESC LIMIT 1" ;
            $result = $this->query($sql) ;
            $nrows = mysql_numrows($result) ;
            if ($nrows == 0) continue;
            if ($nrows != 1) throw new SysMonException (
                __class__.'::'.__METHOD__ ,
                'inconsistent result returned from the database. Wrong schema?') ;
            $attr = mysql_fetch_array ($result, MYSQL_ASSOC) ;
            array_push($list, array (
                'id'        => intval($fs['id']) ,
                'group'     =>        $fs['group'] ,
                'name'      =>        $fs['name'] ,
                'used'      => intval($attr['used']) ,
                'available' => intval($attr['available'])
            )) ;
        }
        return $list ;
    }

    /* -------------------------------------
     *   File migraton delays subscription
     * -------------------------------------
     */

    /**
     * Find a subscription entry for delayed file migration for teh specified
     * user account. Return 'nul' if none is found.
     * 
     * @param string $uid
     * @return stdClass
     * @throws DataPortalException
     */
    public function find_fm_delay_subscriber ($uid) {
        $uid_escaped = $this->escape_string(trim($uid)) ;
        $sql = "SELECT * FROM {$this->database}.fm_delay_subscriber WHERE uid='{$uid_escaped}'" ;
        $result = $this->query($sql) ;
        $nrows = mysql_numrows($result) ;
        if (!$nrows) return null ;
        if ($nrows == 1)
            return $this->_fm_delay_subscriber2obj (
                mysql_fetch_array (
                    $result ,
                    MYSQL_ASSOC)) ;

        throw new DataPortalException (
            __METHOD__ ,
            "duplicate entries for migration subscriber: {$uid} in database.") ;
    }

    /* Get all known subscribers for delayed migration notifications.
     * 
     * The method will return an array (a list) of entries similar to
     * the ones reported by the previous method.
     */

    public function fm_delay_subscribers () {
        $sql = "SELECT * FROM {$this->database}.fm_delay_subscriber" ;
        $result = $this->query($sql) ;
        $list = array();
        for ($nrows = mysql_numrows($result), $i = 0; $i < $nrows; $i++)
            array_push (
                $list ,
                $this->_fm_delay_subscriber2obj (
                    mysql_fetch_array (
                        $result ,
                        MYSQL_ASSOC))) ;

        return $list ;
    }

    /**
     * Return an object representing a subscription entry. The object is
     * ready to be serialized into JSON.
     *
     *   Key             | Type      | Description
     *   ----------------+-----------+----------------------------------------------------------------------------
     *   .uid             | string   | user account of a person subscribed
     *   .gecos           | string   | full user name fr the user account (if available)
     *   .subscribed_uid  | string   | user account of a person who requested the subscription 
     *   .subscribed_time | stdClass | time when the subscription was made:
     *     .sec           | integer  | - the nuumber of seconds
     *     .nsec          | integer  | - the nuumber of nanoseconds
     *     .day           | string   | - the day like 2014-04-12
     *     .hms           | string   | - the hour-minute-second like 20:23:05
     *   .subscribed_host | string   | host (IP address or DNS name) name from which the operation was requested
     *   .instr           | string   | the name of an instrument (optional)
     *   .last_sec        | integer  | the number of last seconds to take into account
     *   .delay_sec       | integer  | the minimum duration of delays to take into account
     *
     * @param stdClass $row
     */
    private function _fm_delay_subscriber2obj ($row) {

        $obj = new \stdClass ;
        $obj->uid = trim($row['uid']) ;

        RegDB::instance()->begin() ;
        $user = RegDB::instance()->find_user_account($obj->uid) ;
        $obj->gecos = $user ? $user['gecos'] : $obj->uid ;

        $obj->subscribed_uid = trim($row['subscribed_uid']) ;
        $subscribed_user = RegDB::instance()->find_user_account($obj->subscribed_uid) ;
        $obj->subscribed_gecos = $subscribed_user ? $subscribed_user['gecos'] : $obj->subscribed_uid ;

        $subscribed_time = LusiTime::from64(trim($row['subscribed_time'])) ;
        $obj->subscribed_time = new \stdClass ;
        $obj->subscribed_time->sec  = $subscribed_time->sec ;
        $obj->subscribed_time->nsec = $subscribed_time->nsec ;
        $obj->subscribed_time->day  = $subscribed_time->toStringDay() ;
        $obj->subscribed_time->hms  = $subscribed_time->toStringHMS() ;
        $obj->subscribed_host = trim($row['subscribed_host']) ;
        $obj->instr = is_null($row['instr']) ? '' : strtoupper(trim($row['instr'])) ;
        $obj->last_sec = intval($row['last_sec']) ;
        $obj->delay_sec = intval($row['delay_sec']) ;
        return $obj ;
    }
    
    /**
     * Register a new subscriber and return an object describing
     * the newely made subscripton.
     *
     * @param string $uid
     * @param string $subscribed_uid
     * @param string $instr
     * @param integer $last_sec
     * @param integer $delay_sec
     * @return stdClass
     */
    public function add_fm_delay_subscriber (
        $uid ,
        $subscribed_uid ,
        $instr ,
        $last_sec ,
        $delay_sec) {

        $subscribed_time_64 = LusiTime::now()->to64() ;

        $uid_escaped             = $this->escape_string(trim($uid)) ;
        $subscribed_uid_escaped  = $this->escape_string(trim($subscribed_uid)) ;
        $subscribed_host_escaped = $this->escape_string(trim(AuthDB::instance()->authRemoteAddr())) ;
        $instr_escaped           = $this->escape_string(trim($instr)) ;

        $last_sec  = intval($last_sec) ;
        $delay_sec = intval($delay_sec) ;
        
        $sql = <<<HERE
INSERT INTO {$this->database}.fm_delay_subscriber VALUES (
    '{$uid_escaped}' ,
    '{$subscribed_uid_escaped}' ,
    {$subscribed_time_64} ,
    '{$subscribed_host_escaped}' ,
    '{$instr_escaped}' ,
    {$last_sec} ,
    {$delay_sec}
)
HERE;
        $this->query($sql) ;

        return $this->find_fm_delay_subscriber ($uid) ;
    }

    /**
     * Update an entry of an existing subscriber and return an object
     * describing the current status of the subscripton.
     *
     * @param string $uid
     * @param string $subscribed_uid
     * @param string $instr
     * @param integer $last_sec
     * @param integer $delay_sec
     * @return stdClass

     */
    public function update_fm_delay_subscriber (
        $uid ,
        $subscribed_uid ,
        $instr ,
        $last_sec ,
        $delay_sec) {

        $subscribed_time_64 = LusiTime::now()->to64() ;

        $uid_escaped             = $this->escape_string(trim($uid)) ;
        $subscribed_uid_escaped  = $this->escape_string(trim($subscribed_uid)) ;
        $subscribed_host_escaped = $this->escape_string(trim(AuthDB::instance()->authRemoteAddr())) ;
        $instr_escaped           = $this->escape_string(trim($instr)) ;

        $last_sec  = intval($last_sec) ;
        $delay_sec = intval($delay_sec) ;
        
        $sql = <<<HERE
UPDATE {$this->database}.fm_delay_subscriber SET
    subscribed_uid='{$subscribed_uid_escaped}' ,
    subscribed_time={$subscribed_time_64} ,
    subscribed_host='{$subscribed_host_escaped}' ,
    instr='{$instr_escaped}' ,
    last_sec={$last_sec} ,
    delay_sec={$delay_sec}
  WHERE
    uid='{$uid_escaped}'
HERE;
        $this->query($sql) ;

        return $this->find_fm_delay_subscriber ($uid) ;
    }

    /**
     * Remove any subscriptions for the specified user
     *
     * @param string $uid
     */
    public function remove_fm_delay_subscriber ($uid) {
        $uid_escaped = $this->escape_string(trim($uid)) ;
        $sql = "DELETE FROM {$this->database}.fm_delay_subscriber WHERE uid='{$uid_escaped}'" ;
        $this->query($sql) ;
    }
}

?>
