<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use \AuthDB\AuthDB ;
use \FileMgr\DbConnection ;

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
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." ) ;
        return new IrepUser (
            $this ,
            mysql_fetch_array($result, MYSQL_ASSOC)) ;
    }
}

?>
