<?php

namespace AuthDB;

require_once( 'authdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use FileMgr\DbConnection;

use RegDB\RegDB;

/**
 * Class AuthDB encapsulates operations with the 'roles' database
 *
 * @author gapon
 */
class AuthDB extends DbConnection {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $instance = null;

    /**
     * Singleton to simplify certain operations.
     *
     * @return AuthDB
     */
    public static function instance() {
        if( is_null( AuthDB::$instance )) AuthDB::$instance =
            new AuthDB(
                AUTHDB_DEFAULT_HOST,
                AUTHDB_DEFAULT_USER,
                AUTHDB_DEFAULT_PASSWORD,
                AUTHDB_DEFAULT_DATABASE);
        return AuthDB::$instance;
    }

    public static function reporErrorHtml( $message, $link=null ) {
        $suggested_back_link =
            is_null($link) ?
            'the <b>BACK</b> button of your browser' :
            '<a href="'.$link.'">here</a>';
        return <<<HERE
<center>
  <br>
  <br>
  <div style="background-color:#f0f0f0; border:solid 2px red; max-width:640px;">
    <h1 style="color:red;">Authorization Error</h1>
    <div style="height:2px; background-color:red;"></div>
    <p>{$message}</p>
    <p>Click {$suggested_back_link} to return to the previous context</p>
  </div>
</center>
HERE;
    }

    /* Constructor
     *
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     */
    public function __construct ($host, $user, $password, $database) {
        parent::__construct ( $host, $user, $password, $database );
    }

    /*
     * ====================================
     *   AUTHENTICATION REQUEST OPERATION
     * ====================================
     */
    public function authName       () { return $_SERVER['REMOTE_USER']; }
    public function authType       () { return $_SERVER['AUTH_TYPE']; }
    public function authRemoteAddr () { return $_SERVER['REMOTE_ADDR']; }
    public function isAuthenticated() { return AuthDB::instance()->authName() != ''; }

    /*
     * =========================================
     *   AUTHORIZATION REQUESTS AND OPERATIONS
     * =========================================
     */
    public function canRead() {
    	// Anyone who's been authenticated can read the contents of
    	// this database.
    	//
        return $this->isAuthenticated();
    }

    public function canEdit() {
        if( !$this->isAuthenticated()) return false;
        $this->begin();
        return $this->hasRole ($this->authName(), null, 'RoleDB', 'Admin' );
    }

    public function roles( $exper_id ) {

        $list = array();

        $sql = "SELECT user.user,user.exp_id,role.* FROM {$this->database}.user, {$this->database}.role WHERE ((user.exp_id IS NULL) OR (user.exp_id={$exper_id})) AND user.role_id=role.id ORDER BY role.app,role.name";
        $result = $this->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new AuthDBRole ($this, mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    /**
     * Get user roles in a context of instruments/experiments.
     * 
     * All parameters to the method are optional. The default value (null) of
     * the parameter implies all possible values of the paramer. The result is
     * returned as a list of triplets sorted by instrument and experiment names.
     * 
     * @param string $user
     * @param string $application
     * @param string $instrument
     * @param string $experiment
     * @return array( 'instr'=> string, 'instr' => string, 'role' => AuthDBRole )
     */
    public function roles_by( $user=null, $application=null, $instrument=null, $experiment=null ) {

    	$user_selector        = !$user        ? '' : " AND u.user='{$user}'";
    	$application_selector = !$application ? '' : " AND r.app='{$application}'";
    	$instrument_selector  = !$instrument  ? '' : " AND i.name='{$instrument}'";
    	$experiment_selector  = !$experiment  ? '' : " AND e.name='{$experiment}'";

    	$sql =<<<HERE

SELECT u.user AS 'user',
       r.id   AS 'id',
       r.name AS 'name',
       r.app  AS 'app',
       i.name AS 'instr',
       e.name AS 'exper',
       e.id   AS 'exper_id'

       FROM user `u`,
            role `r`,
            regdb.experiment `e`,
            regdb.instrument `i`

       WHERE u.role_id IN (SELECT id FROM role)
         AND u.role_id=r.id
         AND u.exp_id=e.id
         AND i.id=e.instr_id
         {$user_selector}
         {$application_selector}
         {$instrument_selector}
         {$experiment_selector}

UNION

SELECT u.user,
       r.id,
       r.name,
       r.app,
       '*',
       '*',
       '*'

       FROM user `u`,
            role `r`

       WHERE r.id IN (SELECT id FROM role)
         AND u.role_id=r.id
         AND u.exp_id IS NULL
         {$user_selector}
         {$application_selector}

ORDER BY instr,exper,user,app,name

HERE;

        $result = $this->query ( $sql );
		$nrows = mysql_numrows( $result );
    	$list = array();
        for( $i = 0; $i < $nrows; $i++ ) {
        	$attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push (
                $list,
                array (
                    'instr'    => $attr['instr'   ],
                    'exper'    => $attr['exper'   ],
                    'exper_id' => $attr['exper_id'],
                    'role'     => new AuthDBRole ($this, $attr)
                )
            );
        }
        return $list;
    }

    public function roles_by_id( $role_id ) {

        $list = array();

        $sql = "SELECT user.user,user.exp_id,role.* FROM {$this->database}.user, {$this->database}.role WHERE role.id={$role_id} AND user.role_id=role.id ORDER BY user.exp_id, role.app";
        $result = $this->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new AuthDBRole ($this, mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function roles_by_application( $application ) {

        $list = array();

        $sql = "SELECT * FROM {$this->database}.role WHERE app='{$application}' ORDER BY name";
        $result = $this->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push ( $list, mysql_fetch_array( $result, MYSQL_ASSOC ) );

        return $list;
    }

    public function find_role( $application, $role ) {

        $sql = "SELECT * FROM {$this->database}.role WHERE app='{$application}' AND name='{$role}'";
        $result = $this->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 ) return mysql_fetch_array( $result, MYSQL_ASSOC );
        throw new AuthDBException (
            __METHOD__,
            "inconsister results returned from the database. The database may be corrupted." );
    }


    public function applications() {

        $list = array();

        $sql = "SELECT DISTINCT app FROM {$this->database}.role ORDER BY app";
        $result = $this->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $row = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push ( $list, $row['app'] );
        }
        return $list;
    }


    public function role_privileges( $role_id ) {

        $list = array();

        $sql = "SELECT name FROM {$this->database}.priv WHERE role_id={$role_id} ORDER BY name";
        $result = $this->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $row = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push ( $list, $row['name'] );
        }
        return $list;
    }

    public function hasRole( $user, $exper_id, $app, $role ) {

        // First try to see if there is a direct record for the user account
        //
        if( $this->hasRoleImpl( $user, $exper_id, $app, $role )) return true;

        // If the specified user is actually a group (starts with 'gid:')
        // then we just quit becaus eth erest of thsi algorithm implies
        // a real user account not a group name.
        //
        if( substr($user, 0, 4) == 'gid:' ) return false;

        // Now try via the groups.
        //
        $authorized_groups = array();
        {
            $holders = $this->rolePlayers( $exper_id, $app, $role );
            foreach( $holders as $user_or_group ) {
                if( "gid:" == substr( $user_or_group, 0, 4 ))
                    array_push( $authorized_groups, substr( $user_or_group, 4 ));
            }
        }
        if( count( $authorized_groups ) <= 0 ) return false;

        // Check if the user has required role via one of groups
        // his/her account is member of.
        //
        RegDB::instance()->begin();
        $user_account = RegDB::instance()->find_user_account( $user );
        if( is_null( $user_account ))
            throw new AuthDBException (
                __METHOD__,
                "no such user: {$user}" );

        // TODO: We may have a better implemnetation of this algorithm
        //       instead of the nested loops as used below.
        //
        foreach( $authorized_groups as $g ) {
        	foreach( $user_account['groups'] as $ug ) {
        	    if( $ug == $g ) return true;
        	}
        }
        return false;
    }

    public function hasRoleImpl( $user, $exper_id, $app, $role ) {

        $sql =
            "SELECT * FROM {$this->database}.user u, {$this->database}.role r".
            " WHERE r.name='{$role}' AND r.app='{$app}'".
            " AND u.user='{$user}' AND u.role_id=r.id".
            ( is_null($exper_id) ? "" : " AND u.exp_id={$exper_id}" );
        $result = $this->query ( $sql );

        $nrows = mysql_numrows( $result );
        return $nrows >= 1;
    }

    /**
     * Find a list of users and/or groups who plays the specified role in
     * the specified context (experiment and application).
     *
     * @param integer $exper_id
     * @param string $app
     * @param string $role
     *
     * @return array() of user and/or group names
     */
    public function rolePlayers( $exper_id, $app, $role) {

    	// NOTE: Regardless of whether a valid experiment identifier is given
    	//       the requested roke player is also checked among those
    	//       role playes who're not associated with any particular experiment.
    	//
    	$list = array();

    	$user_u = $this->database.".user u";
    	$role_r = $this->database.".role r";

        $exper_id_attr = is_null( $exper_id ) ? 'NULL' : $exper_id;
    	$sql = "SELECT DISTINCT u.user FROM {$user_u},{$role_r} WHERE u.role_id IN (SELECT r.id FROM {$role_r} WHERE r.name='{$role}' AND r.app='{$app}') AND ((u.exp_id IS NULL) OR (u.exp_id={$exper_id_attr})) ORDER BY u.user";
        $result = $this->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $row = mysql_fetch_array( $result, MYSQL_ASSOC );
            $user = $row['user'];
            array_push ( $list, $user );
        }
        return $list;
    }

    public function hasPrivilege_inefficient( $user, $exper_id, $app, $priv ) {

        /* TODO: connect to RegDB and get all groups the user may belong to.
         * Then try to see if the privilege is granted either to that user directly,
         * or any group it belongs to. In the later case use the following syntax:
         *
         *   gid:<group_name>
         *
         * NOTE #1: Follow the same approach for other operations.
         * NOTE #2: Test the code.
         */
        RegDB::instance()->begin();
        $users = RegDB::instance()->user_accounts( $user );
        if( count( $users ) <= 0 )
                throw new AuthDBException (
                    __METHOD__,
                    "no such user: {$user}" );

        // Try the groups first because using groups based authorization is
        // the most desired approach.
        //
        $groups = $users[0]['groups'];
        foreach( $groups as $g )
            if( $this->hasPrivilegeImpl( "gid:{$g}", $exper_id, $app, $priv )) return true;

        // Finally try the user account directly
        //
        return $this->hasPrivilegeImpl( $user, $exper_id, $app, $priv );
    }

    public function hasPrivilege_more_efficient( $user, $exper_id, $app, $priv ) {

        // First try to see if there is a direct record for the user account
        //
        if( $this->hasPrivilegeImpl( $user, $exper_id, $app, $priv )) return true;

        // Now try the groups.
        //
        RegDB::instance()->begin();
        $users = RegDB::instance()->user_accounts( $user );
        if( count( $users ) <= 0 )
                throw new AuthDBException (
                    __METHOD__,
                    "no such user: {$user}" );

        $groups = $users[0]['groups'];
        foreach( $groups as $g )
            if( $this->hasPrivilegeImpl( "gid:{$g}", $exper_id, $app, $priv )) return true;

        return false;
    }

    public function hasPrivilege( $user, $exper_id, $app, $priv ) {

        // First try to see if there is a direct record for the user account
        //
        if( $this->hasPrivilegeImpl( $user, $exper_id, $app, $priv )) return true;

        // Now try via the groups.
        //
        $authorized_groups = array();
        {
            $holders = $this->whoHasPrivilege( $exper_id, $app, $priv, true );
            foreach( $holders as $user_or_group ) {
                if( "gid:" == substr( $user_or_group, 0, 4 ))
                    array_push( $authorized_groups, substr( $user_or_group, 4 ));
            }
        }
        if( count( $authorized_groups ) <= 0 ) return false;

        // Check if the user has required privilege via one of groups
        // his/her account is member of.
        //
        RegDB::instance()->begin();
        $user_account = RegDB::instance()->find_user_account( $user );
        if( is_null( $user_account ))
            throw new AuthDBException (
                __METHOD__,
                "no such user: {$user}" );

        // TODO: We may have a better implemnetation of this algorithm
        //       instead of the nested loops as used below.
        //
        foreach( $authorized_groups as $g ) {
        	foreach( $user_account['groups'] as $ug ) {
        	    if( $ug == $g ) return true;
        	}
        }
        return false;
    }

    private function hasPrivilegeImpl( $user, $exper_id, $app, $priv ) {

    	// NOTE: Regardless of whether a valid experiment identifier is given
    	//       the requested privilege holder is also checked among those
    	//       role playes who're not associated with any particular experiment.
    	//
        $sql =
            "SELECT * FROM {$this->database}.user u, {$this->database}.role r, {$this->database}.priv p".
            " WHERE p.name='{$priv}' AND p.role_id=r.id AND r.app='{$app}'".
            " AND u.user='{$user}' AND u.role_id=r.id".
            (is_null($exper_id) ? " AND u.exp_id IS NULL" : " AND ((u.exp_id={$exper_id}) OR (u.exp_id IS NULL))");
        $result = $this->query ( $sql );

        $nrows = mysql_numrows( $result );
        return $nrows >= 1;
    }

    /**
     * Find a list of users and/or groups who has the specified privilege in
     * the specified context (experiment and application).
     *
     * @param integer $exper_id
     * @param sting $app
     * @param string $priv
     *
     * @return array() of user and/or group names
     */
    public function whoHasPrivilege( $exper_id, $app, $priv) {

    	// NOTE: Regardless of whether a valid experiment identifier is given
    	//       the requested privilege holder is also checked among those
    	//       role playes who're not associated with any particular experiment.
    	//
    	$list = array();

    	$user_u = $this->database.".user u";
    	$role_r = $this->database.".role r";
    	$priv_p = $this->database.".priv p";

        $exper_id_attr = is_null( $exper_id ) ? 'NULL' : $exper_id;
    	$sql = "SELECT DISTINCT u.user FROM {$user_u},{$role_r} WHERE u.role_id IN (SELECT r.id FROM {$role_r},{$priv_p} WHERE r.id=p.role_id AND r.app='{$app}' AND p.name='{$priv}') AND ((u.exp_id IS NULL) OR (u.exp_id={$exper_id_attr})) AND u.role_id=r.id ORDER BY u.user";
        $result = $this->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $row = mysql_fetch_array( $result, MYSQL_ASSOC );
            $user = $row['user'];
            array_push ( $list, $user );
        }
        return $list;
    }

    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function createRole( $application, $role, $privileges ) {
        $this->query ( "INSERT INTO {$this->database}.role VALUES(NULL,'{$role}','{$application}')" );
        $result = $this->query ( "SELECT LAST_INSERT_ID() AS 'role_id'" );
        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new AuthDBException (
                __METHOD__,
                "inconsistent result of a query. The database may be corrupted." );
        $row = mysql_fetch_array( $result, MYSQL_ASSOC );
        $role_id = $row['role_id'];
        foreach( $privileges as $p) {
            $this->query( "INSERT INTO {$this->database}.priv VALUES(NULL,'{$p}',{$role_id})" );
        }
    }

    public function deleteRole( $id ) {
        $this->query ( "DELETE FROM {$this->database}.priv WHERE role_id={$id}" );
        $this->query ( "DELETE FROM {$this->database}.user WHERE role_id={$id}" );
        $this->query ( "DELETE FROM {$this->database}.role WHERE id={$id}" );
    }

    public function deleteApplication( $name ) {
        $this->query ( "DELETE FROM {$this->database}.priv WHERE role_id=(SELECT id FROM role WHERE app='{$name}')" );
        $this->query ( "DELETE FROM {$this->database}.user WHERE role_id=(SELECT id FROM role WHERE app='{$name}')" );
        $this->query ( "DELETE FROM {$this->database}.role WHERE app='{$name}'" );
    }

    public function createRolePlayer( $application, $role, $exper_id, $player ) {
        $exper_id_attr = is_null( $exper_id ) ? 'NULL' : $exper_id;
        $this->query ( "INSERT INTO {$this->database}.user VALUES(NULL,{$exper_id_attr},'{$player}',(SELECT id FROM {$this->database}.role WHERE name='{$role}' AND app='{$application}'))" );
    }

    public function deleteRolePlayer( $application, $role, $exper_id, $player ) {
        $exper_id_attr = is_null( $exper_id ) ? 'IS NULL' : "={$exper_id}";
        $sql = "DELETE FROM {$this->database}.user ".
        	   "WHERE user='{$player}' ".
               "AND   role_id=(SELECT id FROM {$this->database}.role WHERE name='{$role}' AND app='{$application}') ".
               "AND   exp_id {$exper_id_attr}";
        $this->query( $sql );
    }
}

/*
 * Unit tests
 *

function toYesNo( $boolean_val ) { return '<b>'.( $boolean_val ? 'Yes' : 'No').'</b>'; }
try {
    AuthDB::instance()->begin();

    print( "<br>has LogBook privelege: : ".toYesNo( AuthDB::instance()->hasPrivilege( 'rolles', 86, 'LogBook', 'read' )));
    print( "<br>has LDAP privelege: : ".toYesNo( AuthDB::instance()->hasPrivilege( 'rolles', null, 'LDAP', 'manage_groups' )));
    
    $roles = AuthDB::instance()->roles_by( 'xppopr', 'LogBook', 'XPP' );
    print_r($roles);

    AuthDB::instance()->commit();

} catch( AuthDBException $e ) { print( $e->toHtml()); }
 
*/

/*
 * Unit tests
 *

use RegDB\RegDBException;

function resolve_exper_id( $instrument_name, $experiment_name ) {
    RegDB::instance()->begin();
    $experiment = RegDB::instance()->find_experiment( $instrument_name, $experiment_name )
        or die( "no such experiment" );
    $exper_id = $experiment->id();
    RegDB::instance()->commit();
    return $exper_id;
}
try {
    AuthDB::instance()->begin();

    $exper_id = resolve_exper_id( 'SXR', 'sxrcom10');
    $user = 'sxropr'; //'maad';

    print( "<h1>privileges of user '{$user}' for 'LogBook' of experiment {$exper_id}</h1>" );
    print( "<br>'read: : ".toYesNo( AuthDB::instance()->hasPrivilege( $user, $exper_id, 'LogBook', 'read' )));
    print( "<br>'post: : ".toYesNo( AuthDB::instance()->hasPrivilege( $user, $exper_id, 'LogBook', 'post' )));
    print( "<br>'edit: : ".toYesNo( AuthDB::instance()->hasPrivilege( $user, $exper_id, 'LogBook', 'edit' )));
    print( "<br>'delete: : ".toYesNo( AuthDB::instance()->hasPrivilege( $user, $exper_id, 'LogBook', 'delete' )));
    print( "<br>'manage_shifts: : ".toYesNo( AuthDB::instance()->hasPrivilege( $user, $exper_id, 'LogBook', 'manage_shifts' )));

    print( "<br>whoHasPrivilege( {$exper_id}, 'LogBook', 'read' ): " );
    $users = AuthDB::instance()->whoHasPrivilege( $exper_id, 'LogBook', 'read' );
    print_r( $users );

    print( "<br>whoHasPrivilege( {$exper_id}, 'LogBook', 'post' ): " );
    $users = AuthDB::instance()->whoHasPrivilege( $exper_id, 'LogBook', 'post' );
    print_r( $users );

    print( "<br>whoHasPrivilege( {$exper_id}, 'LogBook', 'manage_shifts' ): " );
    $users = AuthDB::instance()->whoHasPrivilege( $exper_id, 'LogBook', 'manage_shifts' );
    print_r( $users );
    
    print( "<br>whoHasPrivilege( {$exper_id}, 'LogBook', 'edit' ): " );
    $users = AuthDB::instance()->whoHasPrivilege( $exper_id, 'LogBook', 'edit' );
    print_r( $users );
    
    print( "<br>whoHasPrivilege( {$exper_id}, 'LogBook', 'delete' ): " );
    $users = AuthDB::instance()->whoHasPrivilege( $exper_id, 'LogBook', 'delete' );
    print_r( $users );

    AuthDB::instance()->commit();

} catch( AuthDBException $e ) { print( $e->toHtml()); }
  catch( RegDBException  $e ) { print( $e->toHtml()); }

*/
?>
