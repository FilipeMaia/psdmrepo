<?php

require_once( 'RegDB/RegDB.inc.php' );

/**
 * Class AuthDB encapsulates operations with the 'roles' database
 *
 * @author gapon
 */
class AuthDB {

    // ---------------------------------------------------
    // --- SIMPLIFIED INTERFACE AND ITS IMPLEMENTATION ---
    // ---------------------------------------------------

    private static $instance = null;

    /**
     * Singleton to simplify certain operations.
     *
     * @return unknown_type
     */
    public static function instance() {
        if( is_null( AuthDB::$instance )) AuthDB::$instance = new AuthDB();
        return AuthDB::$instance;
    }

    public function authName() {
        return $_SERVER['REMOTE_USER'];
    }

    public function authType() {
        return $_SERVER['AUTH_TYPE'];
    }

    public function isAuthenticated() {
        return AuthDB::instance()->authName() != '';
    }

    public function canRead() {
    	// Anyone who's been authenticated can read the contents of
    	// this database.
    	//
        return $this->isAuthenticated();
    }

    public function canEdit() {
        if( !$this->isAuthenticated()) return false;
        $this->begin();
        return $this->hasRole(
            $this->authName(), null, 'RoleDB', 'Admin' );
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

    // -----------------------------------------
    // --- CORE CLASS AND ITS IMPLEMENTATION ---
    // -----------------------------------------
    
    /* Data members
     */
    private $connection;
    private $regdb;

    /* Constructor
     *
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     */
    public function __construct (
        $host     = null,
        $user     = null,
        $password = null,
        $database = null ) {

        $this->connection =
            new AuthDBConnection (
                is_null($host)     ? AUTHDB_DEFAULT_HOST : $host,
                is_null($user)     ? AUTHDB_DEFAULT_USER : $user,
                is_null($password) ? AUTHDB_DEFAULT_PASSWORD : $password,
                is_null($database) ? AUTHDB_DEFAULT_DATABASE : $database);

        $this->regdb = new RegDB();
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

    /*
     * =================================
     *   INFORMATION REQUEST OPERATION
     * =================================
     */
    public function roles( $exper_id ) {

        $list = array();

        $sql = "SELECT user.user,user.exp_id,role.* FROM {$this->connection->database}.user, {$this->connection->database}.role WHERE ((user.exp_id IS NULL) OR (user.exp_id={$exper_id})) AND user.role_id=role.id ORDER BY role.app,role.name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new AuthDBRole (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function roles_by_id( $role_id ) {

        $list = array();

        $sql = "SELECT user.user,user.exp_id,role.* FROM {$this->connection->database}.user, {$this->connection->database}.role WHERE role.id={$role_id} AND user.role_id=role.id ORDER BY user.exp_id, role.app";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new AuthDBRole (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function roles_by_application( $application ) {

        $list = array();

        $sql = "SELECT * FROM {$this->connection->database}.role WHERE app='{$application}' ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push ( $list, mysql_fetch_array( $result, MYSQL_ASSOC ) );

        return $list;
    }

    public function find_role( $application, $role ) {

        $sql = "SELECT * FROM {$this->connection->database}.role WHERE app='{$application}' AND name='{$role}'";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 ) return mysql_fetch_array( $result, MYSQL_ASSOC );
        throw new AuthDBException (
            __METHOD__,
            "inconsister results returned from the database. The database may be corrupted." );
    }


    public function applications() {

        $list = array();

        $sql = "SELECT DISTINCT app FROM {$this->connection->database}.role ORDER BY app";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $row = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push ( $list, $row['app'] );
        }
        return $list;
    }


    public function role_privileges( $role_id ) {

        $list = array();

        $sql = "SELECT name FROM {$this->connection->database}.priv WHERE role_id={$role_id} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $row = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push ( $list, $row['name'] );
        }
        return $list;
    }

    public function hasRole( $user, $exper_id, $app, $role ) {

        //return true ;

        $sql =
            "SELECT * FROM {$this->connection->database}.user u, {$this->connection->database}.role r".
            " WHERE r.name='{$role}' AND r.app='{$app}'".
            " AND u.user='{$user}' AND u.role_id=r.id".
            ( is_null($exper_id) ? "" : " AND u.exp_id={$exper_id}" );
        $result = $this->connection->query ( $sql );

        $nrows = mysql_numrows( $result );
        return $nrows >= 1;
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
        $users = $this->regdb->user_accounts( $user );
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
        $users = $this->regdb->user_accounts( $user );
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

        // Check the primary group of the user
        //
        $user_account = $this->regdb->find_user_account( $user );
        if( is_null( $user_account ))
            throw new AuthDBException (
                __METHOD__,
                "no such user: {$user}" );
        $primary_group = $user_account['gid'];
        foreach( $authorized_groups as $g ) {

        	if( $g == $primary_group ) return true;

        	// ATTENTION: Note 'false' as the last parameter of the method
        	// called below. Setting its value to 'true' would impose a significant
        	// performance penalty! Besides, we've already know the user's primary group name
        	// and the test for it failed above above.
        	//
        	$group_members = $this->regdb->posix_group_members ( $g, /*$and_as_primary_group=*/ false );
        	foreach( $group_members as $m ) {
        	    if( $m['uid'] == $user ) return true;
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
            "SELECT * FROM {$this->connection->database}.user u, {$this->connection->database}.role r, {$this->connection->database}.priv p".
            " WHERE p.name='{$priv}' AND p.role_id=r.id AND r.app='{$app}'".
            " AND u.user='{$user}' AND u.role_id=r.id".
            (is_null($exper_id) ? "" : " AND ((u.exp_id={$exper_id}) OR (u.exp_id IS NULL))");
        $result = $this->connection->query ( $sql );

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

    	$user_u = $this->connection->database.".user u";
    	$role_r = $this->connection->database.".role r";
    	$priv_p = $this->connection->database.".priv p";

        $exper_id_attr = is_null( $exper_id ) ? 'NULL' : $exper_id;
    	$sql = "SELECT DISTINCT u.user FROM {$user_u},{$role_r} WHERE u.role_id IN (SELECT r.id FROM {$role_r},{$priv_p} WHERE r.id=p.role_id AND r.app='{$app}' AND p.name='{$priv}') AND ((u.exp_id IS NULL) OR (u.exp_id={$exper_id_attr})) AND u.role_id=r.id ORDER BY u.user";
        $result = $this->connection->query ( $sql );
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
        $this->connection->query ( "INSERT INTO {$this->connection->database}.role VALUES(NULL,'{$role}','{$application}')" );
        $result = $this->connection->query ( "SELECT LAST_INSERT_ID() AS 'role_id'" );
        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new AuthDBException (
                __METHOD__,
                "inconsistent result of a query. The database may be corrupted." );
        $row = mysql_fetch_array( $result, MYSQL_ASSOC );
        $role_id = $row['role_id'];
        foreach( $privileges as $p) {
            $this->connection->query( "INSERT INTO {$this->connection->database}.priv VALUES(NULL,'{$p}',{$role_id})" );
        }
    }

    public function deleteRole( $id ) {
        $this->connection->query ( "DELETE FROM {$this->connection->database}.priv WHERE role_id={$id}" );
        $this->connection->query ( "DELETE FROM {$this->connection->database}.user WHERE role_id={$id}" );
        $this->connection->query ( "DELETE FROM {$this->connection->database}.role WHERE id={$id}" );
    }

    public function deleteApplication( $name ) {
        $this->connection->query ( "DELETE FROM {$this->connection->database}.priv WHERE role_id=(SELECT id FROM role WHERE app='{$name}')" );
        $this->connection->query ( "DELETE FROM {$this->connection->database}.user WHERE role_id=(SELECT id FROM role WHERE app='{$name}')" );
        $this->connection->query ( "DELETE FROM {$this->connection->database}.role WHERE app='{$name}'" );
    }

    public function createRolePlayer( $application, $role, $exper_id, $player ) {
        $exper_id_attr = is_null( $exper_id ) ? 'NULL' : $exper_id;
        $this->connection->query ( "INSERT INTO {$this->connection->database}.user VALUES(NULL,{$exper_id_attr},'{$player}',(SELECT id FROM {$this->connection->database}.role WHERE name='{$role}' AND app='{$application}'))" );
    }
}

/*
 * Unit tests
 *
require_once( 'AuthDB/AuthDB.inc.php' );
try {
    $authdb = new AuthDB();
    $authdb->begin();

    $roles = $authdb->roles_by_id(12);
    print_r($roles);

    $authdb->commit();

} catch( AuthDBException $e ) {
    print( $e->toHtml());
}
*/

/*
 * Unit tests
 *
require_once( 'AuthDB/AuthDB.inc.php' );
require_once( 'RegDB/RegDB.inc.php' );

function toYesNo( $boolean_val ) { return '<b>'.( $boolean_val ? 'Yes' : 'No').'</b>'; }

function resolve_exper_id( $instrument_name, $experiment_name ) {
    $regdb = new RegDB();
	$regdb->begin();
    $experiment = $regdb->find_experiment( $instrument_name, $experiment_name )
        or die( "no such experiment" );
    $exper_id = $experiment->id();
    $regdb->commit();
    return $exper_id;
}
try {
    $authdb = new AuthDB();
    $authdb->begin();

    $exper_id = resolve_exper_id( 'AMO', 'amo02709');
    $user = 'dimauro' ; //'gapon';

    print( "<h1>privileges of user '{$user}' for 'LogBook' of experiment {$exper_id}</h1>" );
    print( "<br>'read: : ".toYesNo( $authdb->hasPrivilege( $user, $exper_id, 'LogBook', 'read' )));
    print( "<br>'post: : ".toYesNo( $authdb->hasPrivilege( $user, $exper_id, 'LogBook', 'post' )));
    print( "<br>'edit: : ".toYesNo( $authdb->hasPrivilege( $user, $exper_id, 'LogBook', 'edit' )));
    print( "<br>'delete: : ".toYesNo( $authdb->hasPrivilege( $user, $exper_id, 'LogBook', 'delete' )));
    print( "<br>'manage_shifts: : ".toYesNo( $authdb->hasPrivilege( $user, $exper_id, 'LogBook', 'manage_shifts' )));

    print( "<br>hoHasPrivilege( {$exper_id}, 'LogBook', 'read' ): " );
    $users = $authdb->whoHasPrivilege( $exper_id, 'LogBook', 'read' );
    print_r( $users );

    print( "<br>hoHasPrivilege( {$exper_id}, 'LogBook', 'post' ): " );
    $users = $authdb->whoHasPrivilege( $exper_id, 'LogBook', 'post' );
    print_r( $users );

    print( "<br>hoHasPrivilege( {$exper_id}, 'LogBook', 'manage_shifts' ): " );
    $users = $authdb->whoHasPrivilege( $exper_id, 'LogBook', 'manage_shifts' );
    print_r( $users );
    
    print( "<br>hoHasPrivilege( {$exper_id}, 'LogBook', 'edit' ): " );
    $users = $authdb->whoHasPrivilege( $exper_id, 'LogBook', 'edit' );
    print_r( $users );
    
    print( "<br>hoHasPrivilege( {$exper_id}, 'LogBook', 'delete' ): " );
    $users = $authdb->whoHasPrivilege( $exper_id, 'LogBook', 'delete' );
    print_r( $users );

    $authdb->commit();

} catch( AuthDBException $e ) {
    print( $e->toHtml());
} catch( RegDBException $e ) {
    print( $e->toHtml());
}

 *
 */
?>
