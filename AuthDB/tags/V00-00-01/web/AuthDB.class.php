<?php

require_once( 'RegDB/RegDB.inc.php' );

/**
 * Class AuthDB encapsulates operations with the 'roles' database
 *
 * @author gapon
 */
class AuthDB {

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
            new LogBookConnection (
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

        $sql = "SELECT user.user,user.exp_id,role.* FROM user,role WHERE ((user.exp_id IS NULL) OR (user.exp_id={$exper_id})) AND user.role_id=role.id ORDER BY role.app,role.name";
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

        $sql = "SELECT user.user,user.exp_id,role.* FROM user,role WHERE role.id={$role_id} AND user.role_id=role.id ORDER BY user.exp_id, role.app";
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

        $sql = "SELECT * FROM role WHERE app='{$application}' ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push ( $list, mysql_fetch_array( $result, MYSQL_ASSOC ) );

        return $list;
    }

    public function find_role( $application, $role ) {

        $sql = "SELECT * FROM role WHERE app='{$application}' AND name='{$role}'";
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

        $sql = "SELECT DISTINCT app FROM role ORDER BY app";
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

        $sql = "SELECT name FROM priv WHERE role_id={$role_id} ORDER BY name";
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
            "SELECT * FROM user u, role r".
            " WHERE r.name='{$role}' AND r.app='{$app}'".
            " AND u.user='{$user}' AND u.role_id=r.id".
            " AND u.exp_id={$exper_id}";
        $result = $this->connection->query ( $sql );

        $nrows = mysql_numrows( $result );
        return $nrows >= 1;
    }

    public function hasPrivilege( $user, $exper_id, $app, $priv ) {

        //return true ;

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

    private function hasPrivilegeImpl( $user, $exper_id, $app, $priv ) {

        $sql =
            "SELECT * FROM user u, role r, priv p".
            " WHERE p.name='{$priv}' AND p.role_id=r.id AND r.app='{$app}'".
            " AND u.user='{$user}' AND u.role_id=r.id".
            " AND u.exp_id={$exper_id}";
        $result = $this->connection->query ( $sql );

        $nrows = mysql_numrows( $result );
        return $nrows >= 1;
    }

    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function createRole( $application, $role, $privileges ) {
        $this->connection->query ( "INSERT INTO role VALUES(NULL,'{$role}','{$application}')" );
        $result = $this->connection->query ( "SELECT LAST_INSERT_ID() AS 'role_id'" );
        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new AuthDBException (
                __METHOD__,
                "inconsistent result of a query. The database may be corrupted." );
        $row = mysql_fetch_array( $result, MYSQL_ASSOC );
        $role_id = $row['role_id'];
        foreach( $privileges as $p) {
            $this->connection->query( "INSERT INTO priv VALUES(NULL,'{$p}',{$role_id})" );
        }
    }

    public function deleteRole( $id ) {
        $this->connection->query ( "DELETE FROM priv WHERE role_id={$id}" );
        $this->connection->query ( "DELETE FROM user WHERE role_id={$id}" );
        $this->connection->query ( "DELETE FROM role WHERE id={$id}" );
    }

    public function deleteApplication( $name ) {
        $this->connection->query ( "DELETE FROM priv WHERE role_id=(SELECT id FROM role WHERE app='{$name}')" );
        $this->connection->query ( "DELETE FROM user WHERE role_id=(SELECT id FROM role WHERE app='{$name}')" );
        $this->connection->query ( "DELETE FROM role WHERE app='{$name}'" );
    }

    public function createRolePlayer( $application, $role, $exper_id, $player ) {
        $exper_id_attr = is_null( $exper_id ) ? 'NULL' : $exper_id;
        $this->connection->query ( "INSERT INTO user VALUES(NULL,{$exper_id_attr},'{$player}',(SELECT id FROM role WHERE name='{$role}' AND app='{$application}'))" );
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

function toYesNo( $boolean_val ) { return '<b>'.( $boolean_val ? 'Yes' : 'No').'</b>'; }

try {
    $authdb = new AuthDB();
    $authdb->begin();

    $user = 'gapon';

    print( "<h1>privileges of user '{$user}' for 'LogBook' of experiment '53'</h1>" );
    print( "<br>'read: : ".toYesNo( $authdb->hasPrivilege( $user, 53, 'LogBook', 'read' )));
    print( "<br>'post: : ".toYesNo( $authdb->hasPrivilege( $user, 53, 'LogBook', 'post' )));
    print( "<br>'edit: : ".toYesNo( $authdb->hasPrivilege( $user, 53, 'LogBook', 'edit' )));
    print( "<br>'delete: : ".toYesNo( $authdb->hasPrivilege( $user, 53, 'LogBook', 'delete' )));
    print( "<br>'manage_shifts: : ".toYesNo( $authdb->hasPrivilege( $user, 53, 'LogBook', 'manage_shifts' )));

    $authdb->commit();

} catch( AuthDBException $e ) {
    print( $e->toHtml());
}

 *
 */
?>
