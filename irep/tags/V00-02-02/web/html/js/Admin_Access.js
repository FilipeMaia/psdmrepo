define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../irep/css/Admin_Access.css') ;

    /**
     * The application for managing access privileges
     *
     * @returns {Admin_Access}
     */
    function Admin_Access (app_config) {

        var roles = {
            'ADMINISTRATOR': { is_administrator: true,  can_edit_inventory: true } ,
            'EDITOR'       : { is_administrator: false, can_edit_inventory: true } ,
            'OTHER'        : { is_administrator: false, can_edit_inventory: false}
        } ;

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this.on_update() ;
        } ;

        this.on_deactivate = function() {
            this._init() ;
        } ;

        // Automatically refresh the page at specified interval only

        this._update_ival_sec = 10 ;
        this._prev_update_sec = 0 ;

        this.on_update = function () {
            if (this.active) {
                var now_sec = Fwk.now().sec ;
                if (now_sec - this._prev_update_sec > this._update_ival_sec) {
                    this._prev_update_sec = now_sec ;
                    this._init() ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this._app_config = app_config ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._access = null ;

        this._can_manage = function () { return this._app_config.current_user.is_administrator ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="admin-access" >' +

  '<div style="float:left;" class="notes" >' +
    '<p>This section allows to assign user accounts to various roles defined' +
    '   in a context of the application. See a detailed description of each' +
    '   role in the corresponding subsection below.</p>' +
  '</div>' +
  '<div style="float:right;" >' +
    '<button name="update" class="control-button" title="update from the database" >UPDATE</button>' +
  '</div>' +
  '<div style="clear:both;" ></div>' +

  '<div class="info" id="info"    style="float:left;"  >&nbsp;</div>' +
  '<div class="info" id="updated" style="float:right;" >&nbsp;</div>' +
  '<div style="clear:both;"></div>' +

  '<div id="tabs" >' +

    '<ul>' +
      '<li><a href="#administrators" >Administrators</a></li>' +
      '<li><a href="#editors"        >Editors</a></li>' +
      '<li><a href="#others"         >Other Users</a></li>' +
    '</ul>' +

    '<div id="administrators" >' +
      '<div class="tab-body" >' +
        '<div class="notes" >' +
          '<p>Administrators posses highest level privileges in the application' +
          '   as they\'re allowed to perform any operation on the inventory and' +
          '   other users. The only restriction is that an administrator is not' +
          '   allowed to remove their own account from the list of administrators.</p>' +
        '</div>' +
        '<div style="float:left;" >        ' +
        '  <input type="text"              ' +
        '         size="8"                 ' +
        '         name="ADMINISTRATOR" ' +
        '         title="fill in a UNIX account of a user, press RETURN to save" />' +
        '</div>' +
        '<div style="float:left"  class="hint" > &larr; add new user here</div>' +
        '<div style="clear:both;" ></div>' +
        '<div id="admin-access-ADMINISTRATOR" ></div>' +
      '</div>' +
    '</div>' +

    '<div id="editors" >' +
      '<div class="tab-body" >' +
        '<div class="notes" >' +
          '<p>Editors can add new equipment to the inventory, delete or edit' +
          '   existing records of the equipment and also manage certain aspects' +
          '   of the equipment life-cycle.</p>' +
        '</div>' +
        '<div style="float:left;" > ' +
          '<input type="text"       ' +
          '       size="8"          ' +
          '       name="EDITOR" ' +
          '       title="fill in a UNIX account of a user, press RETURN to save" />' +
        '</div>' +
        '<div style="float:left;" class="hint" > &larr; add new user here</div>' +
        '<div style="clear:both;" ></div>' +
        '<div id="admin-access-EDITOR" ></div>' +
      '</div>' +
    '</div>' +

    '<div id="others" >' +
      '<div class="tab-body" >' +
        '<div class="notes" >' +
        '  <p>Other users may be allowed some limited access to manage certain aspects' +
        '     of the equipment life-cycle.</p>' +
        '</div>' +
        '<div style="float:left;" >' +
        '  <input type="text"      ' +
        '         size="8"         ' +
        '         name="OTHER" ' +
        '         title="fill in a UNIX account of a user, press RETURN to save" />' +
        '</div>' +
        '<div style="float:left;" class="hint" > &larr; add new user here</div>' +
        '<div style="clear:both;" ></div>' +
        '<div id="admin-access-OTHER" ></div>' +
      '</div>' +
    '</div>' +
  '</div>' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#admin-access') ;
            }
            return this._wa_elem ;
        } ;
        this._table = function (role) {
            var id = 'admin-access-' + role ;
            if (!this._table_obj) this._table_obj = {} ;
            if (!this._table_obj[role]) {
                var rows = [] ;
                var elem = this._wa().find('#'+id) ;
                var hdr = [] ;
                if (this._can_manage()) hdr.push (
                    {   name: 'DELETE', sorted: false , hideable: true ,
                        type: { after_sort: function () {
                            elem.find('button[name="delete"]').button() ;
                        }}
                    }
                ) ;
                hdr.push (
                    {   name: 'UID'} ,
                    {   name: 'user'}
                ) ;
                if (roles[role].can_edit_inventory && !roles[role].is_administrator) hdr.push (
                    {   name: 'dictionary privilege', sorted: false ,
                        type: { after_sort: function () {
                            elem.find('button[name="dict_priv"]').button() ;
                        }}
                    }
                ) ;
                hdr.push (
                    {   name: 'added'} ,
                    {   name: 'last active'}
                ) ;
                this._table_obj[role] = new SimpleTable.constructor (id, hdr, rows) ;
                this._table_obj[role].display() ;
            }
            return this._table_obj[role] ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) this._info_elem = this._wa().children('#info') ;
            this._info_elem.html(html) ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;
            
            this._wa().children('#tabs').tabs() ;

            this._wa().find('button.control-button').button().click(function () {
                switch (this.name) {
                    case 'update' : _that._load() ; break ;
                }
            }) ;

            var inputs = this._wa().find('input') ;

            if (this._can_manage())
                inputs.keyup(function (e) {
                    var role = this.name ;
                    var uid  = $(this).val() ;
                    if (uid && e.keyCode == 13) {
                        _that._action (
                            'Creating User...' ,
                            '../irep/ws/access_new.php' ,
                            {uid: uid, role: role}
                        ) ;
                    }
                }) ;
            else
                inputs.attr('disabled', 'disabled') ;

            this._load() ;
        } ;
        this._load = function () {
            this._action (
                'Loading...' ,
                '../irep/ws/access_get.php' ,
                {}
            ) ;
        } ;
        this._delete_user = function (uid) {
            Fwk.ask_yes_no (
                'Removing a user' ,
                'Are you sure you want to remove user <b>'+uid+'</b> from the list?' ,
                function () {
                    _that._action (
                        'Deleting...' ,
                        '../irep/ws/access_delete.php' ,
                        {uid: uid}
                    ) ;
                }
            ) ;
        } ;
        this._toggle_priv = function (uid, name) {
            this._action (
                'Changing Dictionary Privilege...' ,
                '../irep/ws/access_toggle_priv.php' ,
                {uid: uid, name: name}
            ) ;
        } ;
        this._action = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_GET (url, params, function (data) {
                _that._access = data.access ;
                _that._display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        this._display = function () {
            this._display_users('ADMINISTRATOR') ;
            this._display_users('EDITOR') ;
            this._display_users('OTHER') ;
        } ;
        this._display_users = function (role) {
            var rows = [] ;
            var users = this._access[role] ;
            for (var i in users) {
                var a = users[i] ;
                var row = [] ;
                if (this._can_manage()) row.push (
                    SimpleTable.html.Button ('X', {
                        name:    'delete' ,
                        classes: 'control-button control-button-small control-button-important' ,
                        onclick: "Fwk.get_application('Admin', 'Access Control')._delete_user('"+a.uid+"')" ,
                        title:   'delete this user from the list'})) ;
                row.push (
                    a.uid ,
                    a.name) ;
                if (roles[role].can_edit_inventory && !roles[role].is_administrator) row.push (
                    this._can_manage()
                        ? SimpleTable.html.Checkbox ({
                            name:    'dict_priv' ,
                            onclick: "Fwk.get_application('Admin', 'Access Control')._toggle_priv('"+a.uid+"','dict_priv')" ,
                            title:   'togle the dictionary privilege' ,
                            checked: a.privilege.dict_priv ? 'checked' : ''})
                        : (a.privilege.dict_priv ? 'Yes' : 'No')) ;
                row.push (
                    a.added_time ,
                    a.last_active_time) ;

                rows.push(row) ;
            }
            this._table(role).load(rows) ;
        } ;
    }
    Class.define_class (Admin_Access, FwkApplication, {}, {}) ;
    
    return Admin_Access ;
}) ;

