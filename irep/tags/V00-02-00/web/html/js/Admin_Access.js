define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../irep/css/Admin_Access.css') ;

    /**
     * The application for managing access privileges
     *
     * @returns {Admin_Access}
     */
    function Admin_Access (app_config) {

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

        this.on_update = function () {
            if (this.active) {
                this._init() ;
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
        '         name="administrator2add" ' +
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
          '       name="editor2add" ' +
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
        '         name="other2add" ' +
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
                    var name = this.name ;
                    var uid = $(this).val() ;
                    if (uid && e.keyCode == 13)
                        switch (this.name) {
                            case 'administrator2add' : _that._create_administrator(uid) ; break ;
                            case 'editor2add'        : _that._create_editor       (uid) ; break ;
                            case 'other2add'         : _that._create_other        (uid) ; break ;
                        }}) ;
            else
                inputs.attr('disabled', 'disabled') ;

            this._load() ;
        } ;

        this._create_administrator = function (uid) { this._create_user(uid, 'ADMINISTRATOR') ; } ;
        this._create_editor        = function (uid) { this._create_user(uid, 'EDITOR') ; } ;
        this._create_other         = function (uid) { this._create_user(uid, 'OTHER') ; } ;

        this._load = function () {
            this._action (
                'Loading...' ,
                '../irep/ws/access_get.php' ,
                {}) ;
        } ;

        this._create_user = function (uid, role) {
            this._action (
                'Creating User...' ,
                '../irep/ws/access_new.php' ,
                {uid: uid, role: role}) ;
        } ;

        this._delete_user = function (uid) {
            this._action (
                'Deleting...' ,
                '../irep/ws/access_delete.php' ,
                {uid: uid}) ;
        } ;

        this._toggle_priv = function (uid, name) {
            this._action (
                'Changing Dictionary Privilege...' ,
                '../irep/ws/access_toggle_priv.php' ,
                {uid: uid, name: name}) ;
        } ;

        this._action = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_GET (url, params, function (data) {
                _that._access = data.access ;
                _that._display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                //_that.notify_load() ;
            }) ;
        } ;

        this._display = function () {
            this._display_users('admin-access-ADMINISTRATOR', this._access.ADMINISTRATOR, true, true) ;
            this._display_users('admin-access-EDITOR',        this._access.EDITOR,        false, true) ;
            this._display_users('admin-access-OTHER',         this._access.OTHER,         false, false) ;
        } ;

        this._display_users = function (id, users, is_administrator, can_edit_inventory) {
            var rows = [] ;
            for (var i in users) {
                var a = users[i] ;
                var row = [] ;
                if (this._can_manage()) row.push (
                    Button_HTML('X', {
                        name:    'delete' ,
                        onclick: "Fwk.get_application('Admin', 'Access Control')._delete_user('"+a.uid+"')" ,
                        title:   'delete this user from the list'})) ;
                row.push (
                    a.uid ,
                    a.name) ;
                if (can_edit_inventory && !is_administrator) row.push (
                    this._can_manage()
                        ? Checkbox_HTML ({
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
            var hdr = [] ;
            if (this._can_manage()) hdr.push(                      {name: 'DELETE',               sorted: false, hideable: true}) ;
            hdr.push (                                             {name: 'UID',                  sorted: false} ,
                                                                   {name: 'user',                 sorted: false}) ;
            if (can_edit_inventory && !is_administrator) hdr.push ({name: 'dictionary privilege', sorted: false}) ;
            hdr.push (                                             {name: 'added',                sorted: false} ,
                                                                   {name: 'last active',          sorted: false}) ;
            var elem = this._wa().find('#'+id) ;

            var table = new Table(id, hdr, rows) ;
            table.display() ;

            elem.find('button[name="delete"]').button() ;
        } ;
    }
    Class.define_class (Admin_Access, FwkApplication, {}, {}) ;
    
    return Admin_Access ;
}) ;

