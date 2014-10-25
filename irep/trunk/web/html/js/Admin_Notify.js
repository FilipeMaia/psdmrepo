define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../irep/css/Admin_Notify.css') ;

    /**
     * The application for managing e-mail notifications on various events
     *
     * @returns {Admin_Notify}
     */
    function Admin_Notify (app_config) {

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

        this.app_config = app_config ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._access      = null ;
        this._event_types = null ;
        this._schedule    = null ;
        this._notify      = null ;
        this._pending     = null ;

        this._can_manage = function () { return this.app_config.current_user.is_administrator ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||

'<div id="admin-notifications" >' +

  '<div style="float:left;" class="notes" >' +
    '<p>In order to avoid an excessive e-mail traffic the notification system' +
    '   will send just one message for any modification made in a specific context.' +
    '   For the very same reason the default behavior of the system is to send' +
    '   a summary daily message with all changes made before a time specified below,' +
    '   unless this site administrators choose a different policy (such as instantaneous' +
    '   notification).</p>' +
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
      '<li><a href="#myself"         > On my equipment        </a></li>' +
      '<li><a href="#administrators" > Sent to administrators </a></li>' +
      '<li><a href="#others"         > Sent to other users    </a></li>' +
      '<li><a href="#pending"        > Pending                </a></li>' +
    '</ul>' +

    '<div id="myself" >' +
      '<div class="tab-body" >' +
        '<div class="notes" >' +
          '<p>This section is aiming at editors who might be interested to track changes' +
          '   made to their equipment by other people involved into various stages' +
          '   of the workflow. Note that editors will not get notifications' +
          '   on changes made by themselves.</p>' +
          '<p>Notification settings found in this section can only be managed by' +
          '   editors themselves or by administrators of the application.</p>' +
        '</div>' +
        '<div style="margin-bottom:20px;" >' +
          '<select name="policy4EDITOR" >' +
            '<option value="DELAYED" > daily notification (08:00am) </option>' +
            '<option value="INSTANT" > instant notification         </option>' +
          '</select>' +
        '</div>' +
        '<div id="admin-notifications-EDITOR" ></div>' +
      '</div>' +
    '</div>' +

    '<div id="administrators" >' +
      '<div class="tab-body" >' +
        '<div class="notes" >' +
          '<p>This section is aiming at administrators of this software who might' +
          '   be interested to track major changes made to the equipment, user accounts' +
          '   or software configuration. Note that administrators will not get notifications' +
          '   on changes made by themselves.</p>' +
          '<p>Notification settings found in this section can only be managed by any' +
          '   administrator of the software.</p>' +
        '</div>' +
        '<div style="margin-bottom:20px;" >' +
          '<select name="policy4ADMINISTRATOR" >' +
            '<option value="DELAYED" > daily notification (08:00am) </option>' +
            '<option value="INSTANT" > instant notification         </option>' +
          '</select>' +
        '</div>' +
        '<div id="admin-notifications-ADMINISTRATOR" ></div>' +
      '</div>' +
    '</div>' +

    '<div id="others" >' +
      '<div class="tab-body" >' +
        '<div class="notes" >' +
          '<p>This section is aiming at users (not necessarily editors) who are involved' +
          '   into various stages of the equipment workflow.</p>' +
          '<p>Only administrators of this application are allowed to modify notification' +
          '   settings found on this page.</p>' +
        '</div>' +
        '<div style="margin-bottom:20px;" >' +
          '<select name="policy4OTHER" >' +
            '<option value="DELAYED" > daily notification (08:00am) </option>' +
            '<option value="INSTANT" > instant notification         </option>' +
          '</select>' +
        '</div>' +
        '<div id="admin-notifications-OTHER" ></div>' +
      '</div>' +
    '</div>' +

    '<div id="pending" >' +
      '<div class="tab-body" >' +
        '<div class="notes" >' +
          '<p>Pending/scheduled notifications (if any found below) can be submitted' +
          '   for instant delivery by pressing a group \'Submit\' button or individually' +
          '   if needed. Notifications can also be deleted if needed. An additional dialog' +
          '   will be initiated to confirm group operations.</p>' +
          '<p>Only administrators of this application are authorized for these operations.</p>' +
        '</div>' +
        '<div style="margin-bottom:20px;" >' +
          '<button name="submit_all" class="control-button" title="Submit all pending notifications to be instantly delivered to their recipient" >SUBMIT</button>' +
          '<button name="delete_all" class="control-button" title="Delete all pending notifications"                                              >DELETE</button>' +
        '</div>' +
        '<div id="admin-notifications-pending" ></div>' +
      '</div>' +
    '</div>' +
  '</div>' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#admin-notifications') ;
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
                    case 'update'     : _that._load() ; break ;
                    case 'submit_all' : _that._pending_submit() ; break ;
                    case 'delete_all' : _that._pending_delete() ; break ;
                }
            }) ;

            if (!this._can_manage())
                this._wa().find('select').attr('disabled', 'disabled') ;

            this._load() ;
        } ;

        this._load = function () {
            this._action(
                'Loading...' ,
                '../irep/ws/notify_get.php' ,
                {}
            ) ;
        } ;
        this._save = function (recipient, uid, event_name, enabled) {
            this._action (
                'Saving...' ,
                '../irep/ws/notify_save.php' ,
                {   recipient:  recipient ,
                    uid:        uid ,
                    event_name: event_name ,
                    enabled:    enabled ? 1 : 0
                }
            ) ;
        } ;
        this._schedule_save = function (recipient, policy) {
            this._action(
                'Saving...' ,
                '../irep/ws/notify_save.php' ,
                {   recipient:  recipient ,
                    policy:     policy
                }
            ) ;
        } ;
        this._pending_submit = function (idx) {
            var params = {action: 'submit'} ;
            if (idx !== undefined) params.id = this._pending[idx].id ;
            this._action (
                'Submitting...' ,
                '../irep/ws/notify_queue.php' ,
                params
            ) ;
        } ;
        this._pending_delete = function (idx) {
            var params = {action: 'delete'} ;
            if (idx !== undefined) params.id = this._pending[idx].id ;
            this._action (
                'Deleting...' ,
                '../irep/ws/notify_queue.php' ,
                params
            ) ;
        } ;
        this._action = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_GET(url, params, function (data) {
                _that._access      = data.access ;
                _that._event_types = data.event_types ;
                _that._schedule    = data.schedule ;
                _that._notify      = data.notify ;
                _that._pending     = data.pending ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                _that._display() ;
            }) ;
        } ;
        this._display = function () {
            this._display_editor() ;
            this._display_administrator() ;
            this._display_other() ;
            this._display_pending() ;
        } ;
        this._display_editor = function () {
            var role = 'EDITOR' ;
            if( !this.app_config.current_user.can_edit_inventory) {
                this._wa().find('#admin-notifications-'+role).html(
                    '<div style="color:maroon ;">'+
                    '  We are sorry! Your account doesn\'t have sufficient privileges to edit the inventory.'+
                    '</div>') ;
                return ;
            }
            this._display_impl(role) ;
        } ;
        this._display_administrator = function () {  this._display_impl('ADMINISTRATOR') ;  } ;
        this._display_other         = function () {  this._display_impl('OTHER') ;          } ;

        this._display_impl = function (role) {

            var hdr  = [ {name: 'event', sorted: false } ] ;
            var rows = [] ;

            switch (role) {

                case 'EDITOR':

                    hdr.push({ name: 'notify', sorted: false }) ;
                    for (var i in this._event_types[role]) {
                        var event  = this._event_types[role][i] ;
                        var notify = this._notify[role][this.app_config.current_user.uid] || {} ;
                        var attr = {
                            classes: event.name,
                            name: this.app_config.current_user.uid,
                            checked: notify[event.name] == true } ;
                        rows.push([
                            event.description,
                            SimpleTable.html.Checkbox(attr) ]) ;
                    }
                    break ;

                case 'ADMINISTRATOR':
                case 'OTHER':

                    for (var i in this._access[role]) {
                        var user = this._access[role][i] ;
                        hdr.push({name: user.uid, sorted: false}) ;
                    }
                    for (var i in this._event_types[role]) {
                        var event = this._event_types[role][i] ;
                        var row   = [ event.description ] ;
                        for (var j in this._access[role]) {
                            var user = this._access[role][j] ;
                            var notify = this._notify[role][user.uid] || {} ;
                            var attr = {
                                classes: event.name,
                                name:    user.uid,
                                checked: notify[event.name] == true } ;
                            if( !( this._can_manage() || ( this.app_config.current_user.uid == user.uid)))
                                attr.disabled = 'disabled' ;
                            row.push(SimpleTable.html.Checkbox(attr)) ;
                        }
                        rows.push(row) ;
                    }
                    break ;
            }
            var table = new SimpleTable.constructor('admin-notifications-'+role, hdr, rows) ;
            table.display() ;

            var policy = this._wa().find('select[name="policy4'+role+'"]') ;
            policy.val(this._schedule[role]) ;
            policy.change(function () {
                _that._schedule_save(role, $(this).val()) ;
            }) ;
            if (this._can_manage()) policy.removeAttr('disabled') ;
            this._wa().find('#admin-notifications-'+role).find('input[type="checkbox"]').click(function () {
                _that._save(role, this.name, $(this).attr('class'), $(this).is(':checked')) ;
            }) ;
        } ;
        this._display_pending = function () {
            var hdr = [
                { name:   'time' },
                { name:   'event' },
                { name:   'originator' },
                { name:   'recipient' },
                { name:   'recipient_role' },
                { name:   'ACTIONS',
                  sorted: false,
                  type:   { after_sort: function () { _that._wa().find('.admin-notifications-pending-tools').button() ; }}}
            ] ;
            var rows = [] ;
            for (var i in this._pending) {
                var entry = this._pending[i] ;
                var find_button = '' ;
                if(( entry.scope == 'EQUIPMENT') && ( entry.event_type_name != 'on_equipment_delete')) {
                    find_button = SimpleTable.html.Button('find equipment', {
                            name:    'find_'+i,
                            classes: 'admin-notifications-pending-tools',
                            onclick: "global_search_requipment_by_id('"+entry.equipment_id+"')",
                            title:   'display this equipment if it is still available'
                        }) ;
                }
                rows.push([
                    entry.event_time,
                    entry.event_type_description,
                    entry.originator_uid,
                    entry.recipient_uid,
                    entry.recipient_role,
                    this.can_manage_access() ?
                        SimpleTable.html.Button('submit', {
                            name:    'submit_'+i,
                            classes: 'admin-notifications-pending-tools',
                            onclick: "Fwk.get_application('Admin', 'E-mail Notifications')._pending_submit('"+i+"')",
                            title:   'submit this event for the instant delivery'
                        })+' '+
                        SimpleTable.html.Button('delete', {
                            name:    'delete_'+i,
                            classes: 'admin-notifications-pending-tools',
                            onclick: "Fwk.get_application('Admin', 'E-mail Notifications')._pending_delete('"+i+"')",
                            title:   'delete this entry from the queue'
                        })+' '+
                        find_button : ''
                    ]) ;
            }
            var table = new SimpleTable.constructor('admin-notifications-pending', hdr, rows) ;
            table.display() ;

            this._wa().find('.admin-notifications-pending-tools').button() ;
        } ;

    }
    Class.define_class (Admin_Notify, FwkApplication, {}, {}) ;
    
    return Admin_Notify ;
}) ;

