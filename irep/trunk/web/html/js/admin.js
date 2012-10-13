function p_appl_admin () {

    var that = this ;

    this.when_done = null ;

    /* -------------------------------------------------------------------------
     *   Data structures and methods to be used/called by external users
     *
     *   select(context, when_done)
     *      select a specific context
     *
     *   select_default()
     *      select default context as implemented in the object
     *
     *   if_ready2giveup(handler2call)
     *      check if the object's state allows to be released, and if so then
     *      call the specified function. Otherwise just ignore it. Normally
     *      this operation is used as a safeguard preventing releasing
     *      an interface focus if there is on-going unfinished editing
     *      within one of the interfaces associated with the object.
     *
     *   update()
     *      is called to make sure that  all data structures and interfaces
     *      of the object are properly updated.
     *
     * -------------------------------------------------------------------------
     */
    this.name      = 'admin' ;
    this.full_name = 'Admin' ;
    this.context   = '' ;
    this.default_context = 'access' ;

    this.select = function (context, when_done) {
        that.context   = context ;
        this.when_done = when_done ;
        this.init() ;
    } ;
    this.select_default = function () {
        if (this.context == '') this.context = this.default_context ;
        this.init() ;
    } ;
    this.if_ready2giveup = function (handler2call) {
        this.init() ;
        handler2call() ;
    } ;
    this.update = function () {
        this.init() ;
        this.access_load() ;
        this.notify_load() ;
        this.slacidnumbers_load() ;
    } ;
    
    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */
    this.initialized = false ;
    this.init = function () {
        if(this.initialized) return ;
        this.initialized = true ;

        $('#admin-access-reload'       ).button().click(function () { that.access_load       () ; }) ;
        $('#admin-notifications-reload').button().click(function () { that.notify_load       () ; }) ;
        $('#admin-slacidnumbers-reload').button().click(function () { that.slacidnumbers_load() ; }) ;

        var administrator2add = $('#admin-access').find('input[name="administrator2add"]') ;
        administrator2add.
            keyup(function (e) {
                var uid = $(this).val() ;
                if (uid == '') { return ; }
                if (e.keyCode == 13) { that.access_create_administrator(uid) ; return ; }}) ;

        var editor2add = $('#admin-access').find('input[name="editor2add"]') ;
        editor2add.
            keyup(function (e) {
                var uid = $(this).val() ;
                if (uid == '') { return ; }
                if (e.keyCode == 13) { that.access_create_editor(uid) ; return ; }}) ;

        var other2add = $('#admin-access').find('input[name="other2add"]') ;
        other2add.
            keyup(function (e) {
                var uid = $(this).val() ;
                if(uid == '') { return ; }
                if(e.keyCode == 13) { that.access_create_other(uid) ; return ; }}) ;

        var submit_pending = $('#admin-notifications').find('button[name="submit_all"]').
            button().
            click(function () { that.notify_pending_submit() ; }) ;

        var delete_pending = $('#admin-notifications').find('button[name="delete_all"]').
            button().
            click(function () { that.notify_pending_delete() ; }) ;

        if (!this.can_manage_access()) {
            administrator2add.attr('disabled', 'disabled') ;
                   editor2add.attr('disabled', 'disabled') ;
                    other2add.attr('disabled', 'disabled') ;
               submit_pending.attr('disabled', 'disabled') ;
               delete_pending.attr('disabled', 'disabled') ;
        }
        var slacidnumbers_ranges_edit   = $('#admin-slacidnumbers').find('#ranges').find('button[name="edit"]'  ).button().button('disable') ;
        var slacidnumbers_ranges_save   = $('#admin-slacidnumbers').find('#ranges').find('button[name="save"]'  ).button().button('disable') ;
        var slacidnumbers_ranges_cancel = $('#admin-slacidnumbers').find('#ranges').find('button[name="cancel"]').button().button('disable') ;

        if (this.can_manage_access()) {
            slacidnumbers_ranges_edit.click  (function () { that.slacidnumbers_ranges_edit       () ; }) ;
            slacidnumbers_ranges_save.click  (function () { that.slacidnumbers_ranges_edit_save  () ; }) ;
            slacidnumbers_ranges_cancel.click(function () { that.slacidnumbers_ranges_edit_cancel() ; }) ;
        }
        $('#admin-access'       ).find('#tabs').tabs() ;
        $('#admin-notifications').find('#tabs').tabs() ;
        $('#admin-slacidnumbers').find('#tabs').tabs() ;

        this.reload_timer_event() ;
    } ;
    this.can_manage_access = function () { return global_current_user.is_administrator ; } ;
    this.can_manage_notify = this.can_manage_access ;

    this.reload_timer = null ;
    this.reload_timer_restart = function () {
        this.reload_timer = window.setTimeout('admin.reload_timer_event()', 60000) ;
    } ;
    this.reload_timer_event = function () {
        if (!this.slacidnumber_editing()) {
            this.access_load() ;
            this.notify_load() ;
            this.slacidnumbers_load() ;
        }
        this.reload_timer_restart() ;
    } ;

    /* ------------------
     *   Access control
     * ------------------
     */
    this.access = null ;
    this.access_create_administrator = function (uid) { this.access_create_user(uid,'ADMINISTRATOR') ; } ;
    this.access_create_editor        = function (uid) { this.access_create_user(uid,'EDITOR') ; } ;
    this.access_create_other         = function (uid) { this.access_create_user(uid,'OTHER') ; } ;
 
    this.editors = function (managers_only) {

        // Return an array of user accounts who are allowed to edit inventory entries.
        // This will include dedicated editors as as well as administrators.
        //
        // Fall back to a static array of 'global_users' if no users have been
        // dynamically loaded so far.
        //
        if( this.access) {
            var result = [] ;
            var users = undefined ;
            if(managers_only)
                users = $.merge(
                    [],
                    this.access.EDITOR
                ) ;
            else
                users = $.merge(
                    $.merge(
                        [],
                        this.access.ADMINISTRATOR
                    ),
                    this.access.EDITOR) ;
            for (var i in users) {
                var user = users[i] ;
                result.push(user.uid) ;
            }
            return result ;
        }
        return global_users ;
    } ;
    this.access_display_users = function (id,users, is_administrator, can_edit_inventory) {
        var rows = [] ;
        for (var i in users) {
            var a = users[i] ;
            var row = [] ;
            if(this.can_manage_access()) row.push(
                Button_HTML('X', {
                    name:    'delete',
                    onclick: "admin.access_delete_user('"+a.uid+"')",
                    title:   'delete this user from the list' })) ;
            row.push(
                a.uid,
                a.name) ;
            if(can_edit_inventory && !is_administrator) row.push(
                this.can_manage_access()
                    ? Checkbox_HTML({
                        name:    'dict_priv',
                        onclick: "admin.access_toggle_priv('"+a.uid+"','dict_priv')",
                        title:   'togle the dictionary privilege',
                        checked: a.privilege.dict_priv ? 'checked' : '' })
                    : ( a.privilege.dict_priv ? 'Yes' : 'No')) ;
            row.push(
                a.added_time,
                a.last_active_time) ;

            rows.push(row) ;
        }
        var hdr = [] ;
        if(this.can_manage_access()) hdr.push(               { name: 'DELETE',               sorted: false }) ;
        hdr.push(                                            { name: 'UID',                  sorted: false } ,
                                                             { name: 'user',                 sorted: false }) ;
        if(can_edit_inventory && !is_administrator) hdr.push({ name: 'dictionary privilege', sorted: false }) ;
        hdr.push(                                            { name: 'added',                sorted: false } ,
                                                             { name: 'last active',          sorted: false }) ;

        var table = new Table(id, hdr, rows) ;
        table.display() ;

        var elem = $('#'+id) ;

        elem.find('button[name="delete"]').button() ;
    } ;

    this.access_display = function () {
        this.access_display_users('admin-access-ADMINISTRATOR', this.access.ADMINISTRATOR, true, true) ;
        this.access_display_users('admin-access-EDITOR',        this.access.EDITOR,        false, true) ;
        this.access_display_users('admin-access-OTHER',         this.access.OTHER,         false, false) ;
    } ;

    this.access_load = function () {
        var params = {} ;
        var jqXHR = $.get('../irep/ws/access_get.php',params,function (data) {
            if(data.status != 'success') { report_error(data.message, null) ; return ; }
            that.access = data.access ;
            that.access_display() ;
        },
        'JSON').error(function () {
            report_error('failed to load access control info because of: '+jqXHR.statusText, null) ;
            return ;
        }) ;
    } ;

    this.access_create_user = function (uid,role) {
        var params = {uid:uid,role:role} ;
        var jqXHR = $.get('../irep/ws/access_new.php',params,function (data) {
            if(data.status != 'success') { report_error(data.message, null) ; return ; }
            that.access = data.access ;
            that.access_display() ;
            that.notify_load() ;
        },
        'JSON').error(function () {
            report_error('failed to add a new user to the access control list because of: '+jqXHR.statusText, null) ;
            return ;
        }) ;
    } ;

    this.access_delete_user = function (uid) {
        var params = {uid:uid} ;
        var jqXHR = $.get('../irep/ws/access_delete.php',params,function (data) {
            if(data.status != 'success') { report_error(data.message, null) ; return ; }
            that.access = data.access ;
            that.access_display() ;
            that.notify_load() ;
        },
        'JSON').error(function () {
            report_error('failed to delete this user from the access control list because of: '+jqXHR.statusText, null) ;
            return ;
        }) ;
    } ;
    this.access_toggle_priv = function (uid,name) {
        var params = {uid:uid, name:name} ;
        var jqXHR = $.get('../irep/ws/access_toggle_priv.php',params,function (data) {
            if(data.status != 'success') { report_error(data.message, null) ; return ; }
            that.access = data.access ;
            that.access_display() ;
            that.notify_load() ;
        },
        'JSON').error(function () {
            report_error('failed to togle the privilege state because of: '+jqXHR.statusText, null) ;
            return ;
        }) ;
    } ;

    /* -----------------
     *   Notifications
     * -----------------
     */
    this.notify             = null ;
    this.notify_event_types = null ;

    this.notify_display_editor = function () {
        var role = 'EDITOR' ;
        if( !global_current_user.can_edit_inventory) {
            $('#admin-notifications-'+role).html(
                '<div style="color:maroon ;">'+
                '  We are sorry! Your account doesn\'t have sufficient privileges to edit the inventory.'+
                '</div>') ;
            return ;
        }
        this.notify_display_impl(role) ;
    } ;
    this.notify_display_administrator = function () {  this.notify_display_impl('ADMINISTRATOR') ;  } ;
    this.notify_display_other         = function () {  this.notify_display_impl('OTHER') ;          } ;

    this.notify_display_impl = function (role) {

        var hdr  = [ {name: 'event', sorted: false } ] ;
        var rows = [] ;

        switch (role) {

            case 'EDITOR':

                hdr.push({ name: 'notify', sorted: false }) ;
                for (var i in this.notify_event_types[role]) {
                    var event  = this.notify_event_types[role][i] ;
                    var notify = this.notify[role][global_current_user.uid] || {} ;
                    var attr = {
                        classes: event.name,
                        name: global_current_user.uid,
                        checked: notify[event.name] == true } ;
                    rows.push([
                        event.description,
                        Checkbox_HTML(attr) ]) ;
                }
                break ;

            case 'ADMINISTRATOR':
            case 'OTHER':

                for (var i in this.notify_access[role]) {
                    var user = this.notify_access[role][i] ;
                    hdr.push({name: user.uid, sorted: false}) ;
                }
                for (var i in this.notify_event_types[role]) {
                    var event = this.notify_event_types[role][i] ;
                    var row   = [ event.description ] ;
                    for (var j in this.notify_access[role]) {
                        var user = this.notify_access[role][j] ;
                        var notify = this.notify[role][user.uid] || {} ;
                        var attr = {
                            classes: event.name,
                            name:    user.uid,
                            checked: notify[event.name] == true } ;
                        if( !( this.can_manage_notify() || ( global_current_user.uid == user.uid)))
                            attr.disabled = 'disabled' ;
                        row.push( Checkbox_HTML(attr)) ;
                    }
                    rows.push(row) ;
                }
                break ;
        }
        var table = new Table('admin-notifications-'+role, hdr, rows) ;
        table.display() ;

        var policy = $('#admin-notifications').find('select[name="policy4'+role+'"]') ;
        policy.val(this.notify_schedule[role]) ;
        policy.change(function () {
            that.notify_schedule_save(role, $(this).val()) ;
        }) ;
        if(this.can_manage_notify()) policy.removeAttr('disabled') ;
        $('#admin-notifications-'+role).find('input[type="checkbox"]').click(function () {
            that.notify_save(role, this.name, $(this).attr('class'), $(this).is(':checked')) ;
        }) ;
    } ;
    this.notify_display_pending = function () {
        var hdr = [
            { name:   'time' },
            { name:   'event' },
            { name:   'originator' },
            { name:   'recipient' },
            { name:   'recipient_role' },
            { name:   'ACTIONS',
              sorted: false,
              type:   { after_sort: function () { $('.admin-notifications-pending-tools').button() ; }}}
        ] ;
        var rows = [] ;
        for (var i in this.notify_pending) {
            var entry = this.notify_pending[i] ;
            var find_button = '' ;
            if(( entry.scope == 'EQUIPMENT') && ( entry.event_type_name != 'on_equipment_delete')) {
                find_button = Button_HTML('find equipment', {
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
                    Button_HTML('submit', {
                        name:    'submit_'+i,
                        classes: 'admin-notifications-pending-tools',
                        onclick: "admin.notify_pending_submit('"+i+"')",
                        title:   'submit this event for teh instant delivery'
                    })+' '+
                    Button_HTML('delete', {
                        name:    'delete_'+i,
                        classes: 'admin-notifications-pending-tools',
                        onclick: "admin.notify_pending_delete('"+i+"')",
                        title:   'delete this entry from the queue'
                    })+' '+
                    find_button : ''
                ]) ;
        }
        var table = new Table('admin-notifications-pending', hdr, rows) ;
        table.display() ;
        
        $('.admin-notifications-pending-tools').button() ;
    } ;
    this.notify_display = function () {
        this.notify_display_editor() ;
        this.notify_display_administrator() ;
        this.notify_display_other() ;
        this.notify_display_pending() ;
    } ;

    this.notify_action = function (url,params) {
        var jqXHR = $.get(url,params,function (data) {
            if(data.status != 'success') { report_error(data.message, null) ; return ; }

            that.notify_access      = data.access ;
            that.notify_event_types = data.event_types ;
            that.notify_schedule    = data.schedule ;
            that.notify             = data.notify ;
            that.notify_pending     = data.pending ;

            that.notify_display() ;
        },
        'JSON').error(function () {
            report_error('failed to load noditications list because of: '+jqXHR.statusText, null) ;
            return ;
        }) ;
    } ;
    this.notify_load = function () {
        this.notify_action(
            '../irep/ws/notify_get.php',
            {}
        ) ;
    } ;
    this.notify_save = function (recipient, uid, event_name, enabled) {
        this.notify_action(
            '../irep/ws/notify_save.php',
            {   recipient:  recipient,
                uid:        uid,
                event_name: event_name,
                enabled:    enabled ? 1 : 0 }
        ) ;
    } ;
    this.notify_schedule_save = function (recipient, policy) {
        this.notify_action(
            '../irep/ws/notify_save.php',
            {   recipient:  recipient,
                policy:     policy }
        ) ;
    } ;
    this.notify_pending_submit = function (idx) {
        var params = { action: 'submit' } ;
        if( idx != undefined) params.id = this.notify_pending[idx].id ;
        this.notify_action(
            '../irep/ws/notify_queue.php',
            params
        ) ;
    } ;
    this.notify_pending_delete = function (idx) {
        var params = { action: 'delete' } ;
        if( idx != undefined) params.id = this.notify_pending[idx].id ;
        this.notify_action(
            '../irep/ws/notify_queue.php',
            params
        ) ;
    } ;

    /* ------------------
     *   SLACid numbers
     * ------------------
     */
    this.slacidnumber_range = null;
    this.slacidnumber_ranges_editing = false;
    this.slacidnumber_editing = function() {
        return this.slacidnumber_ranges_editing;
    };

    this.slacidnumbers_ranges_edit_tools = function(edit_mode) {
        var ranges_elem = $('#admin-slacidnumbers').find('#ranges');
        var slacidnumbers_ranges_edit   = ranges_elem.find('button[name="edit"]'  );
        var slacidnumbers_ranges_save   = ranges_elem.find('button[name="save"]'  );
        var slacidnumbers_ranges_cancel = ranges_elem.find('button[name="cancel"]');
        if( !this.can_manage_access()) return;
        slacidnumbers_ranges_edit.button  (edit_mode ? 'disable' : 'enable');
        slacidnumbers_ranges_save.button  (edit_mode ? 'enable'  : 'disable');
        slacidnumbers_ranges_cancel.button(edit_mode ? 'enable'  : 'disable');
    };
    function count_elements_in_array(obj) {
        var size = 0;
        for( var key in obj ) size++;
        return size;
    }
    this.slacidnumber_ranges_display = function(edit_mode) {
        this.slacidnumber_ranges_editing = edit_mode;
        this.slacidnumbers_ranges_edit_tools(edit_mode);
        var ranges_elem = $('#admin-slacidnumbers').find('#ranges');
        var rows = [];
        for( var r in this.slacidnumber_range ) {
            var range = this.slacidnumber_range[r];
            if( edit_mode && this.can_manage_access())
                rows.push([
                    '',
                    TextInput_HTML({ classes: 'first', name: ''+range.id, value: range.first, size: 6 }),
                    TextInput_HTML({ classes: 'last',  name: ''+range.id, value: range.last,  size: 6 }),
                    '',
                    '',
                    ''
                ]);
            else
                rows.push([
                    this.can_manage_access() ?
                        Button_HTML('X', {
                            name:    range.id,
                            classes: 'admin-slacidnumbers-range-delete',
                            title:   'delete this range' }) : ' ',
                    range.first,
                    range.last,
                    range.last - range.first + 1,
                    count_elements_in_array(range.available),
                    Button_HTML('search', {
                            name:    range.id,
                            classes: 'admin-slacidnumbers-range-search',
                            title:   'search all equipment associated with this range' })
                ]);
        }
        if( edit_mode && this.can_manage_access())
            rows.push([
                '',
                TextInput_HTML({ classes: 'first', name: '0', value: '0', size: 6 }),
                TextInput_HTML({ classes: 'last',  name: '0', value: '0', size: 6 }),
                '',
                '',
                ''
            ]);

        var table = new Table('admin-slacidnumbers-ranges-table', [
            { name: 'DELETE', sorted: false,
              type: { after_sort: function() {
                            ranges_elem.find('.admin-slacidnumbers-range-delete').
                                button().
                                click(function() {
                                    var id = this.name;
                                    that.slacidnumbers_range_delete(id); });
                            ranges_elem.find('.admin-slacidnumbers-range-search').
                                button().
                                click(function() {
                                    var id = this.name;
                                    global_search_equipment_by_slacidnumber_range(id); });
                      }}},
            { name: 'first' },
            { name: 'last' },
            { name: '# total' },
            { name: '# available' },
            { name: 'in use', sorted: false }],
            rows
        );
        table.display();
    };
    this.slacidnumbers_ranges_edit = function() {
        this.slacidnumber_ranges_display(true);
    };
    this.slacidnumbers_range_delete = function(id) {
        var params = { range_id:id };
        var jqXHR = $.get('../irep/ws/slacidnumber_range_delete.php', params, function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.slacidnumber_range = data.range;
            that.slacidnumber_ranges_display(false);
        },
        'JSON').error(function () {
            report_error('failed to delete SLACid number range because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.slacidnumbers_ranges_edit_save = function() {
        var params = {range:''};
        var firsts = $('#admin-slacidnumbers-ranges-table').find('.first');
        var lasts  = $('#admin-slacidnumbers-ranges-table').find('.last');
        if(firsts.length != lasts.length) {
            report_error('slacidnumbers_ranges_edit_save: implementation error, please contact developers');
            return;
        }
        for( var i=0; i < firsts.length; i++) {
            var first = firsts[i];
            var last  = lasts [i];
            if( first.name != last.name) {
                report_error('slacidnumbers_ranges_edit_save: internal implementation error');
                return;
            }
            params.range += (params.range != '' ? ',' : '')+first.name+':'+first.value+':'+last.value;
        }
        var jqXHR = $.get('../irep/ws/slacidnumber_range_save.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
                that.slacidnumber_range = data.range;
                that.slacidnumber_ranges_display(false);
        },
        'JSON').error(function () {
            report_error('failed to save updated SLACid number ranges because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.slacidnumbers_ranges_edit_cancel = function() {
        this.slacidnumber_ranges_display(false);
    };

    this.slacidnumbers_load = function() {
        var params = {};
        var jqXHR = $.get('../irep/ws/slacidnumber_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.slacidnumber_range = data.range;
            that.slacidnumber_ranges_display(false);
        },
        'JSON').error(function () {
            report_error('failed to load SLACid numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };
    return this ;
}
var admin = new p_appl_admin() ;
