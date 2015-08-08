function p_appl_admin() {

    var that = this;

    this.when_done = null;

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
    this.name      = 'admin';
    this.full_name = 'Admin';
    this.context   = '';
    this.default_context = 'access';

    this.select = function(context,when_done) {
        that.context   = context;
        this.when_done = when_done;
        this.init();
    };
    this.select_default = function() {
        if( this.context == '' ) this.context = this.default_context;
        this.init();
    };
    this.if_ready2giveup = function(handler2call) {
        this.init();
        handler2call();
    };
    this.update = function() {
        this.init();
        this.access_load();
        this.notify_load();
        this.cablenumbers_load();
        this.jobnumbers_load();
    };
    
    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */
    this.initialized = false;
    this.init = function() {
        if( this.initialized ) return;
        this.initialized = true;

        $('#admin-access-reload'       ).button().click(function() { that.access_load      (); });
        $('#admin-notifications-reload').button().click(function() { that.notify_load      (); });
        $('#admin-cablenumbers-reload' ).button().click(function() { that.cablenumbers_load(); });
        $('#admin-jobnumbers-reload'   ).button().click(function() { that.jobnumbers_load  (); });

        var administrator2add = $('#admin-access').find('input[name="administrator2add"]');
        administrator2add.
            keyup(function(e) {
                var uid = $(this).val();
                if( uid == '' ) { return; }
                if( e.keyCode == 13 ) { that.access_create_administrator(uid); return; }});

        var projmanager2add = $('#admin-access').find('input[name="projmanager2add"]');
        projmanager2add.
            keyup(function(e) {
                var uid = $(this).val();
                if( uid == '' ) { return; }
                if( e.keyCode == 13 ) { that.access_create_projmanager(uid); return; }});

        var other2add = $('#admin-access').find('input[name="other2add"]');
        other2add.
            keyup(function(e) {
                var uid = $(this).val();
                if( uid == '' ) { return; }
                if( e.keyCode == 13 ) { that.access_create_other(uid); return; }});

        var submit_pending = $('#admin-notifications').find('button[name="submit_all"]').
            button().
            click(function() { that.notify_pending_submit(); });

        var delete_pending = $('#admin-notifications').find('button[name="delete_all"]').
            button().
            click(function() { that.notify_pending_delete(); });

        if(!this.can_manage_access()) {
            administrator2add.attr('disabled','disabled');
              projmanager2add.attr('disabled','disabled');
                    other2add.attr('disabled','disabled');
               submit_pending.attr('disabled','disabled');
               delete_pending.attr('disabled','disabled');
        }
        var cablenumbers_prefixes_edit   = $('#admin-cablenumbers').find('#prefixes').find('button[name="edit"]'  ).button().button('disable');
        var cablenumbers_prefixes_save   = $('#admin-cablenumbers').find('#prefixes').find('button[name="save"]'  ).button().button('disable');
        var cablenumbers_prefixes_cancel = $('#admin-cablenumbers').find('#prefixes').find('button[name="cancel"]').button().button('disable');

        var cablenumbers_ranges_edit   = $('#admin-cablenumbers').find('#ranges'  ).find('button[name="edit"]'  ).button().button('disable');
        var cablenumbers_ranges_save   = $('#admin-cablenumbers').find('#ranges'  ).find('button[name="save"]'  ).button().button('disable');
        var cablenumbers_ranges_cancel = $('#admin-cablenumbers').find('#ranges'  ).find('button[name="cancel"]').button().button('disable');

        var cablenumbers_orphan_scan        = $('#admin-cablenumbers').find('#orphan' ).find('button[name="scan"]'       ).button().button('disable');
        var cablenumbers_orphan_synchronize = $('#admin-cablenumbers').find('#orphan' ).find('button[name="synchronize"]').button().button('disable');

        var cablenumbers_reserved_scan = $('#admin-cablenumbers').find('#reserved' ).find('button[name="scan"]').button().button('disable');
        var cablenumbers_reserved_free = $('#admin-cablenumbers').find('#reserved' ).find('button[name="free"]').button().button('disable');

        if( this.can_manage_access()) {

            cablenumbers_prefixes_edit.button('enable');
            cablenumbers_prefixes_edit.click  (function() { that.cablenumbers_prefixes_edit       (); });
            cablenumbers_prefixes_save.click  (function() { that.cablenumbers_prefixes_edit_save  (); });
            cablenumbers_prefixes_cancel.click(function() { that.cablenumbers_prefixes_edit_cancel(); });

            cablenumbers_ranges_edit.click  (function() { that.cablenumbers_ranges_edit       (); });
            cablenumbers_ranges_save.click  (function() { that.cablenumbers_ranges_edit_save  (); });
            cablenumbers_ranges_cancel.click(function() { that.cablenumbers_ranges_edit_cancel(); });

            cablenumbers_orphan_scan.button       ('enable');
            cablenumbers_orphan_synchronize.button('enable');
            cablenumbers_orphan_scan.click       (function() { that.cablenumbers_orphan_scan       (); });
            cablenumbers_orphan_synchronize.click(function() { that.cablenumbers_orphan_synchronize(); });

            cablenumbers_reserved_scan.button('enable');
            cablenumbers_reserved_free.button('enable');
            cablenumbers_reserved_scan.click(function() { that.cablenumbers_reserved_scan(); });
            cablenumbers_reserved_free.click(function() { that.cablenumbers_reserved_free(); });
        }
        $('#admin-access'       ).find('#tabs').tabs();
        $('#admin-notifications').find('#tabs').tabs();
        $('#admin-cablenumbers' ).find('#tabs').tabs();
        $('#admin-jobnumbers'   ).find('#tabs').tabs();

//        this.access_load();
//        this.notify_load();
//        this.cablenumbers_load();
//        this.jobnumbers_load();
//        this.reload_timer_restart();
        this.reload_timer_event();
    };
    this.can_manage_access = function() { return global_current_user.is_administrator; };
    this.can_manage_notify = this.can_manage_access;

    this.reload_timer = null;
    this.reload_timer_restart = function() {
        this.reload_timer = window.setTimeout('admin.reload_timer_event()', 60000 );
    };
    this.reload_timer_event = function() {
        if( !this.jobnumber_editing() && !this.cablenumber_editing()) {
            this.access_load();
            this.notify_load();
            this.cablenumbers_load();
            this.jobnumbers_load();
        }
        this.reload_timer_restart();
    };

    /* -----------------
     *   Cable numbers
     * -----------------
     */
    this.cablenumber = null;

    this.cablenumber_prefix_editing = false;
    this.cablenumber_ranges_editing = false;
    this.cablenumber_editing = function() {
        return this.cablenumber_prefix_editing || this.cablenumber_ranges_editing;
    };

    this.cable_name2html = function(cnidx) {
        var c = this.cablenumber[cnidx];
        var html =
            c.recently_allocated_name == '' ?
            '' :
            '<a href="javascript:global_search_cable_by_cablenumber('+"'"+c.recently_allocated_name+"'"+')">'+c.recently_allocated_name+'</a>';
        return html;
    };
    this.cablenumbers_prefixes_edit_tools = function(edit_mode) {
        var cablenumbers_prefixes_edit   = $('#admin-cablenumbers').find('#prefixes').find('button[name="edit"]'  );
        var cablenumbers_prefixes_save   = $('#admin-cablenumbers').find('#prefixes').find('button[name="save"]'  );
        var cablenumbers_prefixes_cancel = $('#admin-cablenumbers').find('#prefixes').find('button[name="cancel"]');
        if( !this.can_manage_access()) return;
        cablenumbers_prefixes_edit.button  (edit_mode ? 'disable' : 'enable');
        cablenumbers_prefixes_save.button  (edit_mode ? 'enable'  : 'disable');
        cablenumbers_prefixes_cancel.button(edit_mode ? 'enable'  : 'disable');
    };
    this.cablenumber_prefixes_table = null;
    this.cablenumber_prefixes_display = function(edit_mode) {
        this.cablenumber_prefix_editing = edit_mode;
        var rows = [];
        for( var p in this.cablenumber_prefix ) {
            var prefix = this.cablenumber_prefix[p];
            var html = '';
            for( var l in prefix.location )            html += '<div>'+prefix.location[l]+'</div>';
            if( edit_mode && this.can_manage_access()) html += TextInput_HTML({ classes: 'location', name: prefix.name, value: '', size: 2 });
            rows.push([ prefix.name, html ]);
        }
        if( edit_mode && this.can_manage_access())
            rows.push([ TextInput_HTML({ classes: 'prefix', name: prefix.name, value: '', size: 2 }), '&nbsp;' ]);

        this.cablenumber_prefixes_table = new Table('admin-cablenumbers-prefixes-table', [
            { name: 'prefix',
              selectable: !edit_mode,
              type: {
                    select_action : function(prefix) { that.cablenumber_ranges_display(prefix,false); }}},
            { name: 'location (LLL)', sorted: false } ],
            rows,
            { default_sort_column: 0,
              selected_col: 0 }
        );
        this.cablenumber_prefixes_table.display();
        this.cablenumbers_prefixes_edit_tools(edit_mode);
        this.cablenumber_ranges_display(this.cablenumber_prefixes_table.selected_object(),false);
    };
    this.cablenumbers_prefixes_edit = function() {
        this.cablenumber_prefixes_display(true);
    };
    this.cablenumbers_prefixes_edit_save = function() {
        var params = {};
        var new_prefix = $('#admin-cablenumbers-prefixes-table').find('.prefix').val();
        if(new_prefix != '') params[new_prefix] = '';
        var new_locations = $('#admin-cablenumbers-prefixes-table').find('.location');
        for( var i=0; i < new_locations.length; i++) {
            var value = new_locations[i].value;
            if( value != '' )
                params[new_locations[i].name] = value;
        }
        var jqXHR = $.get('../neocaptar/ws/cablenumber_prefix_save.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.cablenumber_prefix = data.prefix;
            that.cablenumber_prefixes_display(false);
        },
        'JSON').error(function () {
            report_error('failed to load cable numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.cablenumbers_prefixes_edit_cancel = function() {
        this.cablenumber_prefixes_display(false);
    };
    this.cablenumbers_ranges_edit_tools = function(edit_mode) {
        var ranges_elem = $('#admin-cablenumbers').find('#ranges');
        var cablenumbers_ranges_edit   = ranges_elem.find('button[name="edit"]'  );
        var cablenumbers_ranges_save   = ranges_elem.find('button[name="save"]'  );
        var cablenumbers_ranges_cancel = ranges_elem.find('button[name="cancel"]');
        if( !this.can_manage_access()) return;
        cablenumbers_ranges_edit.button  (edit_mode ? 'disable' : 'enable');
        cablenumbers_ranges_save.button  (edit_mode ? 'enable'  : 'disable');
        cablenumbers_ranges_cancel.button(edit_mode ? 'enable'  : 'disable');
    };
    function count_elements_in_array(obj) {
        var size = 0;
        for( var key in obj ) size++;
        return size;
    }
    this.cablenumber_ranges_display = function(prefix_name,edit_mode) {
        this.cablenumber_ranges_editing = edit_mode;
        this.cablenumbers_ranges_edit_tools(edit_mode);
        var ranges_elem = $('#admin-cablenumbers').find('#ranges');
        var rows = [];
        for( var p in this.cablenumber_prefix ) {
            var prefix = this.cablenumber_prefix[p];
            if( prefix.name == prefix_name ) {
                for( var r in prefix.range ) {
                    var range = prefix.range[r];
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
                                    value:   prefix_name,
                                    classes: 'admin-cablenumbers-range-delete',
                                    title:   'delete this range from the prefix' }) : ' ',
                            range.first,
                            range.last,
                            range.last - range.first + 1,
                            count_elements_in_array(range.available),
                            Button_HTML('search', {
                                    name:    range.id,
                                    value:   prefix_name,
                                    classes: 'admin-cablenumbers-range-search',
                                    title:   'search all cables associated with this range and the prefix' })
                        ]);
                }
            }
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

        var table = new Table('admin-cablenumbers-ranges-table', [
            { name: 'DELETE', sorted: false,
              type: { after_sort: function() {
                            ranges_elem.find('.admin-cablenumbers-range-delete').
                                button().
                                click(function() {
                                    var id = this.name;
                                    var prefix_name = this.value;
                                    that.cablenumbers_range_delete(id,prefix_name); });
                            ranges_elem.find('.admin-cablenumbers-range-search').
                                button().
                                click(function() {
                                    var id = this.name;
                                    var prefix_name = this.value;
                                    global_search_cables_by_cablenumber_range(id); });
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
    this.cablenumbers_ranges_edit = function() {
        this.cablenumber_ranges_display(this.cablenumber_prefixes_table.selected_object(),true);
    };
    this.cablenumbers_range_delete = function(id,prefix_name) {
        var params = { prefix:prefix_name, range_id:id };
        var jqXHR = $.get('../neocaptar/ws/cablenumber_range_delete.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            for( var p in  that.cablenumber_prefix ) {
                var prefix = that.cablenumber_prefix[p];
                if( prefix.name == prefix_name ) {
                    that.cablenumber_prefix[p].range = data.range;
                    that.cablenumber_ranges_display(prefix_name, false);
                    break;
                }
            }
        },
        'JSON').error(function () {
            report_error('failed to load cable number ranges info because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.cablenumbers_ranges_edit_save = function() {
        var prefix2save = this.cablenumber_prefixes_table.selected_object();
        var params = {prefix:prefix2save,range:''};
        var firsts = $('#admin-cablenumbers-ranges-table').find('.first');
        var lasts  = $('#admin-cablenumbers-ranges-table').find('.last');
        if(firsts.length != lasts.length) {
            report_error('cablenumbers_ranges_edit_save: implementation error, please contact developers');
            return;
        }
        for( var i=0; i < firsts.length; i++) {
            var first = firsts[i];
            var last  = lasts [i];
            if( first.name != last.name) {
                report_error('cablenumbers_ranges_edit_save: internal implementation error');
                return;
            }
            params.range += (params.range != '' ? ',' : '')+first.name+':'+first.value+':'+last.value;
        }
        var jqXHR = $.get('../neocaptar/ws/cablenumber_range_save.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            for( var p in  that.cablenumber_prefix ) {
                var prefix = that.cablenumber_prefix[p];
                if( prefix.name == prefix2save ) {
                    that.cablenumber_prefix[p].range = data.range;
                    that.cablenumber_ranges_display(prefix2save, false);
                    break;
                }
            }
        },
        'JSON').error(function () {
            report_error('failed to load cable number ranges info because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.cablenumbers_ranges_edit_cancel = function() {
        this.cablenumber_ranges_display(this.cablenumber_prefixes_table.selected_object(),false);
    };

    this.cablenumbers_load = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/ws/cablenumber_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.cablenumber_prefix = data.prefix;
            that.cablenumber_prefixes_display(false);
        },
        'JSON').error(function () {
            report_error('failed to load cable numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };

    function cablenumbers_orphan2html(numbers) {
        var html = '<div>';
        var num = 0;
        for( var n in numbers ) {
            var number = numbers[n];
            html += '<a class="link" style="margin-right:8px;" href="javascript:global_search_cable_by_cablenumber(\''+number.cablenumber+'\')">'+number.cablenumber+'</a>';
            num = num + 1;
            if( num >= 8 ) {
                num = 0;
                html += '</div><div>';
            }
        }
        html += '</div>';
        return html;
    }

    this.cablenumbers_orphan_prefix = null;
    this.cablenumbers_orphan_table = null;

    this.cablenumbers_orphan_display = function() {
        var rows = [];
        for( var prefix_name in this.cablenumbers_orphan_prefix ) {
            var prefix = this.cablenumbers_orphan_prefix[prefix_name];
            rows.push([
                prefix_name,
                cablenumbers_orphan2html(prefix.out_of_range ),
                cablenumbers_orphan2html(prefix.in_range )
            ]);
        }
        this.cablenumbers_orphan_table = new Table('admin-orphan-table', [
            { name: 'prefix' },
            { name: 'out of range numbers', sorted: false },
            { name: 'numbers which can be synchronized', sorted: false } ],
            rows
        );
        this.cablenumbers_orphan_table.display();
    };
    this.cablenumbers_orphan_scan = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/ws/cablenumber_get_orphan.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.cablenumbers_orphan_prefix = data.prefix;
            that.cablenumbers_orphan_display();
        },
        'JSON').error(function () {
            report_error('failed to load orphan cable numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.cablenumbers_orphan_synchronize = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/ws/cablenumber_sync_orphan.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.cablenumbers_orphan_prefix = data.prefix;
            that.cablenumbers_orphan_display();
        },
        'JSON').error(function () {
            report_error('failed to load orphan cable numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };

    function cablenumbers_reserved2html(numbers) {
        var html = '<div>';
        var num = 0;
        for( var n in numbers ) {
            var number = numbers[n];
            html += '<a class="link" style="margin-right:8px;" href="javascript:global_search_cable_by_id(\''+number.cable_id+'\')">'+number.cablenumber+'</a>';
            num = num + 1;
            if( num >= 8 ) {
                num = 0;
                html += '</div><div>';
            }
        }
        html += '</div>';
        return html;
    }

    this.cablenumbers_reserved_prefix = null;
    this.cablenumbers_reserved_table = null;

    this.cablenumbers_reserved_display = function() {
        var rows = [];
        for( var prefix_name in this.cablenumbers_reserved_prefix ) {
            var prefix = this.cablenumbers_reserved_prefix[prefix_name];
            rows.push([
                prefix_name,
                cablenumbers_reserved2html(prefix)
            ]);
        }
        this.cablenumbers_reserved_table = new Table('admin-reserved-table', [
            { name: 'prefix' },
            { name: 'numbers which can be freed', sorted: false } ],
            rows
        );
        this.cablenumbers_reserved_table.display();
    };
    this.cablenumbers_reserved_scan = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/ws/cablenumber_get_reserved.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.cablenumbers_reserved_prefix = data.prefix;
            that.cablenumbers_reserved_display();
        },
        'JSON').error(function () {
            report_error('failed to load reserved cable numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.cablenumbers_reserved_free = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/ws/cablenumber_free_reserved.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.cablenumbers_reserved_prefix = data.prefix;
            that.cablenumbers_reserved_display();
        },
        'JSON').error(function () {
            report_error('failed to load reserved cable numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };

    /* ---------------
     *   Job numbers
     * ---------------
     */
    this.jobnumber            = null;
    this.jobnumber_allocation = null;

    this.jobnumber_editing = function() {
        if( !this.jobnumber ) return false;
        var result = false;
        for( var i in this.jobnumber ) result = result || this.jobnumber[i].editing;
        return result;
    };

    this.job_name2html = function(name) {
        var html =
            name == '' ?
            '' :
            '<a href="javascript:global_search_projects_by_jobnumber('+"'"+name+"'"+')" title="find a project correponding to this number">'+name+'</a>';
        return html;
    };
    this.jobnumber2html = function(jnidx) {
        var j = this.jobnumber[jnidx];
        var html =
            '<tr id="admin-jobnumbers-'+jnidx+'" >'+
            '  <td nowrap="nowrap" class="table_cell table_cell_left " id="admin-jobnumbers-tools-'+jnidx+'">'+
            '    <button class="admin-jobnumbers-tools " name="edit"        onclick="admin.jobnumbers_edit       ('+jnidx+')" title="edit"                                 ><b>E</b></button>'+
            '    <button class="admin-jobnumbers-tools " name="edit_save"   onclick="admin.jobnumbers_edit_save  ('+jnidx+')" title="save changes to the database"         >save</button>'+
            '    <button class="admin-jobnumbers-tools " name="edit_cancel" onclick="admin.jobnumbers_edit_cancel('+jnidx+')" title="cancel editing and ignore any changes">cancel</button>'+
            '  </td>'+
            '  <td nowrap="nowrap" class="table_cell "                                          >&nbsp;'+j.owner                  +'</td>'+
            '  <td nowrap="nowrap" class="table_cell                  prefix "                  >&nbsp;'+j.prefix                 +'</td>'+
            '  <td nowrap="nowrap" class="table_cell                  range "                   >&nbsp;'+j.first+' - '+j.last     +'</td>'+
            '  <td nowrap="nowrap" class="table_cell                  num_in_use "              >&nbsp;'+
            '    <button class="admin-jobnumbers-search" name="'+jnidx+'" title="search all cables using job numbers in this range">search</button>'+
            '</td>'+
            '  <td nowrap="nowrap" class="table_cell                  num_available "           >&nbsp;'+j.num_available          +'</td>'+
            '  <td nowrap="nowrap" class="table_cell                  next_available "          >&nbsp;'+j.next_available         +'</td>'+
            '  <td nowrap="nowrap" class="table_cell                  recently_allocated_name " >&nbsp;'+this.job_name2html(j.recently_allocated_name)+'</td>'+
            '  <td nowrap="nowrap" class="table_cell                  recent_allocation_time "  >&nbsp;'+j.recent_allocation_time +'</td>'+
            '  <td nowrap="nowrap" class="table_cell                  recent_allocation_uid "   >&nbsp;'+j.recent_allocation_uid  +'</td>'+
            '</tr>';
        return html;
    };
    this.jobnumbers_allocation2html = function(jnaidx) {
        var ja = this.jobnumber_allocation[jnaidx];
        var html =
            '<tr id="admin-jobnumber-allocations-'+jnaidx+'" >'+
            '  <td nowrap="nowrap" class="table_cell table_cell_left " >&nbsp;'+this.job_name2html(ja.jobnumber_name)+'</td>'+
            '  <td nowrap="nowrap" class="table_cell                 " >&nbsp;'+ja.owner         +'</td>'+
            '  <td nowrap="nowrap" class="table_cell                 " >&nbsp;'+ja.num_cables    +'</td>'+
            '  <td nowrap="nowrap" class="table_cell                 " >&nbsp;'+ja.allocated_time+'</td>'+
            '  <td nowrap="nowrap" class="table_cell                 " >&nbsp;'+ja.project_title +'</td>'+
            '</tr>';
        return html;
    };
    this.jobnumbers_display = function() {
        var html =
            '<table><tbody>'+
            '  <tr>'+
            '    <td nowrap="nowrap" class="table_hdr " >TOOLS</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >owner</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >prefix</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >range</td>'+
            '    <td nowrap="nowrap" class="table_hdr " ># in use</td>'+
            '    <td nowrap="nowrap" class="table_hdr " ># available</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >next</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >previously allocated</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >last allocation</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >requested by</td>'+
            '  </tr>';
        for( var jnidx in this.jobnumber ) html += this.jobnumber2html(jnidx);
        html +=
            '</tbody></table>';
        $('#admin-jobnumbers-jobnumbers').html(html);
        for( var jnidx in this.jobnumber ) {
            $('#admin-jobnumbers-tools-'+jnidx+' button.admin-jobnumbers-tools').
                button().
                button(this.can_manage_access()?'enable':'disable');
            this.jobnumbers_update_tools(jnidx,false);
        }
        html =
            '<table><tbody>'+
            '  <tr>'+
            '    <td nowrap="nowrap" class="table_hdr " >job</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >owner</td>'+
            '    <td nowrap="nowrap" class="table_hdr " ># cables</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >allocated</td>'+
            '    <td nowrap="nowrap" class="table_hdr " >project</td>'+
            '  </tr>';
        for( var jnaidx in this.jobnumber_allocation ) html += this.jobnumbers_allocation2html(jnaidx);
        html +=
            '</tbody></table>';
        $('#admin-jobnumbers-allocations').html(html);
        $('.admin-jobnumbers-search').
            button().
            click(function () {
                var jnidx = this.name;
                var jn = that.jobnumber[jnidx];
                global_search_projects_by_jobnumber_prefix(jn.prefix);
            });
    };
    this.jobnumbers_update = function(jnidx) {
        var j = this.jobnumber[jnidx];
        j.editing = false;
        $('#admin-jobnumbers-'+jnidx+' .prefix'                 ).html('&nbsp;'+j.prefix);
        $('#admin-jobnumbers-'+jnidx+' .range'                  ).html('&nbsp;'+j.first+' - '+j.last);
        $('#admin-jobnumbers-'+jnidx+' .num_in_use'             ).html('&nbsp;'+j.num_in_use);
        $('#admin-jobnumbers-'+jnidx+' .num_available'          ).html('&nbsp;'+j.num_available);
        $('#admin-jobnumbers-'+jnidx+' .next_available'         ).html('&nbsp;'+j.next_available);
        $('#admin-jobnumbers-'+jnidx+' .recently_allocated_name').html('&nbsp;'+this.job_name2html(j.recently_allocated_name));
        $('#admin-jobnumbers-'+jnidx+' .recent_allocation_time' ).html('&nbsp;'+j.recent_allocation_time);
        $('#admin-jobnumbers-'+jnidx+' .recent_allocation_uid'  ).html('&nbsp;'+j.recent_allocation_uid);
        this.jobnumbers_update_tools(jnidx,false);
    };
    this.jobnumbers_update_tools = function(jnidx,editing) {
        if(!this.can_manage_access()) {
            this.jobnumbers_disable_tools(jnidx);
            return;
        }
        if( editing ) {
            $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit"]'       ).button('disable');
            $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit_save"]'  ).button('enable' );
            $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit_cancel"]').button('enable' );
            return;
        }
        $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit"]'       ).button('enable' );
        $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit_save"]'  ).button('disable');
        $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit_cancel"]').button('disable');
    };
    this.jobnumbers_disable_tools = function(jnidx) {
        $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit"]'       ).button('disable');
        $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit_save"]'  ).button('disable');
        $('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit_cancel"]').button('disable');
    };
    this.jobnumbers_edit = function(jnidx) {
        this.jobnumbers_update_tools(jnidx,true);
        var c = this.jobnumber[jnidx];
        c.editing = true;
        $('#admin-jobnumbers-'+jnidx+' .range').html(
            '<input type="text" size=2 style="text-align:right" name="first" value="'+c.first+'" /><input type="text" size=2 style="text-align:right" name="last" value="'+c.last+'" />'
        );
        if(c.prefix == '')
            $('#admin-jobnumbers-'+jnidx+' .prefix').html(
                '<input type="text" size=2 name="prefix" value="'+c.prefix+'" />'
            );
    };
    this.jobnumbers_edit_save = function(jnidx) {
        this.jobnumbers_disable_tools(jnidx);
        var c = this.jobnumber[jnidx];
        var first  = parseInt($('#admin-jobnumbers-'+jnidx).find('input[name="first"]' ).val());
        var last   = parseInt($('#admin-jobnumbers-'+jnidx).find('input[name="last"]'  ).val());
        if( last <= first ) {
            report_error('invalid range: last number must be strictly larger than the first one', null);
            this.jobnumbers_update_tools(jnidx,true);
            return;
        }
        var params = {id:c.id,first:first,last:last};
        if(c.prefix == '') {
            var prefix = $('#admin-jobnumbers-'+jnidx).find('input[name="prefix"]').val();
            if( prefix != '' ) params.prefix = prefix;
        }
        var jqXHR = $.get('../neocaptar/ws/jobnumber_save.php',params,function(data) {
            if(data.status != 'success') {
                report_error(data.message, null);
                that.jobnumbers_update_tools(jnidx,true);
                return;
            }
            that.jobnumber[jnidx] = data.jobnumber;
            that.jobnumbers_update(jnidx);
        },
        'JSON').error(function () {
            report_error('failed to contact the Web service due to: '+jqXHR.statusText, null);
            that.jobnumbers_update_tools(jnidx,true);
            return;
        });
    };
    this.jobnumbers_edit_cancel = function(jnidx) {
       this.jobnumbers_update(jnidx);
    };
    this.jobnumbers_load = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/ws/jobnumber_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.jobnumber = data.jobnumber;
            that.jobnumber_allocation = data.jobnumber_allocation;
            that.jobnumbers_display();
        },
        'JSON').error(function () {
            report_error('failed to load job numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };


    /* ------------------
     *   Access control
     * ------------------
     */
    this.access = null;
    this.access_create_administrator = function(uid) { this.access_create_user(uid,'ADMINISTRATOR'); };
    this.access_create_projmanager   = function(uid) { this.access_create_user(uid,'PROJMANAGER'); };
    this.access_create_other         = function(uid) { this.access_create_user(uid,'OTHER'); };
 
    this.projmanagers = function(managers_only) {

        // Return an array of user accounts who are allowed to manage
        // projects. This will include dedicated project managers as
        // as well as administrators.
        //
        // Fall back to a static array of 'global_users' if no users have been
        // dynamically loaded so far.
        //
        if( this.access ) {
            var result = [];
            var users = undefined;
            if(managers_only)
                users = $.merge(
                    [],
                    this.access.PROJMANAGER
                );
            else
                users = $.merge(
                    $.merge(
                        [],
                        this.access.ADMINISTRATOR
                    ),
                    this.access.PROJMANAGER );
            for( var i in users ) {
                var user = users[i];
                result.push(user.uid);
            }
            return result;
        }
        return global_users;
    };
    this.access_display_users = function(id,users,is_administrator,can_manage_projects) {
        var rows = [];
        for( var i in users ) {
            var a = users[i];
            var row = [];
            if(this.can_manage_access()) row.push(
                Button_HTML('X', {
                    name:    'delete',
                    onclick: "admin.access_delete_user('"+a.uid+"')",
                    title:   'delete this user from the list' }));
            row.push(
                a.uid,
                a.name);
            if(can_manage_projects && !is_administrator) row.push(
                this.can_manage_access()
                    ? Checkbox_HTML({
                        name:    'dict_priv',
                        onclick: "admin.access_toggle_priv('"+a.uid+"','dict_priv')",
                        title:   'togle the dictionary privilege',
                        checked: a.privilege.dict_priv ? 'checked' : '' })
                    : ( a.privilege.dict_priv ? 'Yes' : 'No' ));
            row.push(
                a.added_time,
                a.last_active_time);

            if(can_manage_projects) row.push(
                Button_HTML('search', {
                    name:    'projects',
                    onclick: "global_search_projects_by_owner('"+a.uid+"')",
                    title:   'search projects owned by this user' }));
            if(can_manage_projects && !is_administrator) row.push(
                Button_HTML('search', {
                    name:    'projects',
                    onclick: "global_search_projects_coowned_by('"+a.uid+"')",
                    title:   'search projects co-managed with other users' }));

            rows.push(row);
        }
        var hdr = [];
        if(this.can_manage_access()) hdr.push(                  { name: 'DELETE',               sorted: false });
        hdr.push(                                               { name: 'UID',                  sorted: false },
                                                                { name: 'user',                 sorted: false });
        if(can_manage_projects && !is_administrator) hdr.push(  { name: 'dictionary privilege', sorted: false });
        hdr.push(                                               { name: 'added',                sorted: false },
                                                                { name: 'last active',          sorted: false });
        if(can_manage_projects) hdr.push(                       { name: 'OWN PROJECTS',         sorted: false });
        if(can_manage_projects && !is_administrator) hdr.push(  { name: 'CO-MANAGED PROJECTS',  sorted: false });

        var table = new Table(id, hdr, rows);
        table.display();

        var elem = $('#'+id);

        elem.find('button[name="delete"]').button();
        elem.find('button[name="projects"]').button();
    };

    this.access_display = function() {
        this.access_display_users('admin-access-ADMINISTRATOR', this.access.ADMINISTRATOR, true, true);
        this.access_display_users('admin-access-PROJMANAGER',   this.access.PROJMANAGER,   false, true);
        this.access_display_users('admin-access-OTHER',         this.access.OTHER,         false, false);
    };

    this.access_load = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/ws/access_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.access = data.access;
            that.access_display();
        },
        'JSON').error(function () {
            report_error('failed to load access control info because of: '+jqXHR.statusText, null);
            return;
        });
    };

    this.access_create_user = function(uid,role) {
        var params = {uid:uid,role:role};
        var jqXHR = $.get('../neocaptar/ws/access_new.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.access = data.access;
            that.access_display();
            that.notify_load();
        },
        'JSON').error(function () {
            report_error('failed to add a new user to the access control list because of: '+jqXHR.statusText, null);
            return;
        });
    };

    this.access_delete_user = function(uid) {
        var params = {uid:uid};
        var jqXHR = $.get('../neocaptar/ws/access_delete.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.access = data.access;
            that.access_display();
            that.notify_load();
        },
        'JSON').error(function () {
            report_error('failed to delete this user from the access control list because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.access_toggle_priv = function(uid,name) {
        var params = {uid:uid, name:name};
        var jqXHR = $.get('../neocaptar/ws/access_toggle_priv.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.access = data.access;
            that.access_display();
            that.notify_load();
        },
        'JSON').error(function () {
            report_error('failed to togle the privilege state because of: '+jqXHR.statusText, null);
            return;
        });    };


    /* -----------------
     *   Notifications
     * -----------------
     */
    this.notify             = null;
    this.notify_event_types = null;

    this.notify_display_projmanager = function() {
        var role = 'PROJMANAGER';
        if( !global_current_user.can_manage_projects ) {
            $('#admin-notifications-'+role).html(
                '<div style="color:maroon;">'+
                '  We are sorry! Your account doesn\'t have sufficient privileges to manage projects.'+
                '</div>');
            return;
        }
        this.notify_display_impl(role);
    };
    this.notify_display_administrator = function() {  this.notify_display_impl('ADMINISTRATOR');  };
    this.notify_display_other         = function() {  this.notify_display_impl('OTHER');          };

    this.notify_display_impl = function(role) {

        var hdr  = [ {name: 'event', sorted: false } ];
        var rows = [];

        switch( role ) {

            case 'PROJMANAGER':

                hdr.push({ name: 'notify', sorted: false });
                for( var i in this.notify_event_types[role] ) {
                    var event  = this.notify_event_types[role][i];
                    var notify = this.notify[role][global_current_user.uid] || {};
                    var attr = {
                        classes: event.name,
                        name: global_current_user.uid,
                        checked: notify[event.name] == true };
                    rows.push([
                        event.description,
                        Checkbox_HTML(attr) ]);
                }
                break;

            case 'ADMINISTRATOR':
            case 'OTHER':

                for( var i in this.notify_access[role] ) {
                    var user = this.notify_access[role][i];
                    hdr.push({name: user.uid, sorted: false});
                }
                for( var i in this.notify_event_types[role] ) {
                    var event = this.notify_event_types[role][i];
                    var row   = [ event.description ];
                    for( var j in this.notify_access[role] ) {
                        var user = this.notify_access[role][j];
                        var notify = this.notify[role][user.uid] || {};
                        var attr = {
                            classes: event.name,
                            name:    user.uid,
                            checked: notify[event.name] == true };
                        if( !( this.can_manage_notify() || ( global_current_user.uid == user.uid )))
                            attr.disabled = 'disabled';
                        row.push( Checkbox_HTML(attr));
                    }
                    rows.push(row);
                }
                break;
        }
        var table = new Table('admin-notifications-'+role, hdr, rows );
        table.display();

        var policy = $('#admin-notifications').find('select[name="policy4'+role+'"]');
        policy.val(this.notify_schedule[role]);
        policy.change(function() {
            that.notify_schedule_save(role, $(this).val());
        });
        if(this.can_manage_notify()) policy.removeAttr('disabled');
        $('#admin-notifications-'+role).find('input[type="checkbox"]').click(function() {
            that.notify_save(role, this.name, $(this).attr('class'), $(this).is(':checked'));
        });
    };
    this.notify_display_pending = function() {
        var hdr = [
            { name:   'time' },
            { name:   'event' },
            { name:   'originator' },
            { name:   'recipient' },
            { name:   'recipient_role' },
            { name:   'ACTIONS',
              sorted: false,
              type:   { after_sort: function() { $('.admin-notifications-pending-tools').button(); }}}
        ];
        var rows = [];
        for( var i in this.notify_pending ) {
            var entry = this.notify_pending[i];
            var find_button = '';
            if(( entry.scope == 'CABLE' ) && ( entry.event_type_name != 'on_cable_delete')) {
                find_button = Button_HTML('find cable', {
                        name:    'find_'+i,
                        classes: 'admin-notifications-pending-tools',
                        onclick: "global_search_cable_by_id('"+entry.cable_id+"')",
                        title:   'display this cable if it is still available'
                    });
            } else if(( entry.scope == 'PROJECT' ) && ( entry.event_type_name != 'on_project_delete')) {
                find_button = Button_HTML('find project', {
                        name:    'find_'+i,
                        classes: 'admin-notifications-pending-tools',
                        onclick: "global_search_project_by_id('"+entry.project_id+"')",
                        title:   'display this project if it is still available'
                    });
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
                ]);
        }
        var table = new Table('admin-notifications-pending', hdr, rows );
        table.display();
        
        $('.admin-notifications-pending-tools').button();
    };
    this.notify_display = function() {
        this.notify_display_projmanager();
        this.notify_display_administrator();
        this.notify_display_other();
        this.notify_display_pending();
    };

    this.notify_action = function(url,params) {
        var jqXHR = $.get(url,params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }

            that.notify_access      = data.access;
            that.notify_event_types = data.event_types;
            that.notify_schedule    = data.schedule;
            that.notify             = data.notify;
            that.notify_pending     = data.pending;

            that.notify_display();
        },
        'JSON').error(function () {
            report_error('failed to load noditications list because of: '+jqXHR.statusText, null);
            return;
        });
    };
    this.notify_load = function() {
        this.notify_action(
            '../neocaptar/ws/notify_get.php',
            {}
        );
    };
    this.notify_save = function(recipient, uid, event_name, enabled) {
        this.notify_action(
            '../neocaptar/ws/notify_save.php',
            {   recipient:  recipient,
                uid:        uid,
                event_name: event_name,
                enabled:    enabled ? 1 : 0 }
        );
    };
    this.notify_schedule_save = function(recipient, policy) {
        this.notify_action(
            '../neocaptar/ws/notify_save.php',
            {   recipient:  recipient,
                policy:     policy }
        );
    };
    this.notify_pending_submit = function(idx) {
        var params = { action: 'submit' };
        if( idx != undefined ) params.id = this.notify_pending[idx].id;
        this.notify_action(
            '../neocaptar/ws/notify_queue.php',
            params
        );
    };
    this.notify_pending_delete = function(idx) {
        var params = { action: 'delete' };
        if( idx != undefined ) params.id = this.notify_pending[idx].id;
        this.notify_action(
            '../neocaptar/ws/notify_queue.php',
            params
        );
    };
    return this;
}
var admin = new p_appl_admin();
