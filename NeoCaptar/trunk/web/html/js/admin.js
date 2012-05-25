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

		$('#admin-access-reload'       ).button().click(function() { that.access_load       (); });
		$('#admin-notifications-reload').button().click(function() { that.notify_load(); });
		$('#admin-cablenumbers-reload' ).button().click(function() { that.cablenumbers_load (); });
		$('#admin-jobnumbers-reload'   ).button().click(function() { that.jobnumbers_load   (); });

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

        $('#admin-access').find('#tabs').tabs();
        $('#admin-notifications').find('#tabs').tabs();
        $('#admin-cablenumbers').find('#tabs').tabs();
        $('#admin-jobnumbers').find('#tabs').tabs();

		this.access_load();
        this.notify_load();
		this.cablenumbers_load();
		this.jobnumbers_load();
	};
    this.can_manage_access = function() { return global_current_user.is_administrator; };
    this.can_manage_notify = this.can_manage_access;

    /* -----------------
     *   Cable numbers
     * -----------------
     */
    this.cablenumber = null;

    this.cable_name2html = function(cnidx) {
        var c = this.cablenumber[cnidx];
        var html =
            c.recently_allocated_name == '' ?
            '' :
            '<a href="javascript:global_search_cable_by_cablenumber('+"'"+c.recently_allocated_name+"'"+')">'+c.recently_allocated_name+'</a>';
        return html;
    };
	this.cablenumbers_display = function() {

        var rows = [];
        for( var cnidx in this.cablenumber ) {
            var c = this.cablenumber[cnidx];
            rows.push([
                this.can_manage_access() ?
                    Button_HTML('E', {
                        name:    'edit_'+cnidx,
                        classes: 'admin-cablenumbers-tools',
                        onclick: "admin.cablenumbers_edit('"+cnidx+"')",
                        title:   'edit'
                    })+' '+
                    Button_HTML('save', {
                        name:    'save_'+cnidx,
                        classes: 'admin-cablenumbers-tools',
                        onclick: "admin.cablenumbers_edit_save('"+cnidx+"')",
                        title:   'save changes to the database'
                    })+' '+
                    Button_HTML('cancel', {
                        name:    'edit_cancel_'+cnidx,
                        classes: 'admin-cablenumbers-tools',
                        onclick: "admin.cablenumbers_edit_cancel('"+cnidx+"')",
                        title:   'cancel editing and ignore any changes'
                    }) : '',

                '<div name="location_'+cnidx+'">'+c.location          +'</div>',
                '<div name="prefix_'  +cnidx+'">'+c.prefix            +'</div>',
                '<div name="range_'   +cnidx+'">'+c.first+' - '+c.last+'</div>',

                Button_HTML('search', {
                    name:    cnidx,
                    classes: 'admin-cablenumbers-search',
                    title:   'search all cables using this range'
                }),

                '<div name="num_available_'          +cnidx+'">'+c.num_available            +'</div>',
                '<div name="next_available_'         +cnidx+'">'+c.next_available           +'</div>',
                '<div name="recently_allocated_name_'+cnidx+'">'+this.cable_name2html(cnidx)+'</div>',
                '<div name="recent_allocation_time_' +cnidx+'">'+c.recent_allocation_time   +'</div>',
                '<div name="recent_allocation_uid_'  +cnidx+'">'+c.recent_allocation_uid    +'</div>'
            ]);
        }
        var table = new Table('admin-cablenumbers-cablenumbers', [
            { name: 'TOOLS',                sorted: false },
            { name: 'location',             sorted: false },
            { name: 'prefix',               sorted: false },
            { name: 'range',                sorted: false },
            { name: 'in use',               sorted: false },
            { name: '# available',          sorted: false },
            { name: 'next',                 sorted: false },
            { name: 'previously allocated', sorted: false },
            { name: 'last allocation',      sorted: false },
            { name: 'requested by',         sorted: false } ],
            rows
        );
        table.display();
        for( var cnidx in this.cablenumber ) {
			$('#admin-cablenumbers-cablenumbers').find('button.admin-cablenumbers-tools').button();
            this.cablenumbers_update_tools(cnidx,false);
        }
        $('.admin-cablenumbers-search').
            button().
            click(function () {
                var cnidx = this.name;
                var cn = that.cablenumber[cnidx];
                global_search_cables_by_prefix(cn.prefix);
            });
    };
	this.cablenumbers_update = function(cnidx) {
        var c = this.cablenumber[cnidx];
        var elem = $('#admin-cablenumbers-cablenumbers');
        elem.find('div[name="prefix_'                 +cnidx+'"]').html(c.prefix);
        elem.find('div[name="range_'                  +cnidx+'"]').html(c.first+' - '+c.last);
        elem.find('div[name="num_available_'          +cnidx+'"]').html(c.num_available);
        elem.find('div[name="next_available_'         +cnidx+'"]').html(c.next_available);
        elem.find('div[name="recently_allocated_name_'+cnidx+'"]').html(this.cable_name2html(cnidx));
        elem.find('div[name="recent_allocation_time_' +cnidx+'"]').html(c.recent_allocation_time);
        elem.find('div[name="recent_allocation_uid_'  +cnidx+'"]').html(c.recent_allocation_uid);
        this.cablenumbers_update_tools(cnidx,false);
    };
    this.cablenumbers_update_tools = function(cnidx,editing) {
        if(!this.can_manage_access()) {
            this.cablenumbers_tools_disable(cnidx);
            return;
        }
        var elem = $('#admin-cablenumbers-cablenumbers');
		if( editing ) {
			elem.find('button[name="edit_'       +cnidx+'"]').button('disable');
			elem.find('button[name="edit_save_'  +cnidx+'"]').button('enable' );
			elem.find('button[name="edit_cancel_'+cnidx+'"]').button('enable' );
			return;
		}
		elem.find('button[name="edit_'       +cnidx+'"]').button('enable' );
		elem.find('button[name="edit_save_'  +cnidx+'"]').button('disable');
		elem.find('button[name="edit_cancel_'+cnidx+'"]').button('disable');
    };
    this.cablenumbers_tools_disable = function(cnidx) {
        var elem = $('#admin-cablenumbers-cablenumbers');
		elem.find('button[name="edit_'       +cnidx+'"]').button('disable');
		elem.find('button[name="edit_save_'  +cnidx+'"]').button('disable');
		elem.find('button[name="edit_cancel_'+cnidx+'"]').button('disable');
    };
    this.cablenumbers_edit = function(cnidx) {
        this.cablenumbers_update_tools(cnidx,true);
        var c = this.cablenumber[cnidx];
        var elem = $('#admin-cablenumbers-cablenumbers');
        elem.find('div[name="range_'+cnidx+'"]').html(
            '<input type="text" size=2 style="text-align:right" name="first" value="'+c.first+'" /><input type="text" size=2 style="text-align:right" name="last" value="'+c.last+'" />'
        );
        if(c.prefix == '')
            elem.find('div[name="prefix_'+cnidx+'"]').html(
                '<input type="text" size=2 name="prefix" value="'+c.prefix+'" />'
            );
    };
    this.cablenumbers_edit_save = function(cnidx) {
        this.cablenumbers_tools_disable(cnidx);
        var c     = this.cablenumber[cnidx];
        var elem  = $('#admin-cablenumbers-cablenumbers');
        var first = parseInt(elem.find('div[name="range_'+cnidx+'"]').find('input[name="first"]' ).val());
        var last  = parseInt(elem.find('div[name="range_'+cnidx+'"]').find('input[name="last"]'  ).val());
        if( last <= first ) {
            report_error('invalid range: last number must be strictly larger than the first one', null);
            this.cablenumbers_update_tools(cnidx,true);
            return;
        }
        var params = { id: c.id, first:first, last:last };
        if(c.prefix == '') {
            var prefix = elem.find('div[name="range_'+cnidx+'"]').find('input[name="prefix"]').val();
            if( prefix != '' ) params.prefix = prefix;
        }
        var jqXHR = $.get('../neocaptar/cablenumber_save.php',params,function(data) {
            if(data.status != 'success') {
                report_error(data.message, null);
                that.cablenumbers_update_tools(cnidx,true);
                return;
            }
            that.cablenumber[cnidx] = data.cablenumber;
            that.cablenumbers_update(cnidx);
        },
        'JSON').error(function () {
            report_error('failed to contact the Web service due to: '+jqXHR.statusText, null);
            that.cablenumbers_update_tools(cnidx,true);
            return;
        });
    };
    this.cablenumbers_edit_cancel = function(cnidx) {
       this.cablenumbers_update(cnidx);
    };
	this.cablenumbers_load = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/cablenumber_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.cablenumber = data.cablenumber;
            that.cablenumbers_display();
        },
        'JSON').error(function () {
            report_error('failed to load cable numbers info because of: '+jqXHR.statusText, null);
            return;
        });
    };

    /* ---------------
     *   Job numbers
     * ---------------
     */
    this.jobnumber            = null;
    this.jobnumber_allocation = null;

    this.job_name2html = function(name) {
        var html =
            name == '' ?
            '' :
            '<a href="javascript:global_search_cables_by_jobnumber('+"'"+name+"'"+')">'+name+'</a>';
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
                global_search_cables_by_jobnumber_prefix(jn.prefix);
            });
    };
	this.jobnumbers_update = function(jnidx) {
        var j = this.jobnumber[jnidx];
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
        var jqXHR = $.get('../neocaptar/jobnumber_save.php',params,function(data) {
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
        var jqXHR = $.get('../neocaptar/jobnumber_get.php',params,function(data) {
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
 
    this.projmanagers = function() {

        // Return an array of user accounts who are allowed to manage
        // projects. This will include dedicated project managers as
        // as well as administrators.
        //
        // Fall back to a static array of 'global_users' if no users have been
        // dynamically loaded so far.
        //
        if( this.access ) {
            var result = [];
            var users = $.merge(
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
    this.access_display_users = function(id,users) {
        var rows = [];
        for( var i in users ) {
            var a = users[i];
            rows.push([
                this.can_manage_access() ? 
                    Button_HTML('delete', {
                        name:    'delete',
                        onclick: "admin.access_delete_user('"+a.uid+"')",
                        title:   'delete this user from the list'
                    }) : '',
                a.uid,
                a.name,
                a.added_time,
                a.last_active_time,
                Button_HTML('search', {
                    name:    'projects',
                    onclick: "global_search_projects_by_owner('"+a.uid+"')",
                    title:   'search projects owned by this user'
                })
            ]);            
        }
        var table = new Table(id, [
            { name: '',            sorted: false },
            { name: 'UID',         sorted: false },
            { name: 'user',        sorted: false },
            { name: 'added',       sorted: false },
            { name: 'last active', sorted: false },
            { name: 'projects',    sorted: false } ],
            rows );
            table.display();

        var elem = $('#'+id);

        elem.find('button[name="delete"]').button();
        elem.find('button[name="projects"]').button();
    };

    this.access_display = function() {
        this.access_display_users('admin-access-ADMINISTRATOR', this.access.ADMINISTRATOR);
        this.access_display_users('admin-access-PROJMANAGER',   this.access.PROJMANAGER);
        this.access_display_users('admin-access-OTHER',         this.access.OTHER);
    };

    this.access_load = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/access_get.php',params,function(data) {
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
        var jqXHR = $.get('../neocaptar/access_new.php',params,function(data) {
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
        var jqXHR = $.get('../neocaptar/access_delete.php',params,function(data) {
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
                    }) : ''
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
            '../neocaptar/notify_get.php',
            {}
        );
    };
    this.notify_save = function(recipient, uid, event_name, enabled) {
        this.notify_action(
            '../neocaptar/notify_save.php',
            {   recipient:  recipient,
                uid:        uid,
                event_name: event_name,
                enabled:    enabled ? 1 : 0 }
        );
    };
    this.notify_schedule_save = function(recipient, policy) {
        this.notify_action(
            '../neocaptar/notify_save.php',
            {   recipient:  recipient,
                policy:     policy }
        );
    };
    this.notify_pending_submit = function(idx) {
        var params = { action: 'submit' };
        if( idx != undefined ) params.id = this.notify_pending[idx].id;
        this.notify_action(
            '../neocaptar/notify_queue.php',
            params
        );
    };
    this.notify_pending_delete = function(idx) {
        var params = { action: 'delete' };
        if( idx != undefined ) params.id = this.notify_pending[idx].id;
        this.notify_action(
            '../neocaptar/notify_queue.php',
            params
        );
    };
    return this;
}
var admin = new p_appl_admin();
