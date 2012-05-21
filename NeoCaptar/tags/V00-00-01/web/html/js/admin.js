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
		this.load_access();
        this.load_notify();
        this.load_cablenumbers();
        this.load_jobnumbers();
    };
    
    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */
	this.initialized = false;
	this.init = function() {
		if( this.initialized ) return;
		this.initialized = true;

		$('#admin-access-reload'       ).button().click(function() { that.load_access       (); });
		$('#admin-notifications-reload').button().click(function() { that.load_notify(); });
		$('#admin-cablenumbers-reload' ).button().click(function() { that.load_cablenumbers (); });
		$('#admin-jobnumbers-reload'   ).button().click(function() { that.load_jobnumbers   (); });

        var administrator2add = $('#admin-access').find('input[name="administrator2add"]');
        administrator2add.
            keyup(function(e) {
                var uid = $(this).val();
                if( uid == '' ) { return; }
                if( e.keyCode == 13 ) { that.new_administrator(uid); return; }});

        var projmanager2add = $('#admin-access').find('input[name="projmanager2add"]');
        projmanager2add.
            keyup(function(e) {
                var uid = $(this).val();
                if( uid == '' ) { return; }
                if( e.keyCode == 13 ) { that.new_projmanager(uid); return; }});

        var listener2add = $('#admin-notifications').find('input[name="listener2add"]');
        listener2add.
            keyup(function(e) {
                var uid = $(this).val();
                if( uid == '' ) { return; }
                if( e.keyCode == 13 ) { that.new_listener(uid); return; }});

        var submit_pending = $('#admin-notifications').find('button[name="submit_all"]').
            button().
            click(function() { that.submit_notify(); });

        var delete_pending = $('#admin-notifications').find('button[name="delete_all"]').
            button().
            click(function() { that.delete_notify(); });

        if(!this.can_manage_access()) {
            administrator2add.attr('disabled','disabled');
            administrator2add.attr('disabled','disabled');
                 listener2add.attr('disabled','disabled');
               submit_pending.attr('disabled','disabled');
               delete_pending.attr('disabled','disabled');
        }

        $('#admin-access').find('#tabs').tabs();
        $('#admin-notifications').find('#tabs').tabs();
        $('#admin-cablenumbers').find('#tabs').tabs();
        $('#admin-jobnumbers').find('#tabs').tabs();

		this.load_access();
        this.load_notify();
		this.load_cablenumbers();
		this.load_jobnumbers();
	};
    this.can_manage_access = function() {
        return global_current_user.is_administrator;
    };

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
	this.display_cablenumbers = function() {

        var rows = [];
        for( var cnidx in this.cablenumber ) {
            var c = this.cablenumber[cnidx];
            rows.push([
                this.can_manage_access() ?
                    Button_HTML('E', {
                        name:    'edit_'+cnidx,
                        classes: 'admin-cablenumbers-tools',
                        onclick: "admin.edit_cablenumber('"+cnidx+"')",
                        title:   'edit'
                    })+' '+
                    Button_HTML('save', {
                        name:    'save_'+cnidx,
                        classes: 'admin-cablenumbers-tools',
                        onclick: "admin.edit_cablenumber_save('"+cnidx+"')",
                        title:   'save changes to the database'
                    })+' '+
                    Button_HTML('cancel', {
                        name:    'edit_cancel_'+cnidx,
                        classes: 'admin-cablenumbers-tools',
                        onclick: "admin.edit_cablenumber_cancel('"+cnidx+"')",
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
            this.update_cablenumber_tools(cnidx,false);
        }
        $('.admin-cablenumbers-search').
            button().
            click(function () {
                var cnidx = this.name;
                var cn = that.cablenumber[cnidx];
                global_search_cables_by_prefix(cn.prefix);
            });
    };
	this.update_cablenumber = function(cnidx) {
        var c = this.cablenumber[cnidx];
        var elem = $('#admin-cablenumbers-cablenumbers');
        elem.find('div[name="prefix_'                 +cnidx+'"]').html(c.prefix);
        elem.find('div[name="range_'                  +cnidx+'"]').html(c.first+' - '+c.last);
        elem.find('div[name="num_available_'          +cnidx+'"]').html(c.num_available);
        elem.find('div[name="next_available_'         +cnidx+'"]').html(c.next_available);
        elem.find('div[name="recently_allocated_name_'+cnidx+'"]').html(this.cable_name2html(cnidx));
        elem.find('div[name="recent_allocation_time_' +cnidx+'"]').html(c.recent_allocation_time);
        elem.find('div[name="recent_allocation_uid_'  +cnidx+'"]').html(c.recent_allocation_uid);
        this.update_cablenumber_tools(cnidx,false);
    };
    this.update_cablenumber_tools = function(cnidx,editing) {
        if(!this.can_manage_access()) {
            this.disable_cablenumber_tools(cnidx);
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
    this.disable_cablenumber_tools = function(cnidx) {
        var elem = $('#admin-cablenumbers-cablenumbers');
		elem.find('button[name="edit_'       +cnidx+'"]').button('disable');
		elem.find('button[name="edit_save_'  +cnidx+'"]').button('disable');
		elem.find('button[name="edit_cancel_'+cnidx+'"]').button('disable');
    };
    this.edit_cablenumber = function(cnidx) {
        this.update_cablenumber_tools(cnidx,true);
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
    this.edit_cablenumber_save = function(cnidx) {
        this.disable_cablenumber_tools(cnidx);
        var c     = this.cablenumber[cnidx];
        var elem  = $('#admin-cablenumbers-cablenumbers');
        var first = parseInt(elem.find('div[name="range_'+cnidx+'"]').find('input[name="first"]' ).val());
        var last  = parseInt(elem.find('div[name="range_'+cnidx+'"]').find('input[name="last"]'  ).val());
        if( last <= first ) {
            report_error('invalid range: last number must be strictly larger than the first one', null);
            this.update_cablenumber_tools(cnidx,true);
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
                that.update_cablenumber_tools(cnidx,true);
                return;
            }
            that.cablenumber[cnidx] = data.cablenumber;
            that.update_cablenumber(cnidx);
        },
        'JSON').error(function () {
            report_error('failed to contact the Web service due to: '+jqXHR.statusText, null);
            that.update_cablenumber_tools(cnidx,true);
            return;
        });
    };
    this.edit_cablenumber_cancel = function(cnidx) {
       this.update_cablenumber(cnidx);
    };
	this.load_cablenumbers = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/cablenumber_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.cablenumber = data.cablenumber;
            that.display_cablenumbers();
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
            '    <button class="admin-jobnumbers-tools " name="edit"        onclick="admin.edit_jobnumber       ('+jnidx+')" title="edit"                                 ><b>E</b></button>'+
            '    <button class="admin-jobnumbers-tools " name="edit_save"   onclick="admin.edit_jobnumber_save  ('+jnidx+')" title="save changes to the database"         >save</button>'+
            '    <button class="admin-jobnumbers-tools " name="edit_cancel" onclick="admin.edit_jobnumber_cancel('+jnidx+')" title="cancel editing and ignore any changes">cancel</button>'+
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
    this.jobnumber_allocation2html = function(jnaidx) {
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
	this.display_jobnumbers = function() {
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
            this.update_jobnumber_tools(jnidx,false);
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
        for( var jnaidx in this.jobnumber_allocation ) html += this.jobnumber_allocation2html(jnaidx);
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
	this.update_jobnumber = function(jnidx) {
        var j = this.jobnumber[jnidx];
        $('#admin-jobnumbers-'+jnidx+' .prefix'                 ).html('&nbsp;'+j.prefix);
        $('#admin-jobnumbers-'+jnidx+' .range'                  ).html('&nbsp;'+j.first+' - '+j.last);
        $('#admin-jobnumbers-'+jnidx+' .num_in_use'             ).html('&nbsp;'+j.num_in_use);
        $('#admin-jobnumbers-'+jnidx+' .num_available'          ).html('&nbsp;'+j.num_available);
        $('#admin-jobnumbers-'+jnidx+' .next_available'         ).html('&nbsp;'+j.next_available);
        $('#admin-jobnumbers-'+jnidx+' .recently_allocated_name').html('&nbsp;'+this.job_name2html(j.recently_allocated_name));
        $('#admin-jobnumbers-'+jnidx+' .recent_allocation_time' ).html('&nbsp;'+j.recent_allocation_time);
        $('#admin-jobnumbers-'+jnidx+' .recent_allocation_uid'  ).html('&nbsp;'+j.recent_allocation_uid);
        this.update_jobnumber_tools(jnidx,false);
    };
    this.update_jobnumber_tools = function(jnidx,editing) {
        if(!this.can_manage_access()) {
            this.disable_jobnumber_tools(jnidx);
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
    this.disable_jobnumber_tools = function(jnidx) {
		$('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit"]'       ).button('disable');
		$('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit_save"]'  ).button('disable');
		$('#admin-jobnumbers-tools-'+jnidx).find('button[name="edit_cancel"]').button('disable');
    };
    this.edit_jobnumber = function(jnidx) {
        this.update_jobnumber_tools(jnidx,true);
        var c = this.jobnumber[jnidx];
        $('#admin-jobnumbers-'+jnidx+' .range').html(
            '<input type="text" size=2 style="text-align:right" name="first" value="'+c.first+'" /><input type="text" size=2 style="text-align:right" name="last" value="'+c.last+'" />'
        );
        if(c.prefix == '')
            $('#admin-jobnumbers-'+jnidx+' .prefix').html(
                '<input type="text" size=2 name="prefix" value="'+c.prefix+'" />'
            );
    };
    this.edit_jobnumber_save = function(jnidx) {
        this.disable_jobnumber_tools(jnidx);
        var c = this.jobnumber[jnidx];
        var first  = parseInt($('#admin-jobnumbers-'+jnidx).find('input[name="first"]' ).val());
        var last   = parseInt($('#admin-jobnumbers-'+jnidx).find('input[name="last"]'  ).val());
        if( last <= first ) {
            report_error('invalid range: last number must be strictly larger than the first one', null);
            this.update_jobnumber_tools(jnidx,true);
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
                that.update_jobnumber_tools(jnidx,true);
                return;
            }
            that.jobnumber[jnidx] = data.jobnumber;
            that.update_jobnumber(jnidx);
        },
        'JSON').error(function () {
            report_error('failed to contact the Web service due to: '+jqXHR.statusText, null);
            that.update_jobnumber_tools(jnidx,true);
            return;
        });
    };
    this.edit_jobnumber_cancel = function(jnidx) {
       this.update_jobnumber(jnidx);
    };
	this.load_jobnumbers = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/jobnumber_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.jobnumber = data.jobnumber;
            that.jobnumber_allocation = data.jobnumber_allocation;
            that.display_jobnumbers();
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
    this.new_administrator = function(uid) { this.new_user(uid,'administrator'); };
    this.new_projmanager   = function(uid) { this.new_user(uid,'projmanager'); };
 
    this.display_access_users = function(id,users) {
        var rows = [];
        for( var i in users ) {
            var a = users[i];
            rows.push([
                this.can_manage_access() ? 
                    Button_HTML('delete', {
                        name:     'delete',
                        onclick:  "admin.delete_user('"+a.uid+"')",
                        title:    'delete this user from the list'
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

    this.display_access = function() {
        this.display_access_users('admin-access-administrators', this.access.administrator);
        this.display_access_users('admin-access-projmanagers',   this.access.projmanager);
    };

    this.load_access = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/access_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.access = data.access;
            that.display_access();
        },
        'JSON').error(function () {
            report_error('failed to load access control info because of: '+jqXHR.statusText, null);
            return;
        });
    };

    this.new_user = function(uid,role) {
        var params = {uid:uid,role:role};
        var jqXHR = $.get('../neocaptar/access_new.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.access = data.access;
            that.display_access();
        },
        'JSON').error(function () {
            report_error('failed to add a new user to the access control list because of: '+jqXHR.statusText, null);
            return;
        });
    };

    this.delete_user = function(uid) {
        var params = {uid:uid};
        var jqXHR = $.get('../neocaptar/access_delete.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.access = data.access;
            that.display_access();
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
    this.can_manage_notify  = this.can_manage_access;
    this.notify             = null;
    this.notify_event_types = null;

    this.display_notify_projmanager = function() {
        
        if( !global_current_user.can_manage_projects ) {
            $('#admin-notifications-myself').html(
                '<div style="color:maroon;">'+
                '  We are sorry! Your account has not been found among registered project administrators.'+
                '</div>');
            return;
        }
        
        var rows = [];
        for( var i in this.notify_event_types.PROJMANAGER ) {
            var event  = this.notify_event_types.PROJMANAGER[i];
            var notify = this.notify.myself;
            rows.push([
                event.description,
                Checkbox_HTML({
                    classes: event.name,
                    checked: notify[event.name] }) ]);
        }
        var table = new Table( 'admin-notifications-myself', [
            { name: 'event',  sorted: false },
            { name: 'notify', sorted: false } ],
            rows );
        table.display();
    };

    this.display_notify_other = function() {

        var hdr = [ {name: '', sorted: false } ];
        for( var i in this.notify.others )
            hdr.push( {
                name: this.notify.others[i].uid,
                sorted: false });

        var rows = [];
        for( var i in this.notify_event_types.OTHER ) {
            var event = this.notify_event_types.OTHER[i];
            var row   = [ event.description ];
            for( var j in this.notify.others ) {
                var notify = this.notify.others[j];
                row.push( Checkbox_HTML({
                    classes: event.name,
                    name:    notify.uid,
                    checked: notify[event.name] }));
            }
            rows.push(row);
        }
        var row = [''];
        for( var i in this.notify.others )
            row.push( Button_HTML('x',{
                classes: 'delete_listener',
                name   : this.notify.others[i].uid,
                title  : 'remove this user from the list' }));
        rows.push(row);

        var table = new Table('admin-notifications-others', hdr, rows );
        table.display();

        var others = $('#admin-notifications-others');
        others.find('button.delete_listener').
            button().
            button(this.can_manage_notify? 'enable':'disable').
            click(function() {
                alert(this.name);
            });
    };

    this.display_notify = function() {
        this.display_notify_projmanager();
        this.display_notify_other();
    };

    this.load_notify = function() {
        var params = {};
        var jqXHR = $.get('../neocaptar/notify_get.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.notify_event_types = data.event_types;
            that.notify = data.notify;
            that.display_notify();
        },
        'JSON').error(function () {
            report_error('failed to load noditications list because of: '+jqXHR.statusText, null);
            return;
        });
    };

    this.new_listener = function(uid,role) {
        var params = {uid:uid};
        var jqXHR = $.get('../neocaptar/notify_new.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.notify = data.notify;
            that.display_notify();
        },
        'JSON').error(function () {
            report_error('failed to add a new user to the notifications list because of: '+jqXHR.statusText, null);
            return;
        });
    };

    this.delete_listener = function(uid) {
        var params = {uid:uid};
        var jqXHR = $.get('../neocaptar/notify_delete.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.notify = data.notify;
            that.display_notify();
        },
        'JSON').error(function () {
            report_error('failed to delete this user from the notifications list because of: '+jqXHR.statusText, null);
            return;
        });
    };

    this.submit_notify = function(idx) {
        alert('submit_notify: not implemented');
    };

    this.delete_notify = function(idx) {
        alert('delete_notify: not implemented');
    };
    return this;
}
var admin = new p_appl_admin();
