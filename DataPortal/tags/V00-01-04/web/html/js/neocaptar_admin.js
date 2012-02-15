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

    this.select = function(context,when_done) {
		that.context   = context;
		this.when_done = when_done;
		this.init();
	};
	this.select_default = function() {
		if( this.context == '' ) this.context = 'cablenumbers';
		this.init();
	};
	this.if_ready2giveup = function(handler2call) {
		this.init();
        handler2call();
	};
    this.update = function() {
		this.init();
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
		$('#admin-cablenumbers-reload').button().click(function() { that.load_cablenumbers(); });
		$('#admin-jobnumbers-reload'  ).button().click(function() { that.load_jobnumbers  (); });
		$('#admin-access-reload'      ).button().click(function() { that.load_access      (); });
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
        if(!this.can_manage_access()) {
            administrator2add.attr('disabled','disabled');
            projmanager2add.attr('disabled','disabled');
        }
		this.load_cablenumbers();
		this.load_jobnumbers();
		this.load_access();
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
    this.cablenumber2html = function(cnidx) {
        var c = this.cablenumber[cnidx];
        var html =
'<tr id="admin-cablenumbers-'+cnidx+'" >'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " id="admin-cablenumbers-tools-'+cnidx+'">'+
'    <button class="admin-cablenumbers-tools " name="edit"        onclick="admin.edit_cablenumber       ('+cnidx+')" title="edit"                                 ><b>E</b></button>'+
'    <button class="admin-cablenumbers-tools " name="edit_save"   onclick="admin.edit_cablenumber_save  ('+cnidx+')" title="save changes to the database"         >save</button>'+
'    <button class="admin-cablenumbers-tools " name="edit_cancel" onclick="admin.edit_cablenumber_cancel('+cnidx+')" title="cancel editing and ignore any changes">cancel</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell "                                          >&nbsp;'+c.location                       +'</td>'+
'  <td nowrap="nowrap" class="table_cell                  prefix "                  >&nbsp;'+c.prefix                         +'</td>'+
'  <td nowrap="nowrap" class="table_cell                  range "                   >&nbsp;'+c.first+' - '+c.last             +'</td>'+
'  <td nowrap="nowrap" class="table_cell                  num_in_use "              >&nbsp;'+
'    <button class="admin-cablenumbers-search" name="'+cnidx+'" title="search all cables using this range">search</button>'+
'</td>'+
'  <td nowrap="nowrap" class="table_cell                  num_available "           >&nbsp;'+c.num_available                  +'</td>'+
'  <td nowrap="nowrap" class="table_cell                  next_available "          >&nbsp;'+c.next_available                 +'</td>'+
'  <td nowrap="nowrap" class="table_cell                  recently_allocated_name " >&nbsp;'+this.cable_name2html(cnidx)      +'</td>'+
'  <td nowrap="nowrap" class="table_cell                  recent_allocation_time "  >&nbsp;'+c.recent_allocation_time         +'</td>'+
'  <td nowrap="nowrap" class="table_cell                  recent_allocation_uid "   >&nbsp;'+c.recent_allocation_uid          +'</td>'+
'</tr>';
        return html;
    };
	this.display_cablenumbers = function() {
        var html =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " >TOOLS</td>'+
'    <td nowrap="nowrap" class="table_hdr " >location</td>'+
'    <td nowrap="nowrap" class="table_hdr " >prefix</td>'+
'    <td nowrap="nowrap" class="table_hdr " >range</td>'+
'    <td nowrap="nowrap" class="table_hdr " >in use</td>'+
'    <td nowrap="nowrap" class="table_hdr " ># available</td>'+
'    <td nowrap="nowrap" class="table_hdr " >next</td>'+
'    <td nowrap="nowrap" class="table_hdr " >previously allocated</td>'+
'    <td nowrap="nowrap" class="table_hdr " >last allocation</td>'+
'    <td nowrap="nowrap" class="table_hdr " >requested by</td>'+
'  </tr>';
        for( var cnidx in this.cablenumber ) html += this.cablenumber2html(cnidx);
        html +=
'</tbody></table>';
        $('#admin-cablenumbers-cablenumbers').html(html);
        for( var cnidx in this.cablenumber ) {
			$('#admin-cablenumbers-tools-'+cnidx+' button.admin-cablenumbers-tools').
                button().
                button(this.can_manage_access()?'enable':'disable');
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
        $('#admin-cablenumbers-'+cnidx+' .prefix'                 ).html('&nbsp;'+c.prefix);
        $('#admin-cablenumbers-'+cnidx+' .range'                  ).html('&nbsp;'+c.first+' - '+c.last);
        $('#admin-cablenumbers-'+cnidx+' .num_available'          ).html('&nbsp;'+c.num_available);
        $('#admin-cablenumbers-'+cnidx+' .next_available'         ).html('&nbsp;'+c.next_available);
        $('#admin-cablenumbers-'+cnidx+' .recently_allocated_name').html('&nbsp;'+this.cable_name2html(cnidx));
        $('#admin-cablenumbers-'+cnidx+' .recent_allocation_time' ).html('&nbsp;'+c.recent_allocation_time);
        $('#admin-cablenumbers-'+cnidx+' .recent_allocation_uid'  ).html('&nbsp;'+c.recent_allocation_uid);
        this.update_cablenumber_tools(cnidx,false);
    };
    this.update_cablenumber_tools = function(cnidx,editing) {
        if(!this.can_manage_access()) {
            this.disable_cablenumber_tools(cnidx);
            return;
        }
		if( editing ) {
			$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit"]'       ).button('disable');
			$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit_save"]'  ).button('enable' );
			$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit_cancel"]').button('enable' );
			return;
		}
		$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit"]'       ).button('enable' );
		$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit_save"]'  ).button('disable');
		$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit_cancel"]').button('disable');
    };
    this.disable_cablenumber_tools = function(cnidx) {
		$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit"]'       ).button('disable');
		$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit_save"]'  ).button('disable');
		$('#admin-cablenumbers-tools-'+cnidx).find('button[name="edit_cancel"]').button('disable');
    };
    this.edit_cablenumber = function(cnidx) {
        this.update_cablenumber_tools(cnidx,true);
        var c = this.cablenumber[cnidx];
        $('#admin-cablenumbers-'+cnidx+' .range').html(
            '<input type="text" size=2 style="text-align:right" name="first" value="'+c.first+'" /><input type="text" size=2 style="text-align:right" name="last" value="'+c.last+'" />'
        );
        if(c.prefix == '')
            $('#admin-cablenumbers-'+cnidx+' .prefix').html(
                '<input type="text" size=2 name="prefix" value="'+c.prefix+'" />'
            );
    };
    this.edit_cablenumber_save = function(cnidx) {
        this.disable_cablenumber_tools(cnidx);
        var c = this.cablenumber[cnidx];
        var first  = parseInt($('#admin-cablenumbers-'+cnidx).find('input[name="first"]' ).val());
        var last   = parseInt($('#admin-cablenumbers-'+cnidx).find('input[name="last"]'  ).val());
        if( last <= first ) {
            report_error('invalid range: last number must be strictly larger than the first one', null);
            this.update_cablenumber_tools(cnidx,true);
            return;
        }
        var params = {id:c.id,first:first,last:last};
        if(c.prefix == '') {
            var prefix = $('#admin-cablenumbers-'+cnidx).find('input[name="prefix"]').val();
            if( prefix != '' ) params.prefix = prefix;
        }
        var jqXHR = $.get('../portal/neocaptar_cablenumber_save.php',params,function(data) {
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
        var jqXHR = $.get('../portal/neocaptar_cablenumber_get.php',params,function(data) {
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
        var jqXHR = $.get('../portal/neocaptar_jobnumber_save.php',params,function(data) {
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
        var jqXHR = $.get('../portal/neocaptar_jobnumber_get.php',params,function(data) {
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




    this.access = null;
    this.new_administrator = function(uid) {
        this.new_user(uid,'administrator');
    };
    this.administrator2html = function(aidx) {
        var a = this.access.administrator[aidx];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left  "><button name="delete" onclick="admin.delete_user('+"'"+a.uid+"'"+')" title="delete this user from the list" ><b>delete</b></button></td>'+
'  <td nowrap="nowrap" class="table_cell                  ">'+a.uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell                  ">'+a.name+'</td>'+
'  <td nowrap="nowrap" class="table_cell                  ">'+a.added_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell                  ">'+a.last_active_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right "><button name="projects" onclick="global_search_projects_by_owner('+"'"+a.uid+"'"+')" title="search projects owned by this user" ><b>search</b></button></td>'+
'</tr>';
        return html;
    };
    this.new_projmanager = function(uid) {
        this.new_user(uid,'projmanager');
    };
    this.projmanager2html = function(aidx) {
        var a = this.access.projmanager[aidx];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left  "><button name="delete" onclick="admin.delete_user('+"'"+a.uid+"'"+')" title="delete this user from the list" ><b>delete</b></button></td>'+
'  <td nowrap="nowrap" class="table_cell                  ">'+a.uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell                  ">'+a.name+'</td>'+
'  <td nowrap="nowrap" class="table_cell                  ">'+a.added_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell                  ">'+a.last_active_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right "><button name="projects" onclick="global_search_projects_by_owner('+"'"+a.uid+"'"+')" title="search projects owned by this user" ><b>search</b></button></td>'+
'</tr>';
        return html;
    };
    this.display_access = function()
    {
        var html =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " >&nbsp</td>'+
'    <td nowrap="nowrap" class="table_hdr " >UID</td>'+
'    <td nowrap="nowrap" class="table_hdr " >user</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >last active</td>'+
'    <td nowrap="nowrap" class="table_hdr " >projects</td>'+
'  </tr>';
        for( var aidx in this.access.administrator ) html += this.administrator2html(aidx);
        html +=
'</tbody></table>';
        var admins = $('#admin-access-administrators');
        admins.html(html);
        admins.find('button[name="delete"]').button().button(this.can_manage_access()? 'enable':'disable');
        admins.find('button[name="projects"]').button();

        html =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " >&nbsp</td>'+
'    <td nowrap="nowrap" class="table_hdr " >UID</td>'+
'    <td nowrap="nowrap" class="table_hdr " >user</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >last active</td>'+
'    <td nowrap="nowrap" class="table_hdr " >projects</td>'+
'  </tr>';
        for( var aidx in this.access.projmanager ) html += this.projmanager2html(aidx);
        html +=
'</tbody></table>';
        var managers = $('#admin-access-projmanagers');
        managers.html(html);
        managers.find('button[name="delete"]').button().button(this.can_manage_access()? 'enable':'disable');
        managers.find('button[name="projects"]').button();
    };
    this.load_access = function() {
        var params = {};
        var jqXHR = $.get('../portal/neocaptar_access_get.php',params,function(data) {
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
        var jqXHR = $.get('../portal/neocaptar_access_new.php',params,function(data) {
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
        var jqXHR = $.get('../portal/neocaptar_access_delete.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.access = data.access;
            that.display_access();
        },
        'JSON').error(function () {
            report_error('failed to delete this user from the access control list because of: '+jqXHR.statusText, null);
            return;
        });
    };
    return this;
}
var admin = new p_appl_admin();
