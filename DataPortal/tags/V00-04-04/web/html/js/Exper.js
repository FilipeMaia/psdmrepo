/*
 * ===============================================
 *  Application: Experiment
 *  DOM namespace of classes and identifiers: exp-
 *  JavaScript global names begin with: exper
 * ===============================================
 */

function exper_create() {

	/* Add this anchor to access this object's variables from within
	 * for anonymous functions. Just using this.varname won't work
	 * due to a well known bug in JavaScript. */

	var that = this;

	/* ---------------------------------------
	 *  This application specific environment
	 * ---------------------------------------
	 *
	 * NOTE: These variables should be initialized externally.
	 */
	this.posix_group = null;

	/* The context for v-menu items
	 */
	var context2_default = {
		'summary' : '',
		'manage'  : 'group'
	};
	this.name = 'experiment';
	this.full_name = 'Experiment';
	this.context1 = 'summary';
	this.context2 = '';
	this.select_default = function() { this.select(this.context1, this.context2); };
	this.select = function(ctx1, ctx2) {
		this.init();
		this.context1 = ctx1;
		this.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};

	/* ------------------------------------
	 *  Initialize experiment summary page
	 * ------------------------------------
	 */
	this.summary_init = function() {
		$('#exp-group-toggler').click(function() {
			if( $('#exp-group-members').hasClass('exp-group-members-hidden')) {
				$('#exp-group-members').removeClass('exp-group-members-hidden').addClass('exp-group-members-visible');
				$('#exp-group-toggler').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			} else {
				$('#exp-group-members').removeClass('exp-group-members-visible').addClass('exp-group-members-hidden');
				$('#exp-group-toggler').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			}
		});
	};

	/* -----------------------------
	 *  Group management operations
	 * -----------------------------
	 */
	this.do_manage_user = function(action,uid) {
		var params = {group: this.posix_group, simple: '', action: action, uid: uid };
		var jqXHR = $.get('../regdb/ws/ManageGroupMembers.php',params,function(data) {
			var result = eval(data);
			if(result.ResultSet.Status != 'success') {
				$('#exp-m-g-members-stat').html(result.ResultSet.Message);
				return;
			}
			that.do_refresh_members();
		},
		'JSON').error(function() {
			$('#exp-m-g-members-stat').html('<span style="color:red;">Failed to '+action+' use '+uid+' from/to group '+this.posix_group+'</span>');
		});
	}
	this.do_search_users = function() {
		if($('#exp-m-g-string2search').val() == '') {
			$('#exp-m-g-string2search').css('background-color', '#ffeeee');
			alert('Please, enter a string to search for!');
			return;
		}
		var params = {
			simple: '',
			string2search: $('#exp-m-g-string2search').val(),
			scope: $('#exp-m-g-scope > input:checked').val()
		};
		$('#exp-m-g-users-stat').html('<span style="color:maroon;">Searching...</span>');
		var jqXHR = $.get('../regdb/ws/RequestUserAccounts.php',params,function(data) {
			var result = eval(data);
			var users = result.ResultSet.Result;
			$('#exp-m-g-users-stat').html('<span style="color:maroon;">Found <b>'+users.length+'</b> users</span>');
			var html =
				'<table><tbody>'+
				'<tr><td class="table_hdr"></td>'+
				'<td class="table_hdr">UID</td>'+
				'<td class="table_hdr">Name</td></tr>';
			for(var i in users) {
				html +=
					'<tr><td class="table_cell table_cell_left"><button class="exp-m-g-add" id="'+users[i].uid+'" title="add to the group">&lArr;</button></td>'+
					'<td class="table_cell table_cell_left">'+users[i].uid+'</td>'+
					'<td class="table_cell table_cell_right">'+users[i].name+'</td></tr>';
			}
			html += '</tbody></table>';
			$('#exp-m-g-users').html(html);
			$('.exp-m-g-add').button().click(function () { that.do_manage_user('include',this.id); });
		},
		'JSON').error(function() {
			$('#exp-m-g-users-stat').html('<span style="color:red;">Failed to get the information from the Web server</span>');
		});
	}
	this.do_refresh_members = function() {
		var params = {group: this.posix_group, simple: '' };
		$('#exp-m-g-members-stat').html('<span style="color:maroon;">Fetching...</span>');
		var jqXHR = $.get('../regdb/ws/ManageGroupMembers.php',params,function(data) {
			var result = eval(data);
			if(result.ResultSet.Status != 'success') {
				$('#exp-m-g-members-stat').html(result.ResultSet.Message);
				return;
			}
			var users = result.ResultSet.Result;
			$('#exp-m-g-members-stat').html('<span style="color:maroon;">Has <b>'+users.length+'</b> members</span>');
			var html =
				'<table><tbody>'+
				'<tr><td class="table_hdr">UID</td>'+
				'<td class="table_hdr">Name</td>'+
				'<td class="table_hdr"></td></tr>';
			for(var i in users) {
				html +=
					'<tr><td class="table_cell table_cell_left">'+users[i].uid+'</td>'+
					'<td class="table_cell">'+users[i].name+'</td>'+
					'<td class="table_cell table_cell_right"><button class="exp-m-g-delete" id="'+users[i].uid+'" title="remove from the group">x</button></td></tr>';
			}
			html += '</tbody></table>';
			$('#exp-m-g-members').html(html);
			$('.exp-m-g-delete').button().click(function () { that.do_manage_user('exclude',this.id); });
		},
		'JSON').error(function() {
			$('#exp-m-g-members-stat').html('<span style="color:red;">Failed to get the information from the Web server</span>');
		});
	}
	this.manage_group_init = function() {
		$('#exp-m-g-scope').buttonset().change(function() { that.do_search_users(); });
		$('#exp-m-g-string2search').keyup(function(e) {
			if($('#exp-m-g-string2search').val() == '')
				$('#exp-m-g-string2search').css('background-color', '#ffeeee');
			else
				$('#exp-m-g-string2search').css('background-color', 'inherit');
			if(e.keyCode == 13) that.do_search_users();
		});
		$('#exp-m-g-refresh').button().click(function() { that.do_refresh_members(); });
		this.do_refresh_members();
	};

	/* ----------------------------------
	 *  Application initialization point
	 * ----------------------------------
	 *
	 * RETURN: true if the real initialization took place. Return false
	 *         otherwise.
	 */
	this.is_initialized = false;
	this.init = function() {
		if(that.is_initialized) return false;
		this.is_initialized = true;
		this.summary_init();
		this.manage_group_init();
		return true;
	};
}

var exper = new exper_create();
