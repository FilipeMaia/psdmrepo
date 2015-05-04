define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Experiment_Group.css') ;

    /**
     * The application for managing the POSIX group of the experiment
     *
     * @returns {Experiment_Group}
     */
    function Experiment_Group (experiment, access_list) {

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

        this._update_interval_sec = 30 ;
        this._prev_update_sec = null ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
                var now_sec = Fwk.now().sec ;
                if (!this._prev_update_sec || (now_sec - this._prev_update_sec) > this._update_interval_sec) {
                    this._prev_update_sec = now_sec ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.experiment  = experiment ;
        this.access_list = access_list ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var this_html = html ||
'<div id="exp-group" >' +

  '<div class="info" id="updated" style="float:right;" ></div> ' +
  '<div style="clear:both;" ></div> ' +

  '<div style="float:left; margin-left:10px; margin-right:20px; margin-bottom:40px; padding-right:30px; border-right: 1px solid #c0c0c0;" > ' +
  '  <div style="height:55px;" >' +
  '    <div style="float:left; font-size: 28px;" ><b>'+this.experiment.posix_group+'</b></div> ' +
      '<div style="float:left; margin-left:10px; padding-top:4px;" > ' +
        '<button class="control-button" name="update" ><img src="../webfwk/img/Update.png" /></button> ' +
       '</div>' +
  '    <div style="clear:both;"></div>' +
  '  </div>' +
  '  <div class="info" id="group-status"></div>' +
  '  <div              id="members" style="margin-top:4px;"></div>' +
  '</div>' +
  '<div style="float:left; padding-left:10px;">' +
  '  <div style="height:50px; padding-top:5px;">' +
  '    <div style="float:left; font-weight:bold; padding-top:9px;">Search users:</div>' +
  '    <div style="float:left; margin-left:5px;">' +
  '      <input type="text" style="margin-top:4px; padding:2px; background-color:#ffeeee;" id="string2search" value="" size=16 title="enter the pattern to search then press RETURN" />' +
  '    </div>' +
  '    <div style="float:left; margin-left:10px; font-weight:bold; padding-top:9px;">by:</div>' +
  '    <div style="float:left; margin-left:5px; padding-top:3px;" id="scope">' +
  '      <input type="radio" id="uid"   name="scope" value="uid"                         /><label for="uid"   class="control-label" >UID</label>' +
  '      <input type="radio" id="gecos" name="scope" value="gecos"                       /><label for="gecos" class="control-label" >name</label>' +
  '      <input type="radio" id="both"  name="scope" value="uid_gecos" checked="checked" /><label for="both"  class="control-label" >both</label>' +
  '    </div>' +
  '    <div style="clear:both;"></div>' +
  '  </div>' +
  '  <div class="info" id="user-status"></div>' +
  '  <div              id="users" style="margin-top:4px;"></div>' +
  '</div>' +
  '<div style="clear:both;"></div>' +

'</div>' ;
                this.container.html(this_html) ;
                this._wa_elem = this.container.children('#exp-group') ;
            }
            return this._wa_elem ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().find('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._button_update = function () {
            if (!this._button_update_elem) {
                this._button_update_elem = this._wa().find('button[name="update"]').button() ;
            }
            return this._button_update_elem ;
        } ;

        this._scope = function () {
            if (!this._scope_elem) this._scope_elem = this._wa().find('#scope').buttonset() ;
            return this._scope_elem ;
        } ;
        this._string2search = function () {
            if (!this._string2search_elem) this._string2search_elem = this._wa().find('#string2search') ;
            return this._string2search_elem ;
        } ;
        this._group_status = function () {
            if (!this._group_status_elem) this._group_status_elem = this._wa().find('#group-status') ;
            return this._group_status_elem ;
        } ;
        this._user_status = function () {
            if (!this._user_status_elem) this._user_status_elem = this._wa().find('#user-status') ;
            return this._user_status_elem ;
        } ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            if (!this.access_list.experiment.manage_group) {
                this._wa(this.access_list.no_page_access_html) ;
                return ;
            }

            this._scope().change(function() { _that._do_search_users() ; }) ;

            this._string2search().keyup(function (e) {
                _that._string2search().css('background-color', _that._string2search().val() === '' ? '#ffeeee' : '') ;
                if (e.keyCode === 13) _that._do_search_users() ;
            }) ;
            this._button_update().click(function() { _that._load() ; }) ;

            this._load() ;
        } ;

        this._do_manage_user = function (action, uid) {

            Fwk.web_service_GET (

                '../regdb/ws/ManageGroupMembers.php' ,

                {   group: this.experiment.posix_group, simple: '', action: action, uid: uid
                } ,

                function (data) {
                    if (data.ResultSet.Status !== 'success') {
                        _that._group_status().html(data.ResultSet.Message) ;
                        return ;
                    }
                    _that._load() ;
                } ,

                function () {
                    _that._group_status().html('<span style="color:red;">Failed to '+action+' user '+uid+' from/to group '+this.experiment.posix_group+'</span>') ;
                }
            ) ;
        } ;

        this._do_search_users = function () {

            if (this._string2search().val() === '') {
                this._string2search().css('background-color', '#ffeeee') ;
                Fwk.report_error('Please, enter a string to search for!') ;
                return ;
            }
            this._user_status().html('<span style="color:maroon;">Searching...</span>') ;

            Fwk.web_service_GET (

                '../regdb/ws/RequestUserAccounts.php' ,

                {   simple:        '' ,
                    string2search: this._string2search().val() ,
                    scope:         this._scope().find('input:checked').val()
                } ,

                function (data) {
                    var users = data.ResultSet.Result ;
                    _that._user_status().html('<span style="color:maroon;">Found <b>'+users.length+'</b> users</span>') ;
                    var html =
'<table><tbody>' +
'  <tr><td class="table_hdr" ></td>' +
'      <td class="table_hdr" >UID</td>' +
'      <td class="table_hdr" >Name</td></tr>' ;
                    for (var i in users) {
                        html +=
'  <tr><td class="table_cell table_cell_left"  ><button class="control-button add" id="'+users[i].uid+'" title="add to the group">&lArr;</button></td>' +
'      <td class="table_cell table_cell_left"  >'+users[i].uid+'</td>' +
'      <td class="table_cell table_cell_right" >'+users[i].name+'</td></tr>' ;
                    }
                    html += '</tbody></table>' ;
                    _that._wa().find('#users').html(html) ;
                    _that._wa().find('button.add').button().click(function () { _that._do_manage_user('include', this.id) ; }) ;
                } ,

                function () {
                    _that._user_status().html('<span style="color:red;">Failed to get the information from the Web server</span>') ;
                }
            ) ;
        } ;

        this._load = function () {

            this._set_updated('Updating...') ;

            Fwk.web_service_GET (

                '../regdb/ws/ManageGroupMembers.php' ,

                {   group: this.experiment.posix_group, simple: ''
                } ,

                function (data) {

                    var users = data.ResultSet.Result ;
                    _that._set_updated('Updated: <b>'+data.updated+'</b>') ;
                    _that._group_status().html('<span style="color:maroon;">Has <b>'+users.length+'</b> members</span>') ;
                    var html =
'<table><tbody>' +
'  <tr><td class="table_hdr" >UID</td>' +
'      <td class="table_hdr" >Name</td>' +
'      <td class="table_hdr" >REMOVE</td></tr>' ;
                    for (var i in users) {
                        html +=
'  <tr><td class="table_cell table_cell_left"  >'+users[i].uid+'</td>' +
'      <td class="table_cell"                  >'+users[i].name+'</td>' +
'      <td class="table_cell table_cell_right" ><button class="control-button delete" id="'+users[i].uid+'" title="remove from the group" >x</button></td></tr>' ;
                    }
                    html +=
'</tbody></table>' ;
                    _that._wa().find('#members').html(html) ;
                    _that._wa().find('button.delete').button().click(function () { _that._do_manage_user('exclude', this.id) ; }) ;
                } ,

                function () {
                    _that._group_status().html('<span style="color:red;">Failed to get the information from the Web server</span>') ;
                }
            );
        } ;
    }
    Class.define_class (Experiment_Group, FwkApplication, {}, {}) ;

    return Experiment_Group ;
});
