define ([
    'webfwk/CSSLoader'
] ,

function (cssloader) {

    cssloader.load('../portal/css/Experiment_Group.css') ;

    /**
     * The application for managing the POSIX group of the experiment
     *
     * @returns {Experiment_Group}
     */
    function Experiment_Group (experiment, access_list) {

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
            this.init() ;
        } ;

        this.on_update = function () {
            if (this.active) {
                this.init() ;
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

        this.wa = null ;    // work area container

        this.is_initialized = false ;

        this.init = function () {

            var that = this ;

            if (this.is_initialized) return ;
            this.is_initialized = true ;

            this.container.html('<div id="exp-group"></div>') ;
            this.wa = this.container.find('div#exp-group') ;

            if (!this.access_list.experiment.manage_group) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<div style="float:left; margin-left:10px; margin-right:20px; margin-bottom:40px; padding-right:30px; border-right: 1px solid #c0c0c0;">' +
'  <div style="height:55px;">' +
'    <div style="float:left; font-size: 300%; font-family: Times, sans-serif;"><b>'+this.experiment.posix_group+'</b></div>' +
'    <div style="float:left; margin-left:10px; padding-top:4px;"><button class="control-button" id="refresh">Refresh</button></div>' +
'    <div style="clear:both;"></div>' +
'  </div>' +
'  <div id="group-status"></div>' +
'  <div id="members" style="margin-top:4px;"></div>' +
'</div>' +
'<div style="float:left; padding-left:10px;">' +
'  <div style="height:55px;">' +
'    <div style="float:left; font-weight:bold; padding-top:8px;">Search users:</div>' +
'    <div style="float:left; margin-left:5px;">' +
'      <input type="text" style="padding:2px; background-color:#ffeeee;" id="string2search" value="" size=16 title="enter the pattern to search then press RETURN" />' +
'    </div>' +
'    <div style="float:left; margin-left:10px; font-weight:bold; padding-top:6px;">by:</div>' +
'    <div style="float:left; margin-left:5px; padding-top:3px;" id="scope">' +
'      <input type="radio" id="uid"   name="scope" value="uid"                         /><label for="uid"   class="control-label" >UID</label>' +
'      <input type="radio" id="gecos" name="scope" value="gecos"                       /><label for="gecos" class="control-label" >name</label>' +
'      <input type="radio" id="both"  name="scope" value="uid_gecos" checked="checked" /><label for="both"  class="control-label" >both</label>' +
'    </div>' +
'    <div style="clear:both;"></div>' +
'  </div>' +
'  <div id="group-status"></div>' +
'  <div id="users" style="margin-top:4px;"></div>' +
'</div>' +
'<div style="clear:both;"></div>' ;
            this.wa.html(html) ;

            this.scope = this.wa.find('#scope').buttonset().change(function() { that.do_search_users() ; }) ;

            this.string2search = this.wa.find('#string2search') ;
            this.string2search.keyup(function (e) {
                that.string2search.css('background-color', that.string2search.val() === '' ? '#ffeeee' : '') ;
                if (e.keyCode === 13) that.do_search_users() ;
            }) ;
            this.wa.find('#refresh').button().click(function() { that.do_refresh_members() ; }) ;

            this.group_status = this.wa.find('#group-status') ;
            this.user_status  = this.wa.find('#user-status') ;

            this.do_refresh_members() ;
        } ;

        this.do_manage_user = function (action, uid) {

            var that = this ;

            Fwk.web_service_GET (

                '../regdb/ws/ManageGroupMembers.php' ,

                {   group: this.experiment.posix_group, simple: '', action: action, uid: uid
                } ,

                function (data) {
                    if (data.ResultSet.Status !== 'success') {
                        that.group_status.html(data.ResultSet.Message) ;
                        return ;
                    }
                    that.do_refresh_members() ;
                } ,

                function () {
                    that.group_status.html('<span style="color:red;">Failed to '+action+' user '+uid+' from/to group '+this.experiment.posix_group+'</span>') ;
                }
            ) ;
        } ;

        this.do_search_users = function () {

            var that = this ;

            if (this.string2search.val() === '') {
                this.string2search.css('background-color', '#ffeeee') ;
                Fwk.report_error('Please, enter a string to search for!') ;
                return ;
            }
            this.user_status.html('<span style="color:maroon;">Searching...</span>') ;

            Fwk.web_service_GET (

                '../regdb/ws/RequestUserAccounts.php' ,

                {   simple: '' ,
                    string2search: this.string2search.val() ,
                    scope: this.scope.find('input:checked').val()
                } ,

                function (data) {
                    var users = data.ResultSet.Result ;
                    that.user_status.html('<span style="color:maroon;">Found <b>'+users.length+'</b> users</span>') ;
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
                    that.wa.find('#users').html(html) ;
                    that.wa.find('button.add').button().click(function () { that.do_manage_user('include', this.id) ; }) ;
                } ,

                function () {
                    that.user_status.html('<span style="color:red;">Failed to get the information from the Web server</span>') ;
                }
            ) ;
        } ;

        this.do_refresh_members = function () {

            var that = this ;

            this.group_status.html('<span style="color:maroon;">Fetching...</span>') ;

            Fwk.web_service_GET (

                '../regdb/ws/ManageGroupMembers.php' ,

                {   group: this.experiment.posix_group, simple: ''
                } ,

                function (data) {
                    if (data.ResultSet.Status !== 'success') {
                        that.group_status.html(result.ResultSet.Message) ;
                        return ;
                    }
                    var users = data.ResultSet.Result ;
                    that.group_status.html('<span style="color:maroon;">Has <b>'+users.length+'</b> members</span>') ;
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
                    that.wa.find('#members').html(html) ;
                    that.wa.find('button.delete').button().click(function () { that.do_manage_user('exclude', this.id) ; }) ;
                } ,

                function () {
                    that.group_status.html('<span style="color:red;">Failed to get the information from the Web server</span>') ;
                }
            );
        } ;
    }
    define_class (Experiment_Group, FwkApplication, {}, {});

    return Experiment_Group ;
});
