// linted with jslint.com

/*jslint node: true */
/*jslint white: true */
/*jslint vars: true */
/*jslint browser:true */
/*jslint indent: 2 */
/*global $:false */
/*global alert:false */
/*global confirm:false */

"use strict";

var areas;
var uses;
var ui;
var current_shift;
var beam_destination_masks;

function log(msg) {
  console.log(msg);
}

function debug(msg) {
  console.debug(msg);
}

function error(msg) {
  console.error(msg);
  alert(msg);
}

function printStackTrace() {
  try {
    throw new Error();
  } catch (e) {
    console.error(e.stack);
  }
}

function zeroPad(num, len) {
  var s = String(num);
  while (s.length < len) {
    s = "0" + s;
  }
  return s;
}

function trim(s) {
  return String(s).replace(/^\s\s*/, '').replace(/\s\s*$/, '');
}

function firstline(s) {
  s = trim(s);
  if (s.indexOf("\n") !== -1) {
    s = s.split("\n")[0];
  }
  return trim(s);
}

// Return yyyy-mm-dd from Date object
function dateToYMD(date) {
  return zeroPad(date.getFullYear(), 4) + "-" + zeroPad(date.getMonth() + 1, 2) + "-" + zeroPad(date.getDate(), 2);
}

// Return hh:mm from Date object
function dateToHM(date) {
  return zeroPad(date.getHours(), 2) + ":" + zeroPad(date.getMinutes(), 2);
}

function dateToString(date) {
  if (!date) {
    return "";
  }
  return dateToYMD(date) + " " + dateToHM(date);
}

// Convert unix_time (seconds since 1970 UTC) to Date() object.
function convert_unix_time_to_Date(unix_time) {
  return new Date(1000 * unix_time);
}

function convert_unix_time_to_YMD(unix_time) {
  return dateToYMD(convert_unix_time_to_Date(unix_time));
}

function convert_unix_time_to_HM(unix_time) {
  return dateToHM(convert_unix_time_to_Date(unix_time));
}

function secondsToString(seconds, zeroPadHours) {
  var hours = Math.floor(seconds / 3600);
  var minutes = Math.round((seconds % 3600) / 60);
  if (zeroPadHours) {
    hours = zeroPad(hours, 2);
  }
  return String(hours) + ":" + zeroPad(minutes, 2);
}

// Convert unix_time (seconds since 1970 UTC) to yyyy-mm-dd hh:mm.
function convert_unix_time_to_string(unix_time) {
  var date = convert_unix_time_to_Date(unix_time);
  if (date === null) {
    return "";
  }
  return dateToString(date);
}

// Convert yyyy-mm-dd hh:mm:ss to local unix_time, or NaN on error.
function parse_unix_date_part(date_part) {
  var date_parts = date_part.split("-");
  if (date_parts.length != 3) {
    log("parse_unix_date_part(" + date_part + "): expecting three sections separated by hyphens");
    return NaN;
  }
  var year = parseInt(date_parts[0], 0);
  var month = parseInt(date_parts[1], 0) - 1;
  var day = parseInt(date_parts[2], 0);
  var seconds = new Date(year, month, day).getTime() / 1000;
  log("parse_unix_date_part(" + date_part + ") -> " + seconds);
  return seconds;
}

// Convert time string to seconds, or NaN on error.
// Handles hh, hh:mm, hh:mm:ss, and h.hhh (e.g. 1.5 == 1:30).
// Also handles 3m or 3min or 3.5m or 3.5min (with or without spaces)
// Also handles 3h or 3.5h
// Suggestions from Bill:
//
// 1:0   --> 1:00 
// :30   -->  0:30 
// 1h30m --> 1:30
// 5m    --> 0:05 
// 1hour --> 1:00 
// 1min  --> 0:01 
// 90min --> 1:30 
// 1hour45min --> 1:45 
// 1hour 45m --> 1:45  (with space and hour and m ) (If this is hard to do don't bother) 
// 1h90m --> 2:30  (Okay no one should ever do this) 
//
function parse_unix_time_part(time_part) {
  var hours;
  var pm = false;
  time_part = trim(time_part);

  // Check for any AM/PM.
  if (time_part.match(/^[0-9:. ]*[aA][mM]$/)) {
    // There is an AM at the end. Remove it (it has no effect).
    time_part = trim(time_part.slice(0, time_part.length - 2));
  } else if (time_part.match(/^[0-9:. ]*[pP][mM]$/)) {
    // There is a PM at the end. Remove it and remember so we can add 12 hours.
    time_part = trim(time_part.slice(0, time_part.length - 2));
    pm = true;
  }

  // Parse hh or hh:mm or h.hhh
  // Note: need to explicitly provide radix 10, otherwise "09" (from "09:00") gets parsed as 0!
  var parts;
  if (time_part.match(/^\d+$/) ||
      time_part.match(/^\d+\s*h$/)) {
    // hh
    // hh 'h'
    hours = parseInt(time_part, 10); // this will get rid of leading zeros, etc
  } else if (time_part.match(/^\d+:\d+$/)) {
    // hh:mm
    parts = time_part.split(":");
    hours = parseInt(parts[0], 10);
    hours += parseInt(parts[1], 10) / 60.0;
  } else if (time_part.match(/^\:\d+$/)) {
    // :mm
    parts = time_part.split(":");
    hours = parseInt(parts[1], 10) / 60.0;
  } else if (time_part.match(/^\d+:\d+:\d+$/)) {
    // hh:mm:ss
    parts = time_part.split(":");
    hours = parseInt(parts[0], 10);
    hours += parseInt(parts[1], 10) / 60.0;
    hours += parseInt(parts[2], 10) / 3600.0;
  } else if (time_part.match(/^\d+\.\d+$/) ||
             time_part.match(/^\d+\.\d+\s*h$/)) {
    // h.hhh
    // h.hhh 'h'
    hours = parseFloat(time_part);
  } else if (time_part.match(/^\d+\s*[mM]$/) ||
             time_part.match(/^\d+\s*[mM][iI][nN]$/)) {
    // m 'm' or m 'min'
    hours = parseInt(time_part, 10) / 60;
  } else if (time_part.match(/^\d+\.\d+\s*[mM]$/) ||
             time_part.match(/^\d+\.\d+\s*[mM][iI][nN]$/)) {
    // m.mmm 'm' or m.mmm 'min'
    hours = parseInt(time_part, 10) / 60;
  } else {
    // invalid format, so return NaN
    return NaN;
  }
  if (pm) {
    hours += 12;
  }
  return 3600 * hours;
}

function now_seconds() {
  return new Date().getTime() / 1000;
}

function str(o) {
  if (typeof(o) === "object") {
    try {
      return JSON.stringify(o);
    } catch (e) {
      return "[" + String(o) + " : " + e.message + "]";
    }
  } else {
    return String(o);
  }
}

function appendToRow(row, node) {
  var cell = document.createElement("td");
  cell.appendChild(node);
  row.appendChild(cell);
}

function removeNode(node) {
  node.parentNode.removeChild(node);
}

function createCommentBox() {
  var commentBox = document.createElement("textarea");
  commentBox.rows = 3;
  commentBox.cols = 40;
  commentBox.value = "";
  return commentBox;
}

function createTextBox() {
  var timeBox = document.createElement("input");
  timeBox.type = "text";
  return timeBox;
}

// Call a web service synchronously, returning result.
// If there is an error, put it in result.ws_error.
function ws_call(php_file, get_or_post, args) {
  var result = $.ajax({type: get_or_post,
                          url: "../shiftmgr/ws/" + php_file,
                          data: args,
                          dataType: 'json',
                          async: false
                          });
  if (result.statusText !== "OK") {
    result.ws_error = firstline(result.statusText);
    return result;
  }
  result = $.parseJSON(result.responseText);
  if (result.status !== "success") {
    result.ws_error = firstline(result.message);
    return result;
  }
  result.ws_error = null;
  return result;
}

// Call a web service asynchronously, passing result to callback function.
// If there is an error, put it in result.ws_error.
function ws_call_async(php_file, get_or_post, args, callback) {
  function result_success(result) {
    if (result.statusText !== "OK") {
      result.ws_error = firstline(result.statusText);
      callback(result);
      return;
    }
    if (result.status !== "success") {
      result.ws_error = firstline(result.message);
      callback(result);
      return;
    }
    result.ws_error = null;
    callback(result);
  }
  function result_error() {
    error(php_file + " failed");
  }
  var result = $.ajax({type: get_or_post,
                          url: "../shiftmgr/ws/" + php_file,
                          data: args,
                          success: result_success,
                          dataType: 'json',
                          async: true
                          });
}

// Get list of all hutches that the current user is permitted to manage
function ws_get_permitted_hutches() {
  return ws_call("get_permitted_hutches.php", "GET");
}

// Get list of uses
function ws_get_uses() {
  return ws_call("get_uses.php", "GET");
}

// Get list of areas
function ws_get_areas() {
  return ws_call("get_areas.php", "GET");
}

// Get list of (shallow) shift objects for hutch and range of start times.
function ws_get_shifts(hutch, earliest_start_time, latest_start_time) {
  return ws_call("get_shifts.php", "GET", {hutch:hutch, earliest_start_time:earliest_start_time, latest_start_time:latest_start_time});
}

// Fetch a fully populated shift (with id, area_evaluations, uses, etc.) for this id.
function ws_get_shift(id) {
  return ws_call("get_shift.php", "GET", {id:id});
}

// Get values for a pv for all times from start to stop.
function ws_get_pv(pv, start, stop) {
  return ws_call("get_pv.php", "GET", {pv:pv, start:start, stop:stop});
}

function ws_get_beam_destination_masks() {
  return ws_call("get_beam_destination_masks.php", "GET", {});
}

// Fetch just the last modified time for the shift with this id.
function ws_get_shift_last_modified_time(id) {
  return ws_call("get_shift_last_modified_time.php", "GET", {id:id});
}

// Fetch just the last modified time for the shift with this id.
function ws_get_shift_last_modified_time_async(id, callback) {
  return ws_call_async("get_shift_last_modified_time.php", "GET", {id:id}, callback);
}

// Creates a shift in the database.
// Returns a fully populated shift (with id, area_evaluations, uses, etc.)
function ws_create_shift(hutch, start_time, end_time) {
  return ws_call("create_shift.php", "POST", {hutch:hutch, start_time:start_time, end_time:end_time});
}

// Updates a shift record in the database.
// Returns updated last modified time.
function ws_update_shift(shift) {
  return ws_call("update_shift.php", "POST", shift);
}

// Updates an area evaluation record in the database.
// Returns shift last_modified_time.
function ws_update_area_evaluation(area) {
  return ws_call("update_area_evaluation.php", "POST", current_shift.area_evaluation[area]);
}

function ws_update_time_use(use) {
  return ws_call("update_time_use.php", "POST", current_shift.time_use[use]); // id, use_name, use_time, comment
}

function ws_get_current_user() {
  return ws_call("get_current_user.php", "GET", {});
}

function addGeneralShiftInformationRows() {
  var now = new Date();
  var body = document.getElementById("shift_report_information_table");

  // username
  var result = ws_get_current_user();
  if (result.ws_error) {
    error("could not determine current user: " + result.ws_error);
    return false;
  }
  ui = {};
  ui.other_notes = document.getElementById("other_notes");
  ui.shift = {};
  ui.shift.username_row = document.createElement("tr");
  ui.shift.username_label = document.createElement("label");
  ui.shift.username_label.innerHTML = "User:";
  appendToRow(ui.shift.username_row, ui.shift.username_label);
  ui.shift.username = document.createElement("label");
  ui.shift.username.innerHTML = result.current_user;
  appendToRow(ui.shift.username_row, ui.shift.username);
  body.appendChild(ui.shift.username_row);

  // hutch
  ui.shift.hutch_row = document.createElement("tr");
  ui.shift.hutch_label = document.createElement("label");
  ui.shift.hutch_label.innerHTML = "<i><b>Loading hutches... please wait...</b></i>";
  appendToRow(ui.shift.hutch_row, ui.shift.hutch_label);
  body.appendChild(ui.shift.hutch_row);

  ui.shift.hutch_select = document.createElement("select");
  result = ws_get_permitted_hutches();
  if (result.ws_error) {
    error("could not fetch list of hutches: " + result.ws_error);
    return false;
  }
  var hutches = result.hutches;
  if (hutches.length === 0) {
    error("current user is not a manager for any hutch.");
    return;
  }
  for (var i = 0; i < hutches.length; i++) {
    var hutch = hutches[i];
    var option = document.createElement("option");
    option.value = hutch;
    option.innerHTML = hutch;
    ui.shift.hutch_select.appendChild(option);
  }

  ui.shift.hutch_label.innerHTML = "Hutch:";
  appendToRow(ui.shift.hutch_row, ui.shift.hutch_select);

  var shift_start_hover = "The Shift Start Time begins when the beam is handed over from the experiment previous -- nominally 9:00 and 21:00.";
  var shift_end_hover = "The Shift End Time should default to 12 hours after the Start Shift time, but it is editable for special circumstances.";

  // This is a function that sends updated shift data to the server
  // if either the start time or date changed.
  function check_for_start_time_changed() {
    var start_date_part = NaN;
    if (ui.shift.start_dateBox.value !== "") {
      start_date_part = parse_unix_date_part(ui.shift.start_dateBox.value);
      if (isNaN(start_date_part)) {
        alert('"' + date_text + '" is not a valid shift start date.');
        ui.shift.start_dateBox.value = "";
      } else {
        ui.shift.start_dateBox.value = convert_unix_time_to_YMD(start_date_part);
      }
    }

    var start_time_part = NaN;
    if (ui.shift.start_timeBox.value !== "") {
      start_time_part = parse_unix_time_part(ui.shift.start_timeBox.value);
      if (isNaN(start_time_part)) {
        alert('"' + ui.shift.start_timeBox.value + '" is not a valid shift start time.');
        ui.shift.start_timeBox.value = "";
      } else {
        ui.shift.start_timeBox.value = secondsToString(start_time_part, true);
      }
    }

    if (isNaN(start_date_part) || isNaN(start_time_part)) {
      return; // don't do anything else if we don't have a valid start date and time
    }

    // Update the end date/time.
    var current_shift_start_time = start_date_part + start_time_part;
    var current_shift_end_time = current_shift_start_time + 12 * 3600;
    ui.shift.end_dateBox.value = convert_unix_time_to_YMD(current_shift_end_time);
    ui.shift.end_timeBox.value = convert_unix_time_to_HM(current_shift_end_time);

    if (! current_shift) {
      return;
    }
    if (current_shift_start_time === current_shift.start_time &&
        current_shift_end_time === current_shift.end_time) {
      return;
    }

    // We have a current shift and something has changed, so update it.
    current_shift.start_time = current_shift_start_time;
    current_shift.end_time = current_shift_end_time;
    result = ws_update_shift(current_shift);
    if (result.ws_error) {
      error("failed to update shift times: " + result.ws_error);
    } else {
      current_shift.last_modified_time = result.last_modified_time;
    }

    // And since the shift length may have changed, we need to update the time use percentages.
    fixTimeUsePercentages();
    updateUI_CalculatedValues();
  }

  // This is a function that sends updated shift data to the server
  // if either the start time or date changed.
  function check_for_end_time_changed() {
    var end_date_part = NaN;
    if (ui.shift.end_dateBox.value !== "") {
      end_date_part = parse_unix_date_part(ui.shift.end_dateBox.value);
      if (isNaN(end_date_part)) {
        alert('"' + date_text + '" is not a valid shift end date.');
        ui.shift.end_dateBox.value = "";
      } else {
        ui.shift.end_dateBox.value = convert_unix_time_to_YMD(end_date_part);
      }
    }

    var end_time_part = NaN;
    if (ui.shift.end_timeBox.value !== "") {
      end_time_part = parse_unix_time_part(ui.shift.end_timeBox.value);
      if (isNaN(end_time_part)) {
        alert('"' + ui.shift.end_timeBox.value + '" is not a valid shift end time.');
        ui.shift.end_timeBox.value = "";
      } else {
        ui.shift.end_timeBox.value = secondsToString(end_time_part, true);
      }
    }

    if (isNaN(end_date_part) || isNaN(end_time_part)) {
      return; // don't do anything else if we don't have a valid end date and time
    }

    if (! current_shift) {
      return;
    }
    var current_shift_end_time = end_date_part + end_time_part;
    if (current_shift_end_time === current_shift.end_time) {
      return;
    }

    // We have a current shift and the end time has changed, so update it.
    current_shift.end_time = current_shift_end_time;
    result = ws_update_shift(current_shift);
    if (result.ws_error) {
      error("failed to update shift times: " + result.ws_error);
    } else {
      current_shift.last_modified_time = result.last_modified_time;
    }

    // And since the shift length may have changed, we need to update the time use percentages.
    fixTimeUsePercentages();
    updateUI_CalculatedValues();
  }

  // shift start date
  ui.shift.start_date_row = document.createElement("tr");
  ui.shift.start_date_label = document.createElement("label");
  ui.shift.start_date_label.innerHTML = "Shift Start Date:";
  ui.shift.start_date_label.title = shift_start_hover;
  appendToRow(ui.shift.start_date_row, ui.shift.start_date_label);
  ui.shift.start_dateBox = createTextBox();
  ui.shift.start_dateBox.value = dateToYMD(now);
  appendToRow(ui.shift.start_date_row, ui.shift.start_dateBox);
  body.appendChild(ui.shift.start_date_row);
  ui.shift.start_dateBox.onchange = check_for_start_time_changed;

  // shift start time
  ui.shift.start_time_row = document.createElement("tr");
  ui.shift.start_time_label = document.createElement("label");
  ui.shift.start_time_label.innerHTML = "Shift Start Time:";
  ui.shift.start_time_label.title = shift_start_hover;
  appendToRow(ui.shift.start_time_row, ui.shift.start_time_label);
  ui.shift.start_timeBox = createTextBox();
  var day_shift = (now.getHours() < 15);
  var start_time_default_text = day_shift ? "09:00" : "21:00";
  ui.shift.start_timeBox.value = start_time_default_text;
  ui.shift.start_timeBox.onchange = check_for_start_time_changed;
  appendToRow(ui.shift.start_time_row, ui.shift.start_timeBox);
  body.appendChild(ui.shift.start_time_row);

  // shift end date
  ui.shift.end_date_row = document.createElement("tr");
  ui.shift.end_date_label = document.createElement("label");
  ui.shift.end_date_label.innerHTML = "Shift End Date:";
  ui.shift.end_date_label.title = shift_end_hover;
  appendToRow(ui.shift.end_date_row, ui.shift.end_date_label);
  ui.shift.end_dateBox = createTextBox();
  ui.shift.end_dateBox.value = dateToYMD(day_shift ? now : now + 1);
  ui.shift.end_dateBox.onchange = check_for_end_time_changed;
  appendToRow(ui.shift.end_date_row, ui.shift.end_dateBox);
  body.appendChild(ui.shift.end_date_row);
  /*
  ui.shift.end_dateBox.onchange = function() {
    var time_text = ui.shift.end_dateBox.value;
    var end_time_part = parse_unix_time_part(time_text);
    if (isNaN(end_time_part)) {
      alert('"' + time_text + '" is not a valid shift end time.');
      ui.shift.end_timeBox.value = "";
    } else {
      // put in standard format
      ui.shift.end_timeBox.value = secondsToString(end_time_part, true);
      var end_hour = Math.round(end_time_part / 3600) + 12;
      if (end_hour >= 24) {
        end_hour -= 24;
      }
      ui.shift.end_timeBox.value = secondsToString(3600 * end_hour, true);
    }
    check_for_start_or_end_time_changed();
  };
  */

  // shift end time
  ui.shift.end_time_row = document.createElement("tr");
  ui.shift.end_time_label = document.createElement("label");
  ui.shift.end_time_label.innerHTML = "Shift End Time:";
  ui.shift.end_time_label.title = shift_end_hover;
  appendToRow(ui.shift.end_time_row, ui.shift.end_time_label);
  ui.shift.end_timeBox = createTextBox();
  var end_time_default_text = day_shift ? "21:00" : "09:00";
  ui.shift.end_timeBox.value = end_time_default_text;
  ui.shift.end_timeBox.onchange = check_for_end_time_changed;
  /*
  ui.shift.end_timeBox.onchange = function() {
    var time_text = trim(ui.shift.end_timeBox.value);
    if (time_text != "") {
      var end_time_part = parse_unix_time_part(time_text);
      if (isNaN(end_time_part)) {
        alert('"' + time_text + '" is not a valid shift end time.');
        ui.shift.end_timeBox.value = "";
      } else {
        // put in standard format
        ui.shift.end_timeBox.value = secondsToString(end_time_part, true);
      }
    }
    check_for_start_or_end_time_changed();
  };
  */
  appendToRow(ui.shift.end_time_row, ui.shift.end_timeBox);
  body.appendChild(ui.shift.end_time_row);

  // shift last update
  ui.shift.last_update_row = document.createElement("tr");
  ui.shift.last_update_label = document.createElement("label");
  ui.shift.last_update_label.innerHTML = "Last Update:";
  appendToRow(ui.shift.last_update_row, ui.shift.last_update_label);
  ui.shift.last_update = document.createElement("label");
  ui.shift.last_update.innerHTML = "";
  appendToRow(ui.shift.last_update_row, ui.shift.last_update);
  body.appendChild(ui.shift.last_update_row);

  // create and open shift buttons
  ui.shift.create_open_button_row = document.createElement("tr");
  ui.shift.create_button = make_button("Create Report", true);
  ui.shift.open_button = make_button("Open Report", false);
  body.appendChild(ui.shift.create_open_button_row);

  // helper function for making create/open shift buttons
  function make_button(text, is_create) {
    var button = document.createElement("input");
    button.type = "submit";
    button.value = text;
    appendToRow(ui.shift.create_open_button_row, button);
    button.onclick = function() {
      createOrOpenShift(is_create);
    };
    return button;
  }
}

function checkForServerUpdates_Process(result) {
  var i, area, use;

  // Do a quick check to see if the server has updates for us.
  var server_has_updates = false;
  if (result.ws_error == "undefined") { // XXX
    result.ws_error = null;
  }
  if (result.ws_error) {
    error("failed checking to see if server has updates: " + result.ws_error);
  } else {
    var delta = (result.last_modified_time - current_shift.last_modified_time);
    if (delta > 0) {
      server_has_updates = true;
      debug("server has updates from " + delta + " seconds ago.");
    }
  }

  // Now push our own updates.
  // Here we only look at changed text fields.
  // Other inputs push changes via their onchange() methods.
  // But we want to push text field changes in progress
  // in case someone is writing a very long comment.
  //
  // Note that these will change current_shift.last_modified_time.
  // That's why we do the test above and remember the result.

  // Look for change in other_notes
  if (current_shift.other_notes !== ui.other_notes.value) {
    current_shift.other_notes = ui.other_notes.value;
    result = ws_update_shift(current_shift);
    if (result.ws_error) {
      error("failed updating other notes: " + result.ws_error);
    } else {
      current_shift.last_modified_time = result.last_modified_time;
    }
  }

  // Look for changes in area comments
  for (i = 0; i < areas.length; i++) {
    var area = areas[i];
    var ui_area_evaluation = ui.area_evaluation[area];
    var area_evaluation = current_shift.area_evaluation[area];
    var ok = (String(area_evaluation.ok) === "1");
    if (! ok) {
      var comment = ui_area_evaluation.commentBox.value;
      if (area_evaluation.comment != comment) {
        area_evaluation.comment = comment;
        var result = ws_update_area_evaluation(area);
        if (result.ws_error) {
          error("Could not update area evaluation comment: " + result.ws_error);
          continue;
        }
        current_shift.last_modified_time = result.last_modified_time;
      }
    }
  }

  // Look for changes in use comments
  for (i = 0; i < uses.length; i++) {
    var use = uses[i];
    var ui_time_use = ui.time_use[use];
    var time_use = current_shift.time_use[use];
    var ok = (String(time_use.ok) === "1");
    if (! ok) {
      var comment = ui_time_use.commentBox.value;
      if (time_use.comment != comment) {
        time_use.comment = comment;
        var result = ws_update_time_use(use);
        if (result.ws_error) {
          error("Could not update time use comment: " + result.ws_error);
          continue;
        }
        current_shift.last_modified_time = result.last_modified_time;
      }
    }
  }

  // Now, if the server had updates for us, pull them over.
  if (server_has_updates) {
    result = ws_get_shift(current_shift.id);
    if (result.ws_error) {
      error("unable to fetch updates from server: " + result.ws_error);
      return;
    }
    current_shift = result.shift;
    ui.other_notes.value = current_shift.other_notes;

    // XXX update whether end_time should be frozen, etc
    // XXX this should be elsewhere

    for (i = 0; i < areas.length; i++) {
      updateUI_AreaEvaluation(areas[i]);
    }

    for (i = 0; i < uses.length; i++) {
      updateUI_TimeUse(uses[i]);
    }
  }

  // Finally, update the UI with the last modified time.
  ui.shift.last_update.innerHTML = convert_unix_time_to_string(current_shift.last_modified_time);
}

// This is the onclick method for the create and open buttons.
function createOrOpenShift(create) {
  var end_time;
  var msg;
  var i;
  var result;
  var delta_hours;
  var closest_shift;
  var closest_shift_delta_hours;
  var closest_shift_delta_days;

  // Get hutch
  var hutch = ui.shift.hutch_select.value;

  // Get start_time
  var start_date_part = parse_unix_date_part(ui.shift.start_dateBox.value);
  if (isNaN(start_date_part)) {
    alert("The start date is not valid.");
    return;
  }
  var start_time_part = parse_unix_time_part(ui.shift.start_timeBox.value);
  if (isNaN(start_time_part)) {
    alert("The start time is not valid.");
    return;
  }
  var start_time = start_date_part + start_time_part;
  log("start time is " + convert_unix_time_to_string(start_time));

  // Get end_time, if any
  var end_time = 0;
  var end_time_text = trim(ui.shift.end_timeBox.value);
  if (end_time_text !== "") {
    var end_time_part = parse_unix_time_part(end_time_text);
    if (isNaN(end_time_part)) {
      alert("The end time is not valid.");
      return;
    }
    end_time = start_date_part + end_time_part;
    if (end_time <= start_time) {
      end_time += 24 * 3600; // shift spanned midnight, so add a day.
    }
    log("end time is " + convert_unix_time_to_string(end_time));
  }

  // Add bounds (smallest and largest start_time).
  // This allows us to fetch just a few interesting shifts from
  // the server/database instead of every single shift!
  var earliest_start_time = start_time - 24 * 3600; // one days earlier
  var latest_start_time   = start_time + 24 * 3600; // one day later
  result = ws_get_shifts(hutch, earliest_start_time, latest_start_time);
  if (result.ws_error) {
    error("could not fetch shifts for " + hutch + ": " + result.ws_error);
    return;
  }
  closest_shift = null;
  for (i = 0; i < result.shifts.length; i++) {
    var test_shift = result.shifts[i];
    delta_hours = Math.abs(start_time - test_shift.start_time) / 3600;
    if (closest_shift === null || delta_hours < closest_shift_delta_hours) {
      closest_shift = test_shift;
      closest_shift_delta_hours = delta_hours;
    }
  }

  // If the user asks to create a shift report and one exists with
  // approximately the same start time and date, ask them if they
  // want to open that shift report instead.
  //
  // Similarly, if the user asks to open a shift report, but none
  // exists close to the specified time, ask if they want to create
  // a new shift report at the specified time.
  if (create) {
    if (closest_shift !== null) {
      msg = "There already is a shift that starts at ";
      msg += convert_unix_time_to_string(closest_shift.start_time) + ".\n";
      msg += "Do you want to open this shift instead of creating a new one?\n";
      if (closest_shift_delta_hours < 1) {
        // Within an hour is TOO close to an existing shift.
        // So the only options are to open the existing shift, or cancel.
        msg += "Click OK to open this shift, or Cancel to enter a different date and time.";
        if (! confirm(msg)) {
          return;
        }
        create = false;
      } else if (closest_shift_delta_hours <= 6) {
        // If an existing shift is within six hours,
        // give the option of opening that shift, or creating a new one.
        msg += "Click OK to open this shift, or Cancel to create a new one.";
        if (confirm(msg)) {
          create = false;
        }
      }
    }
  } else { // open
    if (closest_shift === null) {
      msg = "No shift found. Do you want to create a shift for these times?";
      if (confirm(msg)) {
        create = true;
      } else {
        return;
      }
    } else {
      closest_shift_delta_days = closest_shift_delta_hours / 24;
      if (closest_shift_delta_hours >= 6) {
        if (closest_shift_delta_days > 2) {
          msg = "No shift found within " + Math.floor(closest_shift_delta_days) + " days of start time.\n";
        } else if (closest_shift_delta_days > 1) {
          msg = "No shift found within a day of start time.\n";
        } else {
          msg = "No shift found within " + Math.floor(closest_shift_delta_hours) + " hours of start time.\n";
        }
        msg += "The closest shift starts at " + convert_unix_time_to_string(closest_shift.start_time) + ". Open?\n";
        if (! confirm(msg)) {
          return;
        }
      } else if (closest_shift_delta_hours > 0) {
        msg = "The closest shift found started at " + convert_unix_time_to_string(closest_shift.start_time) + ".\n" +
          "Do you want to open this shift?";
        if (! confirm(msg)) {
          return;
        }
      }
    }
  }

  // Now, create or open the shift report.
  if (create) {
    // Create a new shift object.
    result = ws_create_shift(hutch, start_time, end_time);
    if (result.ws_error) {
      error("could not create shift: " + result.ws_error);
      return false;
    }
    current_shift = result.shift;
  } else {
    // Set current shift to the shift to be opened.
    result = ws_get_shift(closest_shift.id);
    if (result.ws_error) {
      error("could not open shift: " + result.ws_error);
      return false;
    }
    current_shift = result.shift;

    // Update ui.
    // XXX should use existing method for this.
    if (current_shift.end_time) {
      ui.shift.end_timeBox.value = dateToHM(convert_unix_time_to_Date(current_shift.end_time));
    } else {
      ui.shift.end_timeBox.value = "";
    }
    ui.other_notes.value = current_shift.other_notes;
  }

  setInterval(checkForServerUpdatesTimer, 5 * 1000); // 5 seconds

  // And freeze various things 
  removeNode(ui.shift.create_button);
  removeNode(ui.shift.open_button);

  ui.shift.hutch_row.innerHTML = '';
  ui.shift.hutch_label = document.createElement("label");
  ui.shift.hutch_label.innerHTML = "Hutch:";
  appendToRow(ui.shift.hutch_row, ui.shift.hutch_label);
  ui.shift.hutch = document.createElement("label");
  ui.shift.hutch.innerHTML = hutch;
  appendToRow(ui.shift.hutch_row, ui.shift.hutch);

  ui.shift.start_date_row.innerHTML = '';
  ui.shift.start_label = document.createElement("label");
  ui.shift.start_label.innerHTML = "Shift Start:";
  appendToRow(ui.shift.start_date_row, ui.shift.start_label);
  ui.shift.start = document.createElement("label");
  ui.shift.start.innerHTML = convert_unix_time_to_string(current_shift.start_time);
  appendToRow(ui.shift.start_date_row, ui.shift.start);

  removeNode(ui.shift.start_time_row);

  var close_shift_button = document.createElement("input");
  close_shift_button.type = "submit";
  close_shift_button.value = "Close Shift";
  appendToRow(ui.shift.end_time_row, close_shift_button);
  close_shift_button.onclick = function() {
    if (! ui.shift.end_timeBox.value) {
      alert("Please enter an end shift time before closing shift.\n");
      return;
    }
    var end_time_part = parse_unix_time_part(ui.shift.start_timeBox.value);
    if (isNaN(end_time_part)) {
      alert("The end time is not valid.");
      return;
    }
    end_time = start_date_part + end_time_part;
    if (end_time <= start_time) {
      end_time += 24 * 3600; // shift spanned midnight, so add a day.
    }
    current_shift.end_time = end_time;

    ui.shift.end_time_row.innerHTML = "";

    var label1 = document.createElement("label");
    label1.innerHTML = "Shift End:";
    appendToRow(ui.shift.end_time_row, label1);

    var label2 = document.createElement("label");
    label2.innerHTML = convert_unix_time_to_string(current_shift.end_time);
    appendToRow(ui.shift.end_time_row, label2);

    var result = ws_update_shift(current_shift);
    if (result.ws_error) {
      error("failed to close shift: " + result.ws_error);
      return false; // XXX don't freeze the UI *until* this works.
    } else {
      current_shift.last_modified_time = result.last_modified_time;
    }
  };

  updateUI_CalculatedValues();

  ui.area_evaluation = {};
  for (i = 0; i < areas.length; i++) {
    addRow_AreaEvaluation(areas[i]);
  }

  ui.time_use = {};
  for (i = 0; i < uses.length; i++) {
    addRow_TimeUse(uses[i]);
  }
  fixTimeUsePercentages();
}

function onchange_AreaEvaluation(area) {
  var area_evaluation_ui = ui.area_evaluation[area];
  var buttons = area_evaluation_ui.buttons;
  var timeBox = area_evaluation_ui.timeBox;
  var commentBox = area_evaluation_ui.commentBox;

  // Calculate:
  //     ok
  //     downtime
  //     comment
  // and see if they differ from what's in current_shift.area_evaluation[area].
  var ok = buttons[0].checked;
  var downtime = 0;
  var comment = "";
  if (ok) {
    if (! timeBox.disabled) {
      timeBox.disabled = true;
      timeBox.value = "";
    }
    if (! commentBox.disabled) {
      commentBox.disabled = true;
      commentBox.value = "";
    }
  } else {
    if (timeBox.disabled) {
      timeBox.disabled = false;
      timeBox.value = "";
    } else {
      var downtime = parse_unix_time_part(timeBox.value);
      if (isNaN(downtime)) {
        alert('"' + timeBox.value + '" is not a valid time.');
        downtime = 0;
      }
      // put in standard format
      timeBox.value = secondsToString(downtime, false);
    }
    if (commentBox.disabled) {
      commentBox.disabled = false;
    } else {
      comment = commentBox.value;
    }
  }
  var area_evaluation = current_shift.area_evaluation[area];

  if (ok === area_evaluation.ok &&
      downtime === area_evaluation.downtime &&
      comment === area_evaluation.comment) {
    return; // nothing actually changed
  }

  area_evaluation.ok = ok;
  area_evaluation.downtime = downtime;
  area_evaluation.comment = comment;

  // Update the server
  var result = ws_update_area_evaluation(area);
  if (result.ws_error) {
    error("Could not update area evaluation: " + result.ws_error);
    return;
  }
  current_shift.last_modified_time = result.last_modified_time;
}

function updateUI_AreaEvaluation(area) {
  var ui_area_evaluation = ui.area_evaluation[area];
  var area_evaluation = current_shift.area_evaluation[area];
  var ok = (String(area_evaluation.ok) === "1");
  ui_area_evaluation.buttons[0].checked = ok;
  ui_area_evaluation.buttons[1].checked = ! ok;
  ui_area_evaluation.timeBox.disabled = ok;
  ui_area_evaluation.timeBox.value = "";
  ui_area_evaluation.commentBox.disabled = ok;
  ui_area_evaluation.commentBox.value = "";
  if (! ok) {
    ui_area_evaluation.timeBox.value = secondsToString(area_evaluation.downtime, false);
    if (ui_area_evaluation.commentBox.value != area_evaluation.comment) {
      ui_area_evaluation.commentBox.value = area_evaluation.comment;
    }
  }
}

function checkChangeAreaEvaluationUI(area) {
  var ui_area_evaluation = ui.area_evaluation[area];
  var area_evaluation = current_shift.area_evaluation[area];
  var ok = (String(area_evaluation.ok) === "1");

  // Don't bother checking the buttons because
  // they call onchanged() as soon as they are clicked.
  //
  // Don't bother with downtime because it can cause
  // a partially entered time to be parsed or misparsed
  // which is irritating to the user. The only way
  // around this is to keep more state around to see
  // if the downtime has not changed in the past
  // many seconds.
  //
  // So that really only leaves the comment text box.

  if (! ok) {
    if (ui_area_evaluation.commentBox.value !== area_evaluation.comment) {
      return true;
    }
  }
  return false;
}

function addRow_AreaEvaluation(area) {
  // Create empty ui object and add it to map
  var ui_area_evaluation = ui.area_evaluation[area] = {};
  ui_area_evaluation.area = area;

  // Add label to row
  var row = document.createElement("tr");
  var textNode = document.createElement("div");
  textNode.innerHTML = area + ":";
  if (area === "FEL") {
    textNode.title = "To report problems with the machine, operations, FEE, etc.";
  } else if (area === "Beamline") {
    textNode.title = "To report problems with the photon beamline instrument, including HPS and PPS problems.";
  } else if (area === "Controls") {
    textNode.title = "To report problems specific to controls like motors, cameras, MPS, laser controls, etc.";
  } else if (area === "DAQ") {
    textNode.title = "To report DAQ computer, data transfer and device problems.";
  } else if (area === "Laser") {
    textNode.title = "To report problems with the laser (but not laser controls) and timing system.";
  } else if (area === "Hutch/Hall") {
    textNode.title = "To report problem with the hutch like: PCW, temperature, setup space, common stock, etc.";
  } else if (area === "Other") {
    textNode.title = "Any other areas that might have problems can be addressed.";
  }
  appendToRow(row, textNode);

  // Add buttons to row
  ui_area_evaluation.buttons = [];
  var i;
  for (i = 0; i < 2; i++) {
    var button = document.createElement("input");
    button.type = "radio";
    button.name = "area_button_" + area;
    appendToRow(row, button);
    ui_area_evaluation.buttons[i] = button;
  }

  // Add downtime input to row
  ui_area_evaluation.timeBox = createTextBox();
  appendToRow(row, ui_area_evaluation.timeBox);

  // Add comment input to row
  ui_area_evaluation.commentBox = createCommentBox();
  appendToRow(row, ui_area_evaluation.commentBox);

  // Add row to table
  var body = document.getElementById("area_evaluation_table");
  body.appendChild(row);

  // Set ui elements to refect current_shift.area_evaluation[area]
  updateUI_AreaEvaluation(area);

  // Finally, set up onchange handlers
  var onchange = function() {
    onchange_AreaEvaluation(area);
  }
  for (i = 0; i < ui_area_evaluation.buttons.length; i++) {
    ui_area_evaluation.buttons[i].onchange = onchange;
  }
  ui_area_evaluation.timeBox.onchange = onchange;
  ui_area_evaluation.commentBox.onchange = checkForServerUpdatesTimer;
}

function fixTimeUsePercentages() {
  var use_time_total;
  var total_time;
  var use_time_map;
  var percent;
  var i, use, use_time;

  // Calculate total time (in seconds).
  // If there is an end time, then total time = (end_time - start_time).
  // Otherwise, if there is no end time, then total time = (now - start_time).
  if (current_shift.end_time) {
    total_time = current_shift.end_time - current_shift.start_time;
  } else {
    total_time = now_seconds() - current_shift.start_time;
  }
  // But in either case, if the calculated total time does not look "reasonable",
  // then use 12 hours as the total time instead.
  if (total_time <= 0) {
    total_time = 12 * 3600;
  }

  // Calculate sum of use times (not including Other or Total)
  use_time_total = 0;
  use_time_map = {};
  for (i = 0; i < uses.length; i++) {
    use = uses[i];
    if (use !== "Other" && use !== "Total") {
      use_time = parse_unix_time_part(ui.time_use[use].timeBox.value);
      if (use_time) {
        use_time_total += use_time;
        use_time_map[use] = use_time;
      } else {
        use_time_map[use] = 0;
      }
    }
  }

  // One more "reasonableness" test for total_time.
  // This avoids the situation where someone pre-enters values
  // for uses when the shift is only a few minutes old.
  if (total_time < use_time_total) {
    alert("Warning: total use time (" + secondsToString(use_time_total) + ") exceeds total shift time (" + secondsToString(total_time) + ").");
    total_time = use_time_total;
  }
  use_time_map.Total = total_time;

  // "Other" time is total_time (length of shift)
  // minus use_time_total (sum of use times entered in ui).
  use_time_map.Other = total_time - use_time_total;

  // Set time and percentages in the ui.
  for (i = 0; i < uses.length; i++) {
    use = uses[i];
    use_time = use_time_map[use];
    percent = 100 * use_time / total_time;
    if (isNaN(percent) || percent < 0 || percent > 100) {
      percent = 0;
    }
    ui.time_use[use].percentBox.innerHTML = String(Math.round(percent)) + "%";
    if (use === "Other" || use === "Total") {
      ui.time_use[use].timeBox.innerHTML = secondsToString(use_time, false);
    } else {
      ui.time_use[use].timeBox.value = secondsToString(use_time, false);
    }
  }
}

function addRow_TimeUse(use) {
  // Get reference to model data
  var time_use = current_shift.time_use[use];

  // Create empty ui object and add it to map
  var ui_time_use = ui.time_use[use] = {};
  ui_time_use.use = use;
  
  var row = document.createElement("tr");
  var textNode = document.createElement("div");
  textNode.innerHTML = use + ":";
  if (use === "Tuning") {
    textNode.title = "This is machine tuning time.";
  } else if (use === "Alignment") {
    textNode.title = "This is time spent aligning, calibrating, and turning the photon instrumentation.";
  } else if (use === "Data Taking") {
    textNode.title = "This is time spent taking data that can be used in publication.";
  } else if (use === "Access") {
    textNode.title = "This is time spent in the hutch for sample changes, laser tuning, trouble shooting and the like.";
  } else if (use === "Other") {
    textNode.title = "This is any other circumstance: machine downtime, extended specific activities, etc.";
  } else if (use === "Total") {
    textNode.title = "This is the total shift time.";
  }
  appendToRow(row, textNode);

  if (use === "Other" || use === "Total") {
    ui_time_use.timeBox = document.createElement("label");
    ui_time_use.timeBox.type = "text";
    ui_time_use.timeBox.value = "0:00";
  } else {
    ui_time_use.timeBox = document.createElement("input");
    ui_time_use.timeBox.value = secondsToString(time_use.use_time, false);
  }
  appendToRow(row, ui_time_use.timeBox);

  ui_time_use.percentBox = document.createElement("label");
  ui_time_use.percentBox.innerHTML = "";
  appendToRow(row, ui_time_use.percentBox);

  ui_time_use.commentBox = createCommentBox();
  ui_time_use.commentBox.onchange = checkForServerUpdatesTimer;
  appendToRow(row, ui_time_use.commentBox);

  // Add row to table
  var body = document.getElementById("time_use_allocation_table");
  body.appendChild(row);

  // Finally, set up onchange handlers
  ui_time_use.timeBox.onchange = onchange;

  function onchange() {
    var use_time = parse_unix_time_part(ui_time_use.timeBox.value);
    if (isNaN(use_time)) {
      alert('"' + ui_time_use.timeBox.value + '" is not a valid time.');
      ui_time_use.timeBox.value = "";
      use_time = 0;
    }
    // put in standard format
    log("use_time for " + use + " is now " + use_time);
    ui_time_use.timeBox.value = secondsToString(use_time, false);
    log("use_time for " + use + " is now " + ui_time_use.timeBox.value);

    var comment = ui_time_use.commentBox.value;
    var time_use = current_shift.time_use[use];
    if (time_use.use_time == use_time &&
        time_use.comment == comment) {
      debug("nothing changed in use allocation");
      return; // nothing actually changed
    }

    fixTimeUsePercentages();

    debug("will update time_use");

    time_use.use_time = use_time;
    time_use.comment = comment;

    // Update the server
    var result = ws_update_time_use(use);
    if (result.ws_error) {
      error("Could not update time use: " + result.ws_error);
      return;
    }
    current_shift.last_modified_time = result.last_modified_time;
  }
}

function updateUI_TimeUse(use) {
  // Get reference to model data
  var time_use = current_shift.time_use[use];

  // Update time and comment
  var ui_time_use = ui.time_use[use];
  if (ui_time_use.commentBox.value != time_use.comment) {
    ui_time_use.commentBox.value = time_use.comment;
  }
  if (ui_time_use.timeBox.value !== time_use.use_time / 3600) {
    ui_time_use.timeBox.value = time_use.use_time / 3600;
    fixTimeUsePercentages();
  }
}

function updateUI_CalculatedValues() {
  if (! current_shift) {
    log("updateUI_CalculatedValues: there is no current shift.");
    return;
  }
  var start = current_shift.start_time;
  var stop = current_shift.end_time;
  var now = now_seconds();
  //log("start=" + start + ", stop=" + stop + ", now=", now);
  if (stop == undefined || stop > now) {
    stop = now;
  }
  if (start > stop) {
    console.error("updateUI_CalculatedValues: start > stop!");
    return; // nonsense
  }
  var total_time = stop - start;
  var result = ws_get_pv("XRAY_DESTINATIONS", start, stop);
  if (result.ws_error) {
    console.error("could not fetch pvs: " + result.ws_error);
    return;
  }
  var values = result.values;
  var beam_time = 0;
  var hutch = ui.shift.hutch_select.value;
  var hutch_mask = beam_destination_masks[hutch];
  for (var i = 0; i < values.length; i++) {
    var value = values[i];
    if (value.status & hutch_mask) {
      var begin_time = value.begin_time.sec + value.begin_time.nsec / 10e9;
      var end_time = value.end_time.sec + value.end_time.nsec / 10e9;
      beam_time += (end_time - begin_time);
    }
  }
  //log("beam_time = " + beam_time + " seconds, " + beam_time / 3600 + " hours");
  //log("total_time = " + total_time + " seconds, " + total_time / 3600 + " hours");

  var beam_percent = Math.round(100 * beam_time / total_time);
  document.getElementById("stopper_out_time").innerHTML = secondsToString(beam_time) + " hours";
  document.getElementById("stopper_out_percent").innerHTML = String(beam_percent) + "%";

  // XXX these need to be filled in with real values
  document.getElementById("door_open_time").innerHTML = "unknown"
  document.getElementById("door_open_percent").innerHTML = "unknown %"

  document.getElementById("total_shots_count").innerHTML = "unknown"
  document.getElementById("total_shots_percent").innerHTML = "unknown %";
}

function checkForServerUpdatesTimer() {
  updateUI_CalculatedValues();
  function callback(result) {
    checkForServerUpdates_Process(result);
  }
  ws_get_shift_last_modified_time_async(current_shift.id, callback);
}

function getInternetExplorerVersion() {
  if (navigator.appName === 'Microsoft Internet Explorer') {
    var ua = navigator.userAgent;
    var re = new RegExp("MSIE ([0-9]{1,}[.0-9]{0,})");
    if (re.exec(ua)) {
      return parseFloat(RegExp.$1);
    }
  }
  return 0;
}

function onload() {
  var result, obsolete, ieVersion;
  // Check for required browser functionality.
  obsolete = false;
  try {
    throw new Error();
  } catch (e) {
    if (! e.stack) {
      obsolete = true;
    }
  }
  try {
    JSON.stringify(0);
  } catch (e) {
    obsolete = true;
  }

  // If the browser is missing functionality, complain and quit.
  if (obsolete) {
    ieVersion = getInternetExplorerVersion();
    if (ieVersion) {
      alert("Internet Explorer " + ieVersion + " is not supported.\nPlease upgrade, or try Chrome or Firefox.");
    } else {
      alert("Your browser is missing required functionality. Please use a newer version or a different browser.");
    }
    return;
  }

  // Fetch list of areas.
  result = ws_get_areas();
  if (result.ws_error) {
    error("unable to fetch list of areas: " + result.ws_error);
    return;
  }
  areas = result.areas;

  // Fetch list of time uses.
  result = ws_get_uses();
  if (result.ws_error) {
    error("unable to fetch list of uses: " + result.ws_error);
    return;
  }
  uses = result.uses;

  // Fetch beam destination masks.
  result = ws_get_beam_destination_masks();
  if (result.ws_error) {
    error("unable to fetch beam destination masks: " + result.ws_error);
    return;
  }
  beam_destination_masks = result.masks;
  if (beam_destination_masks === null) {
    // XXX
    beam_destination_masks = {};
    beam_destination_masks["FEE"] = 1;
    beam_destination_masks["AMO"] = 2;
    beam_destination_masks["SXR"] = 4;
    beam_destination_masks["XPP"] = 8;
    beam_destination_masks["XRT"] = 16;
    beam_destination_masks["XCS"] = 32;
    beam_destination_masks["CXI"] = 64;
    beam_destination_masks["MEC"] = 128;
  }

  // Start the application.
  addGeneralShiftInformationRows();
}

if (false) {
  str(onload()); // fixes JSHint "defined but never used"
}
