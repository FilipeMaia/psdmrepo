<!DOCTYPE html>
<html>
<head>
<title>An example of the Display Selector widget</title>

<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta name="viewport" content="width=device-width, height=device-height, initial-scale=1.0">

<style>
body {
    margin: 0;
    padding: 0;
}
h2 {
    font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
}
.test {
    padding: 20px;
}
</style>

<script data-main="js/test_displayselector_main.js?bust=<?=date_create()->getTimestamp()?>" src="js/require/require.js"></script>

</head>
<body>
    <div class="test">
        <h2>Display Selector</h2>
        <div id="display" ></div>
    </div>
</body>
</html>