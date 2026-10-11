/* Room Climate load probe. Deliberately ES5: it must run on old WebViews where room-map-card.js may fail to parse. */
(function () {
  var errors = [];
  window.addEventListener("error", function (e) {
    if (e && e.filename && e.filename.indexOf("room-map-card") !== -1) errors.push((e.message || "error") + " @" + e.lineno + ":" + e.colno);
  });
  function report(message) {
    var ha = document.querySelector("home-assistant");
    if (!ha || !ha.hass || !ha.hass.callWS) return false;
    ha.hass.callWS({ type: "room_climate/log_client", message: message, stack: "", ua: navigator.userAgent, where: location.pathname })["catch"](function () {});
    return true;
  }
  var tries = 0;
  var timer = setInterval(function () {
    tries++;
    var defined = !!(window.customElements && customElements.get("room-map-card"));
    if (!defined && tries === 8 || (!defined && errors.length)) {  // ~16 s after load
      if (report("probe: room-map-card NOT defined after " + (tries * 2) + "s; script errors: " + (errors.join("; ") || "none"))) clearInterval(timer);
    } else if (defined || tries > 30) clearInterval(timer);
  }, 2000);
})();
