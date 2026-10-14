/* Room Climate load probe. Deliberately ES5: it must run on old WebViews where room-map-card.js may fail to parse.
   HA loads extra modules with import(), and a failed or stalled import (phone waking up, Wi-Fi <-> cellular
   handoff) never retries and raises no error event, so the card stays undefined all session. If the card isn't
   registered after a few seconds, load it again with a cache-busting URL, and report what happened. */
(function () {
  var CARD = "/room_climate_static/room-map-card.js";
  var RETRY_AT = [3, 6, 10];  // ticks (2 s each) at which to re-load the card
  var errors = [];
  window.addEventListener("error", function (e) {
    if (e && e.filename && e.filename.indexOf("room-map-card") !== -1) errors.push((e.message || "error") + " @" + e.lineno + ":" + e.colno);
  });
  window.addEventListener("unhandledrejection", function (e) {
    var r = e && e.reason, msg = String((r && (r.message || r)) || "");
    if (msg.indexOf("room-map-card") !== -1 || msg.indexOf("room_climate_static") !== -1) errors.push("import rejected: " + msg);
  });
  function report(message) {
    var ha = document.querySelector("home-assistant");
    if (!ha || !ha.hass || !ha.hass.callWS) return false;
    ha.hass.callWS({ type: "room_climate/log_client", message: message, stack: "", ua: navigator.userAgent, where: location.pathname })["catch"](function () {});
    return true;
  }
  function defined() { return !!(window.customElements && customElements.get("room-map-card")); }
  function reload(n) {
    var s = document.createElement("script");
    s.type = "module";
    s.src = CARD + "?retry=" + n + "&t=" + Date.now();
    s.onerror = function () { errors.push("retry " + n + " failed to load"); };
    document.head.appendChild(s);
  }
  var tries = 0, retries = 0, reported = false;
  var timer = setInterval(function () {
    tries++;
    if (defined()) {
      if (retries) report("probe: room-map-card recovered after " + retries + " retr" + (retries > 1 ? "ies" : "y") +
                          (errors.length ? "; earlier errors: " + errors.join("; ") : ""));
      clearInterval(timer);
      return;
    }
    if (RETRY_AT.indexOf(tries) !== -1) { retries++; reload(retries); }
    if (!reported && tries === 15) {  // ~30 s, after every retry had time to land
      reported = report("probe: room-map-card NOT defined after " + (tries * 2) + "s and " + retries +
                        " retries; script errors: " + (errors.join("; ") || "none"));
    }
    if (tries > 45) clearInterval(timer);
  }, 2000);
})();
