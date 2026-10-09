/* Room map card: click a room on the floorplan, set target / priority / schedule.
 * Resource: /room_climate_static/room-map-card.js (type: module)
 * Card config:  type: custom:room-map-card
 *               floorplan: /local/floorplan.svg   # optional; a placeholder grid is drawn otherwise
 */
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

class RoomMapCard extends HTMLElement {
  setConfig(config) { this._config = config; this._sel = null; this._floor = 1; this._data = null; this._build(); }
  set hass(hass) {
    this._hass = hass;
    if (!this._data && !this._loading) this._refresh();
    const now = Date.now();
    if (!this._last || now - this._last > 15000) { this._last = now; this._refresh(); }
  }
  getCardSize() { return 6; }

  async _refresh() {
    this._loading = true;
    try { this._data = await this._hass.callWS({ type: "room_climate/get" }); this._render(); }
    finally { this._loading = false; }
  }
  async _set(patch) {
    this._data = await this._hass.callWS({ type: "room_climate/set_room", room_id: this._sel, ...patch });
    this._render();
  }
  _color(t, target) {
    if (t == null) return "#888";
    const d = t - target;
    return d < -1 ? "#4a90d9" : d > 1 ? "#e0763a" : "#4caf7d";
  }
  _build() {
    this.innerHTML = `<ha-card header="Room climate"><div class="wrap" style="padding:0 16px 16px"></div></ha-card>`;
    this._wrap = this.querySelector(".wrap");
  }
  _render() {
    const d = this._data; if (!d) return;
    const floors = [...new Set(d.rooms.map(r => r.floor || 1))].sort();
    const rooms = d.rooms.filter(r => (r.floor || 1) === this._floor);
    const bg = this._config.floorplan
      ? `<image href="${this._config.floorplan}" width="100" height="100" preserveAspectRatio="xMidYMid meet"/>`
      : `<rect width="100" height="100" fill="none" stroke="currentColor" stroke-opacity=".2"/>`;
    const shapes = rooms.map(r => {
      const live = d.live[r.id] || {}, s = r.shape || {};
      const sel = r.id === this._sel;
      return `<g data-room="${r.id}" style="cursor:pointer">
        <rect x="${s.x}" y="${s.y}" width="${s.w}" height="${s.h}" rx="1.5" fill="${this._color(live.temp, live.target)}"
              fill-opacity=".35" stroke="${sel ? "var(--primary-color)" : "currentColor"}" stroke-width="${sel ? 1 : .3}"/>
        <text x="${s.x + s.w / 2}" y="${s.y + s.h / 2 - 1}" text-anchor="middle" font-size="4" fill="currentColor">${r.name}</text>
        <text x="${s.x + s.w / 2}" y="${s.y + s.h / 2 + 5}" text-anchor="middle" font-size="3.5" fill="currentColor">
          ${live.temp == null ? "no sensor" : live.temp.toFixed(1) + "° → " + live.target + "°"}</text></g>`;
    }).join("");
    const tabs = floors.length > 1 ? floors.map(f =>
      `<button data-floor="${f}" ${f === this._floor ? "disabled" : ""}>Floor ${f}</button>`).join(" ") : "";
    this._wrap.innerHTML = `${tabs}<svg viewBox="0 0 100 100" style="width:100%;max-height:360px">${bg}${shapes}</svg><div class="panel"></div>`;
    this._wrap.querySelectorAll("[data-floor]").forEach(b => b.onclick = () => { this._floor = +b.dataset.floor; this._sel = null; this._render(); });
    this._wrap.querySelectorAll("[data-room]").forEach(g => g.onclick = () => { this._sel = g.dataset.room; this._render(); });
    this._panel();
  }
  _panel() {
    const room = this._data.rooms.find(r => r.id === this._sel);
    const el = this._wrap.querySelector(".panel");
    if (!room) { el.innerHTML = `<p>Select a room on the map.</p>`; return; }
    const live = this._data.live[room.id] || {};
    const sched = (room.schedule || []).map((e, i) => `<tr><td>${e.days.length === 7 ? "Every day" : e.days.map(x => DAYS[x]).join(" ")}</td>
        <td>${e.start}</td><td>${e.temp}°</td><td><button data-del="${i}">✕</button></td></tr>`).join("");
    el.innerHTML = `<h3>${room.name}</h3>
      <p>${live.temp == null ? "No sensor reading" : live.temp.toFixed(1) + "°F"} · target ${live.target}° · unit ${live.mode}${live.reason ? " (" + live.reason + ")" : ""}</p>
      <label>Default temp <input type="number" id="def" step="0.5" value="${room.default_temp}" style="width:5em">°F</label>
      <label style="margin-left:1em">Priority <input type="range" id="prio" min="1" max="5" value="${room.priority}"> <b>${room.priority}</b></label>
      <p>Hold <input type="number" id="hold" step="0.5" value="${room.hold_temp ?? room.default_temp}" style="width:5em">°F for
         <input type="number" id="holdh" value="4" style="width:3em"> h <button id="holdgo">Hold</button> <button id="holdx">Clear</button></p>
      <table>${sched}</table>
      <p>Add: ${DAYS.map((n, i) => `<label><input type="checkbox" data-d="${i}" checked>${n}</label>`).join(" ")}
         <input type="time" id="ns" value="07:00"> <input type="number" id="nt" step="0.5" value="68" style="width:5em">°F <button id="add">Add</button></p>`;
    const q = s => el.querySelector(s);
    q("#def").onchange = e => this._set({ default_temp: +e.target.value });
    q("#prio").onchange = e => this._set({ priority: +e.target.value });
    q("#holdgo").onclick = () => this._set({ hold_temp: +q("#hold").value, hold_hours: +q("#holdh").value });
    q("#holdx").onclick = () => this._set({ hold_temp: null, hold_hours: null });
    el.querySelectorAll("[data-del]").forEach(b => b.onclick = () => this._set({ schedule: room.schedule.filter((_, i) => i !== +b.dataset.del) }));
    q("#add").onclick = () => {
      const days = [...el.querySelectorAll("[data-d]")].filter(c => c.checked).map(c => +c.dataset.d);
      if (!days.length) return;
      this._set({ schedule: [...(room.schedule || []), { days, start: q("#ns").value, temp: +q("#nt").value }] });
    };
  }
}
customElements.define("room-map-card", RoomMapCard);
window.customCards = (window.customCards || []).concat([{ type: "room-map-card", name: "Room map card" }]);
