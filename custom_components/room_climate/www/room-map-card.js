/* Room map card: tap a room on the floorplan, set target / priority / hold, and edit its schedule on a curve.
 * Resource: /room_climate_static/room-map-card.js (type: module)
 * Card config:  type: custom:room-map-card
 *               floorplan: /local/floorplan.svg   # optional; a placeholder grid is drawn otherwise
 *               # Optional: one image holding several floors. Room shapes are then in image pixels.
 *               image_size: [1280, 960]
 *               floor_views: {2: {x: 440, y: 60, w: 400, h: 710}, 3: {...}}   # crop shown per floor
 *               theme: tech                       # stylized dark schematic drawn from room shapes (no image needed)
 *               north: left                       # up | down | left | right: draws a north arrow
 * Rooms without a `unit` are monitor-only (shown in blue on the tech theme).
 */
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const TEMP_MIN = 55, TEMP_MAX = 85;  // schedule editor range (F)

const toMin = s => { const [h, m] = s.split(":"); return +h * 60 + +m; };
const toHHMM = m => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
const fmtTime = m => { const h = Math.floor(m / 60), mm = String(m % 60).padStart(2, "0"); return `${h % 12 || 12}:${mm} ${h < 12 ? "AM" : "PM"}`; };
const todayIdx = () => (new Date().getDay() + 6) % 7;  // Mon=0
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const CSS = `
.rc { color: var(--primary-text-color, #e6edf7); font-size: 16px; padding: 0 12px 12px; box-sizing: border-box; }
.rc *, .rc-modal * { box-sizing: border-box; }
.rc button, .rc-modal button { min-height: 44px; min-width: 44px; padding: 0 14px; font: inherit; color: inherit; cursor: pointer;
  background: var(--secondary-background-color, #263146); border: 1px solid var(--divider-color, #ffffff24); border-radius: 10px;
  touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
.rc button.on, .rc-modal button.on { background: var(--primary-color, #03a9f4); border-color: transparent; color: #fff; }
.rc button:disabled, .rc-modal button:disabled { opacity: .4; }
.rc-seg { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 12px; }
.rc-seg button { flex: 1 1 0; }
.rc-main { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.rc-map { flex: 1 1 340px; min-width: 0; }
.rc-map svg { display: block; width: 100%; height: auto; max-height: 72vh; margin: 0 auto; touch-action: manipulation; }
.rc-panel { flex: 1 1 320px; min-width: 0; }
.rc-panel h3 { margin: 0 0 4px; font-size: 1.3em; }
.rc-sub { color: var(--secondary-text-color, #9aa7b8); margin: 0 0 12px; }
.rc-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 10px 0; flex-wrap: wrap; }
.rc-row > label, .rc-row > .rc-lbl { flex: 1 1 120px; }
.rc-step { display: flex; align-items: center; gap: 4px; }
.rc-step span { min-width: 4.2em; text-align: center; font-size: 1.25em; font-variant-numeric: tabular-nums; }
.rc-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.rc-sect { border-top: 1px solid var(--divider-color, #ffffff24); margin-top: 12px; padding-top: 4px; }
.rc-sect h4 { margin: 10px 0 0; font-size: .9em; text-transform: uppercase; letter-spacing: .06em; color: var(--secondary-text-color, #9aa7b8); }
.rc-modal { position: fixed; inset: 0; z-index: 9999; display: flex; align-items: center; justify-content: center;
  background: rgba(0, 0, 0, .65); color: var(--primary-text-color, #e6edf7); font-family: system-ui, sans-serif; font-size: 16px; }
.rc-sheet { background: var(--card-background-color, #121a28); width: min(760px, 100%); max-height: 100%; display: flex; flex-direction: column;
  border-radius: 16px; overflow: hidden; padding-bottom: env(safe-area-inset-bottom); }
.rc-sheet header { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; gap: 12px;
  border-bottom: 1px solid var(--divider-color, #ffffff24); }
.rc-sheet header h3 { margin: 0; font-size: 1.15em; }
.rc-body { padding: 12px 16px; overflow: auto; }
.rc-sheet footer { display: flex; gap: 10px; padding: 12px 16px; border-top: 1px solid var(--divider-color, #ffffff24); }
.rc-sheet footer button { flex: 1; }
.rc-chart { width: 100%; height: auto; display: block; overflow: visible; touch-action: none; user-select: none; -webkit-user-select: none; }
.rc-units { display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr)); gap: 8px; margin: 8px 0 4px; }
.rc-tile { border-radius: 12px; padding: 8px 10px; background: #ffffff0d; border-top: 3px solid #667085; min-width: 0; }
.rc-tile.heat { border-color: #ff7a45; background: #ff7a4514; } .rc-tile.cool { border-color: #3fa9ff; background: #3fa9ff14; }
.rc-tile .nm { font-size: .8em; color: var(--secondary-text-color, #9aa7b8); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rc-tile .big { font-size: 1.35em; font-weight: 700; line-height: 1.3; } .rc-tile.heat .big { color: #ff8a5c; } .rc-tile.cool .big { color: #5db3ff; }
.rc-tile .big small { font-size: .55em; font-weight: 400; color: var(--secondary-text-color, #9aa7b8); }
.rc-tile .why { font-size: .72em; color: var(--secondary-text-color, #9aa7b8); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 4px; }
.rc-g { position: relative; height: 8px; border-radius: 4px; background: #ffffff14; margin-top: 6px; }
.rc-g .band { position: absolute; top: 0; bottom: 0; left: 50%; transform: translateX(-50%); background: #3de0a533; border-radius: 4px; }
.rc-g .tick { position: absolute; left: 50%; top: -2px; bottom: -2px; width: 2px; margin-left: -1px; background: #ffffff55; }
.rc-g i { position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%; background: #98a2b3; border: 2px solid #0d131d; }
.rc-g i.hot { background: #ff8a5c; } .rc-g i.cold { background: #5db3ff; } .rc-g i.ok { background: #3de0a5; }
.rc-rows { margin-top: 4px; }
.rc-rm { display: grid; grid-template-columns: minmax(0, 36%) 1fr 3.4em; align-items: center; gap: 10px; width: 100%; min-height: 40px; margin: 2px 0; padding: 0 8px; text-align: left; }
.rc-rm .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: .95em; }
.rc-rm .nm .star { color: #ffd166; margin-left: 4px; }
.rc-pips { display: inline-flex; gap: 2px; margin-left: 6px; vertical-align: middle; }
.rc-pips b { width: 5px; height: 5px; border-radius: 50%; background: #ffffff2e; } .rc-pips b.on { background: var(--primary-color, #03a9f4); }
.rc-rm .t { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.rc-pr { display: flex; align-items: center; justify-content: space-between; gap: 10px; width: 100%; min-height: 44px; margin: 4px 0; padding: 0 12px; text-align: left; }
.rc-pr .nm { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rc-pr .nm .star { color: #ffd166; margin-left: 4px; }
.rc-pr .rc-d { font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.rc-pr .rc-d small { font-weight: 400; color: var(--secondary-text-color, #9aa7b8); }
.rc-chips2 { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.rc-chips2 button { min-height: 36px; padding: 0 10px; font-size: .9em; }
.rc-chips2 b { margin-left: 6px; color: #5db3ff; }
.rc-legend { display: flex; justify-content: space-between; font-size: .7em; color: var(--secondary-text-color, #9aa7b8); padding: 0 8px; margin-top: 2px; }
@media (max-width: 600px) {
  .rc-modal { align-items: flex-end; }
  .rc-sheet { border-radius: 16px 16px 0 0; height: 100%; max-height: 100%; }
}
`;

class RoomMapCard extends HTMLElement {
  setConfig(config) {
    this._config = config; this._sel = null; this._floor = 1; this._data = null;
    try { this._build(); } catch (e) { this._fail("setConfig", e); }
  }
  set hass(hass) {
    this._hass = hass;
    if (!this._data && !this._loading) this._refresh();
    const now = Date.now();
    if (!this._last || now - this._last > 15000) { this._last = now; this._refresh(); }
  }
  getCardSize() { return 8; }
  disconnectedCallback() { this._closeSchedule(); }

  async _refresh() {
    this._loading = true;
    try {
      this._data = await this._hass.callWS({ type: "room_climate/get" });
      this._failed = null;
      if (!this._modal) this._render();  // don't rebuild the page under an open editor
      else if (this._diag) this._diagUpdate();  // diagnostics stay live
    } catch (e) { this._fail("refresh", e); }
    finally { this._loading = false; }
  }
  async _set(patch) {
    try {
      this._data = await this._hass.callWS({ type: "room_climate/set_room", room_id: this._sel, ...patch });
      this._render();
    } catch (e) { this._fail("set_room", e); }
  }
  // Show the problem in the card (instead of a blank page) and send it to the HA log so it can be diagnosed remotely.
  _fail(where, e) {
    const msg = `${where}: ${(e && e.message) || (e && e.code) || JSON.stringify(e) || e}`;
    console.error("[room-map-card]", msg, e);
    if (this._failed !== msg) {  // report each distinct failure once, not on every 15s retry
      this._failed = msg;
      try { this._hass.callWS({ type: "room_climate/log_client", message: msg, stack: String((e && e.stack) || "").slice(0, 1500), ua: navigator.userAgent, where: location.pathname }).catch(() => {}); } catch (_) { /* hass not ready */ }
    }
    if (this._wrap && !this._data) this._wrap.innerHTML = `<div style="padding:16px;color:#ff8a5c;font-family:monospace;font-size:13px;white-space:pre-wrap">Room Climate could not load\n${msg.replace(/[<&]/g, "")}\nRetrying every 15 s.</div>`;
  }
  _color(t, target) {
    if (t == null) return "#888";
    const d = t - target;
    return d < -1 ? "#4a90d9" : d > 1 ? "#e0763a" : "#4caf7d";
  }
  _build() {
    // Styles are inlined: HA renders cards inside shadow DOM, where <head> stylesheets don't apply.
    this.innerHTML = `<style>${CSS}</style><ha-card header="Room climate"><div class="rc"></div></ha-card>`;
    this._wrap = this.querySelector(".rc");
    this._wrap.addEventListener("click", e => this._click(e));
  }

  _click(e) {
    const t = e.target.closest("[data-act],[data-room],[data-floor]");
    if (!t || !this._data) return;
    if (t.dataset.act === "diag") { this._openDiag(); return; }
    const room = this._data.rooms.find(r => r.id === this._sel);
    const live = room ? this._data.live[room.id] || {} : {};
    if (t.dataset.floor) { this._floor = +t.dataset.floor; this._sel = null; this._render(); return; }
    if (t.dataset.room) {
      this._sel = t.dataset.room; this._holdTemp = null;
      const r = this._data.rooms.find(x => x.id === this._sel); if (r) this._floor = r.floor || 1;  // overview rows can be on another floor
      this._render(); return;
    }
    const d = +t.dataset.d;
    switch (t.dataset.act) {
      case "def": this._set({ default_temp: room.default_temp + d }); break;
      case "hold-step": this._holdTemp = (this._holdTemp ?? room.hold_temp ?? live.target) + d; this._panel(); break;
      case "prio": this._set({ priority: +t.dataset.v }); break;
      case "hold": this._set({ hold_temp: this._holdTemp ?? room.hold_temp ?? live.target, hold_hours: +t.dataset.h }); break;
      case "hold-clear": this._set({ hold_temp: null, hold_hours: null }); break;
      case "sched": this._openSchedule(room); break;
    }
  }

  _render() {
    const d = this._data; if (!d) return;
    const floors = [...new Set(d.rooms.map(r => r.floor || 1))].sort();
    if (!floors.includes(this._floor)) this._floor = floors[0];
    const rooms = d.rooms.filter(r => (r.floor || 1) === this._floor);
    let view = (this._config.floor_views || {})[this._floor];
    const [iw, ih] = this._config.image_size || [100, 100];
    if (!view && this._config.theme === "tech") view = this._fit(rooms);  // no geometry needed in the card config
    const vb = view ? [view.x, view.y, view.w, view.h] : [0, 0, 100, 100];
    const k = vb[2] / 100;  // 1 on the legacy 0-100 grid; scales text/strokes on a pixel-space crop
    const tech = this._config.theme === "tech";
    const ink = tech ? "#cfe8ff" : "currentColor";
    const bg = tech
      ? `<defs>
           <pattern id="rc-grid" width="${10 * k}" height="${10 * k}" patternUnits="userSpaceOnUse">
             <path d="M${10 * k} 0H0V${10 * k}" fill="none" stroke="#3fa9ff" stroke-opacity=".09" stroke-width="${.25 * k}"/></pattern>
           <filter id="rc-glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="${.9 * k}" result="b"/>
             <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
         </defs>
         <rect x="${vb[0]}" y="${vb[1]}" width="${vb[2]}" height="${vb[3]}" fill="#0a111d"/>
         <rect x="${vb[0]}" y="${vb[1]}" width="${vb[2]}" height="${vb[3]}" fill="url(#rc-grid)"/>`
      : this._config.floorplan
        ? `<image href="${this._config.floorplan}" width="${iw}" height="${view ? ih : 100}" preserveAspectRatio="xMidYMid meet"/>`
        : `<rect width="100" height="100" fill="none" stroke="currentColor" stroke-opacity=".2"/>`;
    const palette = { hot: "#ff7a45", cold: "#3fa9ff", none: "#667085" };
    // tech theme: green = controllable (drives a unit), blue = monitor-only
    const GREEN = "#3de0a5", BLUE = "#3fa9ff";
    const shapes = rooms.map(r => {
      const live = d.live[r.id] || {}, s = r.shape || {};
      const sel = r.id === this._sel;
      const ctl = !!r.unit;
      const col = tech ? (live.temp == null ? palette.none : ctl ? GREEN : BLUE) : this._color(live.temp, live.target);
      const active = tech && (live.mode === "heat" || live.mode === "cool");
      const cx = s.x + s.w / 2, cy = s.y + s.h / 2;
      const txt = live.temp == null ? "no sensor" : live.temp.toFixed(1) + "°";
      if (!tech) return `<g data-room="${r.id}" style="cursor:pointer">
        <rect x="${s.x}" y="${s.y}" width="${s.w}" height="${s.h}" rx="${1.5 * k}" fill="${col}"
              fill-opacity=".35" stroke="${sel ? "var(--primary-color)" : "currentColor"}" stroke-width="${(sel ? 1 : .3) * k}"/>
        <text x="${cx}" y="${cy - k}" text-anchor="middle" font-size="${4 * k}" fill="currentColor">${r.name}</text>
        <text x="${cx}" y="${cy + 5 * k}" text-anchor="middle" font-size="${3.5 * k}" fill="currentColor">
          ${live.temp == null ? "no sensor" : live.temp.toFixed(1) + "° → " + live.target + "°"}</text></g>`;
      const small = Math.min(s.w, s.h) < 20 * k;
      return `<g data-room="${r.id}" style="cursor:pointer" filter="${sel ? "url(#rc-glow)" : ""}">
        <rect x="${s.x}" y="${s.y}" width="${s.w}" height="${s.h}" rx="${1 * k}" fill="${col}" fill-opacity="${sel ? .22 : .1}"
              stroke="${col}" stroke-width="${(sel ? 1.1 : .6) * k}" ${sel ? `stroke-dasharray="${3 * k} ${1.5 * k}"><animate attributeName="stroke-dashoffset" from="0" to="${-9 * k}" dur="1.2s" repeatCount="indefinite"/></rect>` : "/>"}
        <path d="M${s.x} ${s.y + 4 * k}V${s.y}H${s.x + 4 * k}M${s.x + s.w - 4 * k} ${s.y + s.h}H${s.x + s.w}V${s.y + s.h - 4 * k}" fill="none" stroke="${col}" stroke-width="${1.2 * k}"/>
        ${active ? `<circle cx="${s.x + s.w - 3 * k}" cy="${s.y + 3 * k}" r="${1.2 * k}" fill="${live.mode === "heat" ? palette.hot : palette.cold}"><animate attributeName="opacity" values="1;.2;1" dur="1.4s" repeatCount="indefinite"/></circle>` : ""}
        <text x="${cx}" y="${cy - (small ? 1 : 2) * k}" text-anchor="middle" font-size="${(small ? 2.6 : 3.2) * k}" fill="${ink}" fill-opacity=".75" letter-spacing="${.3 * k}">${r.name.toUpperCase()}</text>
        <text x="${cx}" y="${cy + (small ? 3.5 : 5.5) * k}" text-anchor="middle" font-size="${(small ? 4 : 6) * k}" font-weight="600" fill="${col}">${txt}</text>
        ${small ? "" : `<text x="${cx}" y="${cy + 10 * k}" text-anchor="middle" font-size="${2.6 * k}" fill="${ink}" fill-opacity=".55">${ctl ? "TARGET " + live.target + "°" : "MONITOR ONLY"}</text>`}</g>`;
    }).join("");
    const rot = { up: 0, right: 90, down: 180, left: 270 }[this._config.north];
    const compass = rot == null ? "" : `<g transform="translate(${vb[0] + vb[2] - 8 * k} ${vb[1] + 8 * k}) rotate(${rot})" fill="${ink}" opacity=".7">
        <path d="M0 -6 L2.5 2 L0 0.5 L-2.5 2 Z" transform="scale(${k})"/>
        <text y="${-7 * k}" text-anchor="middle" font-size="${3.5 * k}" transform="rotate(${-rot} 0 ${-7 * k})">N</text></g>`;
    const tabs = floors.length > 1 ? `<div class="rc-seg">${floors.map(f =>
      `<button data-floor="${f}" class="${f === this._floor ? "on" : ""}">Floor ${f}</button>`).join("")}</div>` : "";
    this._wrap.innerHTML = `${tabs}<div class="rc-main">
      <div class="rc-map"><svg viewBox="${vb.join(" ")}" style="${tech ? "border-radius:10px" : ""}">${bg}${shapes}${compass}</svg></div>
      <div class="rc-panel"></div></div>`;
    this._panel();
  }

  // Bounding box of a floor's room shapes plus margin (extra on top for the north arrow).
  _fit(rooms) {
    const sh = rooms.map(r => r.shape).filter(s => s && s.w);
    if (!sh.length) return undefined;
    const x0 = Math.min(...sh.map(s => s.x)), x1 = Math.max(...sh.map(s => s.x + s.w));
    const y0 = Math.min(...sh.map(s => s.y)), y1 = Math.max(...sh.map(s => s.y + s.h));
    const pad = .08 * Math.max(x1 - x0, y1 - y0);
    return { x: x0 - pad, y: y0 - pad * 1.7, w: x1 - x0 + pad * 2, h: y1 - y0 + pad * 2.7 };
  }

  _panel() {
    const room = this._data.rooms.find(r => r.id === this._sel);
    const el = this._wrap.querySelector(".rc-panel");
    if (!room) { el.innerHTML = this._overview(); return; }
    const live = this._data.live[room.id] || {};
    const reading = live.temp == null ? "No sensor reading" : live.temp.toFixed(1) + "°F";
    if (!room.unit) {
      el.innerHTML = `<h3>${room.name}</h3><p class="rc-sub">${reading} · monitor only (no heating/cooling unit)</p>`;
      return;
    }
    const step = (act, val) => `<div class="rc-step"><button data-act="${act}" data-d="-1" aria-label="Lower">−</button>
      <span>${val}°</span><button data-act="${act}" data-d="1" aria-label="Raise">+</button></div>`;
    const holding = room.hold_temp != null;
    const holdVal = this._holdTemp ?? room.hold_temp ?? live.target;
    const today = (room.schedule || []).filter(e => e.days.includes(todayIdx()))
      .sort((a, b) => toMin(a.start) - toMin(b.start))
      .map(e => `${fmtTime(toMin(e.start))} → ${e.temp}°`).join(" · ");
    el.innerHTML = `<h3>${room.name}</h3>
      <p class="rc-sub">${reading} · target ${live.target}° · unit ${live.mode}${live.reason ? " (" + live.reason + ")" : ""}</p>
      <div class="rc-row"><span class="rc-lbl">Default temp</span>${step("def", room.default_temp)}</div>
      <div class="rc-row"><span class="rc-lbl">Priority</span>
        <div class="rc-chips">${[1, 2, 3, 4, 5].map(n => `<button data-act="prio" data-v="${n}" class="${n === room.priority ? "on" : ""}">${n}</button>`).join("")}</div></div>
      <div class="rc-sect"><h4>Hold${holding ? " · active" : ""}</h4>
        <div class="rc-row"><span class="rc-lbl">Hold at</span>${step("hold-step", holdVal)}</div>
        <div class="rc-row"><span class="rc-lbl">For</span><div class="rc-chips">
          ${[1, 2, 4, 8].map(h => `<button data-act="hold" data-h="${h}">${h} h</button>`).join("")}
          ${holding ? `<button data-act="hold-clear">Clear</button>` : ""}</div></div></div>
      <div class="rc-sect"><h4>Schedule</h4>
        <p class="rc-sub" style="margin:8px 0">${today || "No schedule for today — using default temp."}</p>
        <button data-act="sched" style="width:100%">Edit schedule…</button></div>`;
  }

  /* ---------- system overview (nothing selected) ---------- */
  // Home panel (nothing selected): just what each heat pump is doing.
  _overview() {
    const d = this._data, units = d.units || [], ul = d.units_live || {};
    const label = { heat: "Heating", cool: "Cooling", off: "Off" };
    const tiles = units.map(u => {
      const x = ul[u.id]; if (!x) return "";
      return `<div class="rc-tile ${x.mode}"><div class="nm">${(u.name || u.id).replace(/ (air handler|minisplit)$/i, "")}</div>
        <div class="big">${x.mode === "heat" ? "▲ " : x.mode === "cool" ? "▼ " : ""}${label[x.mode] || x.mode}</div></div>`;
    }).join("");
    const db = (d.settings_effective || {}).deadband ?? 1;
    const top = units.flatMap(u => ((ul[u.id] || {}).rooms || []).map((r, _, all) => ({ ...r, drv: all.length > 1 && ul[u.id].driver === r.id })))
      .sort((a, b) => b.priority - a.priority || a.name.localeCompare(b.name)).slice(0, 3);
    const pr = top.map(r => {
      const c = r.delta == null ? "" : r.delta > db ? "hot" : r.delta < -db ? "cold" : "ok";
      return `<button class="rc-pr" data-room="${r.id}"><span class="nm">${r.name}${r.drv ? `<span class="star" title="Driving its unit">★</span>` : ""}
        <span class="rc-pips" title="Priority ${r.priority}">${[1, 2, 3, 4, 5].map(n => `<b class="${n <= r.priority ? "on" : ""}"></b>`).join("")}</span></span>
        <span class="rc-d ${c}">${r.temp == null ? "—" : r.temp.toFixed(1) + "°"} <small>→ ${r.target}°</small></span></button>`;
    }).join("");
    return `<h3>Heat pumps</h3><div class="rc-units">${tiles}</div>
      ${pr ? `<div class="rc-sect"><h4>Highest priority</h4>${pr}</div>` : ""}
      <p class="rc-sub" style="margin:12px 0">Tap a room to adjust it.</p>
      <button data-act="diag" style="width:100%">Diagnostics</button>`;
  }

  _openDiag() {
    this._closeSchedule();
    const m = this._modal = document.createElement("div");
    m.className = "rc-modal";
    m.innerHTML = `<style>${CSS}</style><div class="rc-sheet" role="dialog" aria-label="Diagnostics">
      <header><h3>Diagnostics</h3><button data-x="close" aria-label="Close">✕</button></header>
      <div class="rc-body" data-slot="diag"></div>
      <footer><button data-x="close" class="on">Close</button></footer></div>`;
    document.body.append(m);
    this._diag = true;
    m.addEventListener("click", e => {
      const b = e.target.closest("[data-x],[data-room]");
      if (!b && !e.target.classList.contains("rc-modal")) return;
      const id = b && b.dataset.room;
      this._closeSchedule();
      if (id) {  // a room row: jump to that room
        this._sel = id; this._holdTemp = null;
        const r = this._data.rooms.find(x => x.id === id); if (r) this._floor = r.floor || 1;
      }
      this._render();
    });
    this._onKey = e => { if (e.key === "Escape") { this._closeSchedule(); this._render(); } };
    document.addEventListener("keydown", this._onKey);
    this._diagUpdate();
  }
  _diagUpdate() {
    const el = this._modal && this._modal.querySelector('[data-slot="diag"]');
    if (el) el.innerHTML = this._diagHtml();
  }

  _diagHtml() {
    const d = this._data, units = d.units || [], ul = d.units_live || {}, db = (d.settings_effective || {}).deadband ?? 1;
    const RANGE = 3;  // gauges span +/- 3 F around target
    const cls = dl => dl == null ? "" : dl > db ? "hot" : dl < -db ? "cold" : "ok";
    const gauge = v => `<div class="rc-g"><span class="band" style="width:${db / RANGE * 100}%"></span><span class="tick"></span>
      ${v == null ? "" : `<i class="${cls(v)}" style="left:${50 + 50 * clamp(v / RANGE, -1, 1)}%"></i>`}</div>`;
    const running = units.filter(u => (ul[u.id] || {}).mode !== "off").length;
    const short = u => (u.name || u.id).replace(/ (air handler|minisplit)$/i, "");
    const tiles = units.map(u => {
      const x = ul[u.id]; if (!x) return "";
      const big = x.mode === "off" ? "OFF" : `${x.mode === "heat" ? "▲" : "▼"} ${x.setpoint ?? "—"}°`;
      return `<div class="rc-tile ${x.mode}"><div class="nm">${short(u)}</div><div class="big">${big}</div>${gauge(x.error)}
        <div class="why">${x.reason || ""}</div></div>`;
    }).join("");
    const rows = units.flatMap(u => ((ul[u.id] || {}).rooms || []).map(r => ({ ...r, drv: ul[u.id].rooms.length > 1 && ul[u.id].driver === r.id })))
      .sort((a, b) => b.priority - a.priority || a.name.localeCompare(b.name))
      .map(r => `<button class="rc-rm" data-room="${r.id}"><span class="nm">${r.name}${r.drv ? `<span class="star" title="Driving its unit">★</span>` : ""}
          <span class="rc-pips" title="Priority ${r.priority}">${[1, 2, 3, 4, 5].map(n => `<b class="${n <= r.priority ? "on" : ""}"></b>`).join("")}</span></span>
        ${gauge(r.delta)}<span class="t">${r.temp == null ? "—" : r.temp.toFixed(1) + "°"}</span></button>`).join("");
    const monRooms = d.rooms.filter(r => !r.unit);
    const mon = monRooms.map(r => { const l = (d.live || {})[r.id] || {};
      const dup = monRooms.some(o => o !== r && o.name === r.name);  // disambiguate e.g. two baths
      return `<button data-room="${r.id}">${r.name}${dup ? ` · F${r.floor || 1}` : ""}<b>${l.temp == null ? "—" : l.temp.toFixed(0) + "°"}</b></button>`; }).join("");
    return `<p class="rc-sub" style="margin:0">${running} of ${units.length} units running</p>
      <div class="rc-units">${tiles}</div>
      <div class="rc-sect"><h4>Rooms · by priority</h4>
        <div class="rc-rows">${rows}</div>
        <div class="rc-legend"><span>▼ colder than target</span><span>on target</span><span>warmer ▲</span></div></div>
      ${mon ? `<div class="rc-sect"><h4>Monitor only</h4><div class="rc-chips2">${mon}</div></div>` : ""}`;
  }

  /* ---------- schedule editor (modal) ---------- */
  _openSchedule(room) {
    this._closeSchedule();
    const days = DAYS.map((_, d) => (room.schedule || []).filter(e => e.days.includes(d))
      .map(e => ({ m: toMin(e.start), t: e.temp })).sort((a, b) => a.m - b.m));
    // Every day gets an explicit 12:00 AM point (fixed in time) so the ends of the curve don't depend on other points.
    // Seeded with the temperature already in force, so existing behavior is unchanged. A day's end is the next day's start.
    const starts = days.map((_, d) => RoomMapCard._carryFor(days, d, room.default_temp));
    days.forEach((pts, d) => { if (!pts.length || pts[0].m !== 0) pts.unshift({ m: 0, t: starts[d] }); });
    this._ed = { room, days, day: todayIdx(), sel: null, selEnd: false, drag: false };
    const m = this._modal = document.createElement("div");
    m.className = "rc-modal";
    m.innerHTML = `<style>${CSS}</style><div class="rc-sheet" role="dialog" aria-label="Schedule">
      <header><h3>Schedule · ${room.name}</h3><button data-x="cancel" aria-label="Close">✕</button></header>
      <div class="rc-body">
        <div class="rc-seg" data-slot="days"></div>
        <svg class="rc-chart" viewBox="0 0 640 320"></svg>
        <div data-slot="readout"></div>
        <div class="rc-sect"><h4>Copy this day to</h4><div class="rc-chips" style="margin:8px 0">
          <button data-x="copy" data-to="weekdays">Weekdays</button><button data-x="copy" data-to="weekends">Weekends</button>
          <button data-x="copy" data-to="all">All days</button></div></div>
      </div>
      <footer><button data-x="cancel">Cancel</button><button data-x="save" class="on">Save</button></footer></div>`;
    document.body.append(m);
    m.addEventListener("click", e => this._edClick(e));
    const svg = m.querySelector("svg");
    svg.addEventListener("pointerdown", e => this._edDown(e, svg));
    svg.addEventListener("pointermove", e => this._edMove(e, svg));
    svg.addEventListener("pointerup", () => { this._ed.drag = false; });
    svg.addEventListener("pointercancel", () => { this._ed.drag = false; });
    this._onKey = e => { if (e.key === "Escape") this._closeSchedule(); };
    document.addEventListener("keydown", this._onKey);
    this._edRender();
  }
  _closeSchedule() {
    if (this._modal) { this._modal.remove(); this._modal = null; }
    if (this._onKey) { document.removeEventListener("keydown", this._onKey); this._onKey = null; }
    this._ed = null; this._diag = false;
  }

  // Temperature in force at the start of the edited day: last point of the nearest earlier day that has any.
  static _carryFor(days, day, def) {
    for (let back = 1; back <= 7; back++) {
      const p = days[(day - back + 7) % 7];
      if (p.length) return p[p.length - 1].t;
    }
    return def;
  }
  _carry() { return RoomMapCard._carryFor(this._ed.days, this._ed.day, this._ed.room.default_temp); }

  _edRender() {
    const ed = this._ed, m = this._modal; if (!ed) return;
    const W = 640, H = 320, L = 46, R = 14, T = 14, B = 30;
    const x = mn => L + (W - L - R) * mn / 1440, y = t => T + (H - T - B) * (1 - (t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN));
    const pts = ed.days[ed.day], endP = ed.days[(ed.day + 1) % 7][0];
    let grid = "";
    for (let t = TEMP_MIN; t <= TEMP_MAX; t += 5)
      grid += `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="currentColor" stroke-opacity=".12"/>
               <text x="${L - 8}" y="${y(t) + 4}" text-anchor="end" font-size="12" fill="currentColor" fill-opacity=".6">${t}°</text>`;
    ["12a", "3a", "6a", "9a", "12p", "3p", "6p", "9p", "12a"].forEach((l, i) => {
      grid += `<line y1="${T}" y2="${H - B}" x1="${x(i * 180)}" x2="${x(i * 180)}" stroke="currentColor" stroke-opacity=".12"/>
               <text x="${x(i * 180)}" y="${H - 10}" text-anchor="middle" font-size="12" fill="currentColor" fill-opacity=".6">${l}</text>`;
    });
    let line = `M${x(0)} ${y(pts[0].t)}`;
    pts.slice(1).forEach(p => { line += `H${x(p.m)}V${y(p.t)}`; });
    line += `H${x(1440)}V${y(endP.t)}`;  // step to tomorrow's start temp at midnight
    const area = `${line}V${y(TEMP_MIN)}H${x(0)}Z`;
    // Start/end handles are squares (vertical-only); interior points are circles.
    const sq = (attr, px, py, on) => `<g ${attr}><circle cx="${px}" cy="${py}" r="26" fill="transparent"/>
        <rect x="${px - (on ? 12 : 10)}" y="${py - (on ? 12 : 10)}" width="${on ? 24 : 20}" height="${on ? 24 : 20}" rx="5"
              fill="${on ? "#3de0a5" : "#3fa9ff"}" stroke="#fff" stroke-width="${on ? 3 : 2}"/></g>`;
    const dots = pts.map((p, i) => {
      const on = !ed.selEnd && p === ed.sel;
      if (p.m === 0) return sq(`data-i="${i}"`, x(0), y(p.t), on);
      return `<g data-i="${i}"><circle cx="${x(p.m)}" cy="${y(p.t)}" r="26" fill="transparent"/>
        <circle cx="${x(p.m)}" cy="${y(p.t)}" r="${on ? 13 : 10}" fill="${on ? "#3de0a5" : "#3fa9ff"}" stroke="#fff" stroke-width="${on ? 3 : 2}"/></g>`;
    }).join("") + sq(`data-end="1"`, x(1440), y(endP.t), ed.selEnd);
    let tip = "";
    if (ed.sel) {
      const tx = clamp(x(ed.selEnd ? 1440 : ed.sel.m), L + 52, W - R - 52), ty = y(ed.sel.t) - 22 < T + 8 ? y(ed.sel.t) + 34 : y(ed.sel.t) - 22;
      tip = `<text x="${tx}" y="${ty}" text-anchor="middle" font-size="15" font-weight="600" fill="currentColor">${ed.selEnd ? "End of day" : fmtTime(ed.sel.m)} · ${ed.sel.t}°</text>`;
    }
    m.querySelector("svg").innerHTML = `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent" data-plot="1"/>
      <g pointer-events="none">${grid}<path d="${area}" fill="#3fa9ff" fill-opacity=".15"/>
      <path d="${line}" fill="none" stroke="#3fa9ff" stroke-width="3" stroke-linejoin="round"/></g>${dots}<g pointer-events="none">${tip}</g>`;
    m.querySelector('[data-slot="days"]').innerHTML = DAYS.map((n, d) =>
      `<button data-x="day" data-d="${d}" class="${d === ed.day ? "on" : ""}">${n}</button>`).join("");
    const sel = ed.sel, fixed = sel && (ed.selEnd || sel.m === 0), dis = fixed ? "disabled" : "";
    m.querySelector('[data-slot="readout"]').innerHTML = sel
      ? `<div class="rc-row"><div class="rc-step"><button data-x="t" data-d="-1">−</button><span>${sel.t}°</span><button data-x="t" data-d="1">+</button></div>
           <div class="rc-step"><button data-x="m" data-d="-15" ${dis}>−15m</button><span style="min-width:5.5em">${ed.selEnd ? "End of day" : fmtTime(sel.m)}</span><button data-x="m" data-d="15" ${dis}>+15m</button></div>
           <button data-x="del" ${dis}>Delete</button></div>
         ${fixed ? `<p class="rc-sub" style="margin:0">${ed.selEnd ? `End of day is the start of ${DAYS[(ed.day + 1) % 7]}: only its temperature can change.` : "Start of day: only its temperature can change."}</p>` : ""}`
      : `<p class="rc-sub" style="margin:8px 0">Tap the chart to add a point · drag a point to change its time and temperature.
           The temperature holds until the next point${pts.length ? "" : " (no points yet: using the default temp)"}.</p>`;
  }

  _svgPt(e, svg) {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    return p.matrixTransform(svg.getScreenCTM().inverse());
  }
  _fromChart(p) {
    const W = 640, H = 320, L = 46, R = 14, T = 14, B = 30;
    const m = clamp(Math.round((p.x - L) / (W - L - R) * 1440 / 15) * 15, 0, 1425);
    const t = clamp(Math.round(TEMP_MIN + (1 - (p.y - T) / (H - T - B)) * (TEMP_MAX - TEMP_MIN)), TEMP_MIN, TEMP_MAX);
    return { m, t };
  }
  _edDown(e, svg) {
    const ed = this._ed, pts = ed.days[ed.day];
    const g = e.target.closest("[data-i],[data-end]");
    if (g && g.dataset.end) { ed.sel = ed.days[(ed.day + 1) % 7][0]; ed.selEnd = true; }
    else if (g) { ed.sel = pts[+g.dataset.i]; ed.selEnd = false; }
    else if (e.target.closest("[data-plot]")) {
      const np = this._fromChart(this._svgPt(e, svg));
      ed.selEnd = false;
      if (pts.some(p => p.m === np.m)) { ed.sel = pts.find(p => p.m === np.m); }
      else { pts.push(np); pts.sort((a, b) => a.m - b.m); ed.sel = np; }
    } else return;
    ed.drag = true;
    try { svg.setPointerCapture(e.pointerId); } catch (_) { /* synthetic events in tests */ }
    this._edRender();
  }
  _edMove(e, svg) {
    const ed = this._ed;
    if (!ed.drag || !ed.sel) return;
    const np = this._fromChart(this._svgPt(e, svg)), pts = ed.days[ed.day];
    const fixed = ed.selEnd || ed.sel.m === 0;  // start/end handles only move vertically
    if (!fixed && !pts.some(p => p !== ed.sel && p.m === np.m)) ed.sel.m = np.m;  // never stack two points on one minute
    ed.sel.t = np.t;
    pts.sort((a, b) => a.m - b.m);
    this._edRender();
  }
  _edClick(e) {
    const ed = this._ed;
    if (e.target.classList.contains("rc-modal")) { this._closeSchedule(); return; }
    const b = e.target.closest("[data-x]"); if (!b) return;
    const pts = ed.days[ed.day], d = +b.dataset.d;
    switch (b.dataset.x) {
      case "cancel": this._closeSchedule(); return;
      case "save": {
        const sched = this._toSchedule(ed.days);
        this._closeSchedule();
        this._set({ schedule: sched });
        return;
      }
      case "day": ed.day = d; ed.sel = null; ed.selEnd = false; break;
      case "t": ed.sel.t = clamp(ed.sel.t + d, TEMP_MIN, TEMP_MAX); break;
      case "m": {
        if (ed.selEnd || ed.sel.m === 0) break;
        const nm = clamp(ed.sel.m + d, 0, 1425);
        if (!pts.some(p => p !== ed.sel && p.m === nm)) ed.sel.m = nm;
        pts.sort((a, b) => a.m - b.m); break;
      }
      case "del":
        if (ed.selEnd || ed.sel.m === 0) break;
        pts.splice(pts.indexOf(ed.sel), 1); ed.sel = null; break;
      case "copy": {
        const to = { weekdays: [0, 1, 2, 3, 4], weekends: [5, 6], all: [0, 1, 2, 3, 4, 5, 6] }[b.dataset.to];
        to.filter(t => t !== ed.day).forEach(t => { ed.days[t] = pts.map(p => ({ ...p })); });
        ed.sel = null; ed.selEnd = false;  // the end handle may have just been replaced
        break;
      }
    }
    this._edRender();
  }

  // Per-day point lists -> schedule entries, merging days that share an identical (time, temp) point.
  _toSchedule(days) {
    const by = new Map();
    days.forEach((pts, d) => pts.forEach(p => {
      const key = `${p.m}|${p.t}`;
      if (!by.has(key)) by.set(key, { days: [], start: toHHMM(p.m), temp: p.t });
      by.get(key).days.push(d);
    }));
    return [...by.values()].sort((a, b) => toMin(a.start) - toMin(b.start));
  }
}
// Any render/click handler that throws is caught and reported rather than leaving the card blank.
["_render", "_click", "_panel", "_edRender", "_diagUpdate"].forEach(m => {
  const f = RoomMapCard.prototype[m];
  if (typeof f === "function") RoomMapCard.prototype[m] = function (...a) { try { return f.apply(this, a); } catch (e) { this._fail(m, e); } };
});
customElements.define("room-map-card", RoomMapCard);
window.customCards = (window.customCards || []).concat([{ type: "room-map-card", name: "Room map card" }]);
