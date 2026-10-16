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
// The chart editor edits either schedule: what each point's value is, its range and gridlines, and how it's shown.
const SCHED_KINDS = {
  temp: { key: "schedule", field: "temp", title: "Schedule", lo: TEMP_MIN, hi: TEMP_MAX, grid: 5, fmt: v => `${v}°`, def: r => r.default_temp,
          hint: "The temperature holds until the next point" },
  prio: { key: "priority_schedule", field: "priority", title: "Priority schedule", lo: 1, hi: 5, grid: 1, fmt: v => `P${v}`, def: r => r.priority,
          hint: "The priority holds until the next point (5 = matters most)" },
};

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
.rc-ctl { display: inline-block; font-size: .68em; font-weight: 600; letter-spacing: .03em; padding: 2px 7px; border-radius: 99px; margin: 4px 0 2px;
  background: #ffd16622; color: #ffd166; } .rc-ctl.rc { background: #3de0a522; color: #3de0a5; } .rc-ctl.md { background: #b48cff22; color: #b48cff; }
.rc-tbl { width: 100%; border-collapse: collapse; font-size: .82em; margin-top: 6px; }
.rc-tbl th, .rc-tbl td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--divider-color, #ffffff1a); white-space: nowrap; }
.rc-tbl th { color: var(--secondary-text-color, #9aa7b8); font-weight: 500; }
.rc-tbl td.ok { color: #3de0a5; } .rc-tbl td.warn { color: #ffd166; } .rc-tbl td.bad { color: #ff7a45; }
.rc-tbl small { color: var(--secondary-text-color, #9aa7b8); }
.rc-badge { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: .8em; font-weight: 600; }
.rc-badge.ok { background: #3de0a522; color: #3de0a5; } .rc-badge.warn { background: #ffd16622; color: #ffd166; }
.rc-tile button.rc-hand { width: 100%; min-height: 36px; margin-top: 6px; font-size: .75em; padding: 0 6px; }
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

const FLOOR_NAMES = { 1: "Basement", 2: "First Floor", 3: "Second Floor" };

class RoomMapCard extends HTMLElement {
  setConfig(config) {
    // Opens on the First Floor (floor 2); card config `default_floor` overrides.
    this._config = config; this._sel = null; this._floor = config.default_floor || 2; this._data = null;
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
  async _setControl(unitId, control) {
    try {
      this._data = await this._hass.callWS({ type: "room_climate/set_control", unit_id: unitId, control });
      this._render(); this._diagUpdate();
    } catch (e) { this._fail("set_control", e); }
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
  // Card config `floor_names: {1: "...", ...}` overrides these.
  _floorName(f) {
    return ((this._config || {}).floor_names || {})[f] || FLOOR_NAMES[f] || `Floor ${f}`;
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
    if (t.dataset.act === "model") { this._openModel(); return; }
    if (t.dataset.act === "ctl") { this._setControl(t.dataset.u, t.dataset.c); return; }
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
      case "psched": this._openSchedule(room, "prio"); break;
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
      const active = tech && ((d.units_live || {})[r.unit] || {}).control === "room_climate" && (live.mode === "heat" || live.mode === "cool");
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
      `<button data-floor="${f}" class="${f === this._floor ? "on" : ""}">${this._floorName(f)}</button>`).join("")}</div>` : "";
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
    const todayOf = (list, fmt) => (list || []).filter(e => e.days.includes(todayIdx()))
      .sort((a, b) => toMin(a.start) - toMin(b.start))
      .map(e => `${fmtTime(toMin(e.start))} → ${fmt(e)}`).join(" · ");
    const today = todayOf(room.schedule, e => `${e.temp}°`);
    const ptoday = todayOf(room.priority_schedule, e => `P${e.priority}`);
    const pnow = live.priority ?? room.priority;
    el.innerHTML = `<h3>${room.name}</h3>
      <p class="rc-sub">${reading} · target ${live.target}° · ${((this._data.units_live || {})[room.unit] || {}).control === "room_climate"
        ? `Room Climate in control · unit ${live.mode}${live.reason ? " (" + live.reason + ")" : ""}`
        : `<span style="color:#ffd166">thermostat in control</span> · Room Climate would ${live.mode}${live.reason ? " (" + live.reason + ")" : ""}`}</p>
      <div class="rc-row"><span class="rc-lbl">Default temp</span>${step("def", room.default_temp)}</div>
      <div class="rc-row"><span class="rc-lbl">${(room.priority_schedule || []).length ? "Default priority" : "Priority"}</span>
        <div class="rc-chips">${[1, 2, 3, 4, 5].map(n => `<button data-act="prio" data-v="${n}" class="${n === room.priority ? "on" : ""}">${n}</button>`).join("")}</div></div>
      <div class="rc-sect"><h4>Hold${holding ? " · active" : ""}</h4>
        <div class="rc-row"><span class="rc-lbl">Hold at</span>${step("hold-step", holdVal)}</div>
        <div class="rc-row"><span class="rc-lbl">For</span><div class="rc-chips">
          ${[1, 2, 4, 8].map(h => `<button data-act="hold" data-h="${h}">${h} h</button>`).join("")}
          ${holding ? `<button data-act="hold-clear">Clear</button>` : ""}</div></div></div>
      <div class="rc-sect"><h4>Schedule</h4>
        <p class="rc-sub" style="margin:8px 0">${today || "No schedule for today — using default temp."}</p>
        <button data-act="sched" style="width:100%">Edit schedule…</button></div>
      <div class="rc-sect"><h4>Priority schedule · now P${pnow}</h4>
        <p class="rc-sub" style="margin:8px 0">${ptoday || "No priority schedule for today — using the default priority."}</p>
        <button data-act="psched" style="width:100%">Edit priority schedule…</button></div>`;
  }

  // A unit's real state (from HA) next to what the controller wants. Shared by the home panel and diagnostics.
  _unitStatus(x) {
    const d = this._data;
    const a = x.actual;  // what the unit is really doing; x.mode/x.setpoint are what Room Climate wants
    const am = a ? (a.state === "heat_cool" ? "auto" : a.state) : x.mode;
    const cls = am === "heat" ? "heat" : am === "cool" ? "cool" : "off";
    const sp = a && a.setpoint != null ? Math.round(a.setpoint) : null;
    const fmt = (m, t) => m === "off" ? "off" : `${m}${t != null ? ` ${t}°` : ""}`;
    const actual = !a ? "no data" : a.state === "unavailable" ? "unavailable"
      : [fmt(am, sp), a.action && a.action !== "off" && a.action !== "idle" ? a.action : null, a.temp != null ? `reads ${a.temp.toFixed(1)}°` : null].filter(Boolean).join(" · ");
    const big = !a ? "—" : a.state === "unavailable" ? "N/A" : am === "off" ? "OFF" : `${am === "heat" ? "▲" : am === "cool" ? "▼" : "●"} ${sp ?? "—"}°`;
    const wantTxt = fmt(x.mode, x.mode === "off" ? null : x.setpoint != null ? Math.round(x.setpoint) : null);
    const same = a && a.state !== "unavailable" && x.mode === (am === "auto" ? "heat_cool" : am) && (x.mode === "off" || x.setpoint == null || sp === Math.round(x.setpoint));
    const ctl = x.control || "thermostat", ours = ctl !== "thermostat";
    const ready = d.model_ready || { ok: false, why: [] };
    // Who's in control, and buttons to hand the unit over.
    const badge = ctl === "model" ? `<span class="rc-ctl md" title="The model-based controller sets this unit's mode and temperature">MODEL IN CONTROL</span>`
      : ours ? `<span class="rc-ctl rc" title="Room Climate sets this unit's mode and temperature">ROOM CLIMATE IN CONTROL</span>`
      : `<span class="rc-ctl" title="The unit runs on its own thermostat mode and setpoint">THERMOSTAT IN CONTROL</span>`;
    const btn = (id, c, label, extra = "") => `<button class="rc-hand" data-act="ctl" data-u="${id}" data-c="${c}" ${extra}>${label}</button>`;
    const why = ready.ok ? "Hand this unit to the model-based controller" : "Not ready yet: " + (ready.why || []).slice(0, 3).join("; ").replace(/"/g, "'");
    const hand = id => ctl === "thermostat" ? btn(id, "room_climate", "Let Room Climate control")
      : ctl === "model" ? btn(id, "thermostat", "Hand to thermostat") + btn(id, "room_climate", "Back to Room Climate")
      : btn(id, "thermostat", "Hand to thermostat") + btn(id, "model", "Let the model control", `title="${why}" ${ready.ok ? "" : "disabled"}`);
    return { cls, big, actual, wantTxt, same, a, ours, badge, hand, wantLbl: ours ? "Commanding" : "Would set" };
  }

  /* ---------- system overview (nothing selected) ---------- */
  // Home panel (nothing selected): just what each heat pump is doing.
  _overview() {
    const d = this._data, units = d.units || [], ul = d.units_live || {};
    const tiles = units.map(u => {
      const x = ul[u.id]; if (!x) return "";
      const st = this._unitStatus(x);
      const warn = st.ours && st.a && !st.same;
      return `<div class="rc-tile ${st.cls}"><div class="nm">${(u.name || u.id).replace(/ (air handler|minisplit)$/i, "")}</div>
        ${st.badge}<div class="big">${st.big}</div>
        <div class="why"><b>Actual</b> · ${st.actual}</div>
        <div class="why" style="${warn ? "color:#ffd166" : ""}"><b>${st.wantLbl}</b> · ${st.wantTxt}${x.reason ? ` · ${x.reason}` : ""}${warn ? " ⚠" : st.ours && st.a ? " ✓" : ""}</div>
        ${st.hand(u.id)}</div>`;
    }).join("");
    const house = d.house && d.house.mode ? `<p class="rc-sub" style="margin:0 0 6px">House mode: <b>${d.house.mode}</b> · ${d.house.reason || ""}</p>` : "";
    const db = (d.settings_effective || {}).deadband ?? 1;
    const top = units.flatMap(u => ((ul[u.id] || {}).rooms || []).map((r, _, all) => ({ ...r, drv: all.length > 1 && ul[u.id].driver === r.id })))
      .sort((a, b) => b.priority - a.priority || a.name.localeCompare(b.name)).slice(0, 3);
    const pr = top.map(r => {
      const c = r.delta == null ? "" : r.delta > db ? "hot" : r.delta < -db ? "cold" : "ok";
      return `<button class="rc-pr" data-room="${r.id}"><span class="nm">${r.name}${r.drv ? `<span class="star" title="Driving its unit">★</span>` : ""}
        <span class="rc-pips" title="Priority ${r.priority}">${[1, 2, 3, 4, 5].map(n => `<b class="${n <= r.priority ? "on" : ""}"></b>`).join("")}</span></span>
        <span class="rc-d ${c}">${r.temp == null ? "—" : r.temp.toFixed(1) + "°"} <small>→ ${r.target}°</small></span></button>`;
    }).join("");
    return `<h3>Heat pumps</h3>${house}<div class="rc-units">${tiles}</div>
      ${pr ? `<div class="rc-sect"><h4>Highest priority</h4>${pr}</div>` : ""}
      <p class="rc-sub" style="margin:12px 0">Tap a room to adjust it.</p>
      <div class="rc-chips" style="display:flex;gap:8px"><button data-act="diag" style="flex:1">Diagnostics</button>
        <button data-act="model" style="flex:1">Model</button></div>`;
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
      const c = e.target.closest('[data-act="ctl"]');
      if (c) { this._setControl(c.dataset.u, c.dataset.c); return; }
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
    const isOn = u => { const a = (ul[u.id] || {}).actual; return a ? ["heat", "cool", "heat_cool", "auto", "dry", "fan_only"].includes(a.state) : (ul[u.id] || {}).mode !== "off"; };
    const running = units.filter(isOn).length;
    const short = u => (u.name || u.id).replace(/ (air handler|minisplit)$/i, "");
    const tiles = units.map(u => {
      const x = ul[u.id]; if (!x) return "";
      const { cls, big, actual, wantTxt, same, a, ours, badge, hand, wantLbl } = this._unitStatus(x);
      return `<div class="rc-tile ${cls}"><div class="nm">${short(u)}</div>${badge}<div class="big">${big}</div>${gauge(x.error)}
        <div class="why"><b>Actual</b> · ${actual}</div>
        <div class="why" style="${ours && a && !same ? "color:#ffd166" : ""}"><b>${wantLbl}</b> · ${wantTxt}${x.reason ? ` · ${x.reason}` : ""}${ours && a ? (same ? " ✓ in sync" : " ⚠ differs") : ""}</div>
        ${hand(u.id)}</div>`;
    }).join("");
    const rows = units.flatMap(u => ((ul[u.id] || {}).rooms || []).map(r => ({ ...r, drv: ul[u.id].rooms.length > 1 && ul[u.id].driver === r.id })))
      .sort((a, b) => b.priority - a.priority || a.name.localeCompare(b.name))
      .map(r => `<button class="rc-rm" data-room="${r.id}"><span class="nm">${r.name}${r.drv ? `<span class="star" title="Driving its unit">★</span>` : ""}
          <span class="rc-pips" title="Priority ${r.priority}">${[1, 2, 3, 4, 5].map(n => `<b class="${n <= r.priority ? "on" : ""}"></b>`).join("")}</span></span>
        ${gauge(r.delta)}<span class="t">${r.temp == null ? "—" : r.temp.toFixed(1) + "°"}</span></button>`).join("");
    const monRooms = d.rooms.filter(r => !r.unit);
    const mon = monRooms.map(r => { const l = (d.live || {})[r.id] || {};
      const dup = monRooms.some(o => o !== r && o.name === r.name);  // disambiguate e.g. two baths
      return `<button data-room="${r.id}">${r.name}${dup ? ` · ${this._floorName(r.floor || 1)}` : ""}<b>${l.temp == null ? "—" : l.temp.toFixed(0) + "°"}</b></button>`; }).join("");
    const ctlN = units.filter(u => (ul[u.id] || {}).control === "room_climate").length;
    return `<p class="rc-sub" style="margin:0 0 6px">Room Climate controls <b>${ctlN} of ${units.length}</b> units${ctlN < units.length ? " · the rest run on their own thermostats" : ""}</p>
      ${d.house && d.house.mode ? `<p class="rc-sub" style="margin:0 0 4px">House mode: <b>${d.house.mode}</b> · ${d.house.reason || ""}</p>` : ""}
      <p class="rc-sub" style="margin:0">${running} of ${units.length} units running</p>
      <div class="rc-units">${tiles}</div>
      <div class="rc-sect"><h4>Rooms · by priority</h4>
        <div class="rc-rows">${rows}</div>
        <div class="rc-legend"><span>▼ colder than target</span><span>on target</span><span>warmer ▲</span></div></div>
      ${mon ? `<div class="rc-sect"><h4>Monitor only</h4><div class="rc-chips2">${mon}</div></div>` : ""}`;
  }

  /* ---------- model page (modal): the model-based controller in "would do" mode, and how well the model predicts ---------- */
  _openModel() {
    this._closeSchedule();
    const m = this._modal = document.createElement("div");
    m.className = "rc-modal";
    m.innerHTML = `<style>${CSS}</style><div class="rc-sheet" role="dialog" aria-label="Model">
      <header><h3>Model</h3><button data-x="close" aria-label="Close">✕</button></header>
      <div class="rc-body" data-slot="model"><p class="rc-sub">Loading…</p></div>
      <footer><button data-mx="refresh">Refresh</button><button data-x="close" class="on">Close</button></footer></div>`;
    document.body.append(m);
    this._mOpen = true;
    m.addEventListener("click", e => {
      const b = e.target.closest("[data-x],[data-mx]");
      if (!b && !e.target.classList.contains("rc-modal")) return;
      if (b && b.dataset.mx === "refresh") { this._modelLoad(); return; }
      if (b && b.dataset.mx === "fit") { this._modelFit(); return; }
      if (b && b.dataset.mx === "room") { this._mRoom = b.dataset.r; this._modelUpdate(); return; }
      if (b && b.dataset.mx === "unit") { this._mUnit = b.dataset.u; this._modelUpdate(); return; }
      if (b && b.dataset.mx === "range") { this._mHours = +b.dataset.h; this._modelLoad(); return; }
      if (b && b.dataset.mx === "bt") { this._mBt = b.dataset.k; this._modelUpdate(); return; }
      this._closeSchedule(); this._render();
    });
    this._onKey = e => { if (e.key === "Escape") { this._closeSchedule(); this._render(); } };
    document.addEventListener("keydown", this._onKey);
    this._mTimer = setInterval(() => this._modelLoad(), 30000);
    this._modelLoad();
  }
  async _modelLoad() {
    try {
      const [d, h] = await Promise.all([this._hass.callWS({ type: "room_climate/model" }),
        this._hass.callWS({ type: "room_climate/model_history", hours: this._mHours || 24 })]);
      this._mData = d; this._mHist = h; this._modelUpdate();
    } catch (e) { this._fail("model", e); }
  }
  async _modelFit() {
    try {
      await this._hass.callWS({ type: "room_climate/fit_model" });
      // A fit takes seconds: check every 2 s until it's done instead of waiting for the 30 s refresh.
      for (let i = 0; i < 90 && this._mOpen; i++) {
        await new Promise(r => setTimeout(r, i ? 2000 : 500));
        await this._modelLoad();
        if (this._mData && this._mData.fit && !this._mData.fit.running) break;
      }
    } catch (e) { this._fail("fit_model", e); }
  }
  _modelUpdate() {
    const el = this._modal && this._mOpen && this._modal.querySelector('[data-slot="model"]');
    if (el && this._mData) el.innerHTML = this._modelHtml(this._mData);
  }

  _modelHtml(d) {
    const m = d.model, meta = (m && m.meta) || {}, sh = d.shadow || {}, plan = sh.plan, fitS = d.fit || {};
    const name = id => ((d.targets || {})[id] || {}).name || (d.today[id] || {}).name || id;
    const f1 = v => v == null ? "—" : (+v).toFixed(1);
    const when = iso => iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : "—";
    // 1. status
    const badge = !m ? `<span class="rc-badge warn">No model yet</span>`
      : meta.usable ? `<span class="rc-badge ok">Model usable</span>`
      : `<span class="rc-badge warn">Still learning: doesn't yet beat "nothing changes"</span>`;
    const status = `<div class="rc-sect"><h4>Status</h4>
      <p style="margin:6px 0">${badge}</p>
      <p class="rc-sub" style="margin:4px 0">${m ? `Fitted ${when(meta.fitted)} on ${meta.rows || 0} minutes of log (${when(meta.from)} – ${when(meta.to)}).`
        : "The first fit runs a few minutes after Home Assistant starts, then every night at 3:40."}
        ${fitS.running ? " <b>Fitting now…</b>" : fitS.seconds != null ? ` Last fit took ${fitS.seconds} s.` : ""}${fitS.error ? ` <span style="color:#ffd166">Last fit: ${fitS.error}</span>` : ""}
        ${sh.error ? ` <span style="color:#ff7a45">Planner error: ${sh.error}</span>` : ""}</p>
      ${d.ready ? (d.ready.ok ? `<p style="margin:6px 0"><span class="rc-badge ok">Ready to control units</span></p>`
        : `<p class="rc-sub" style="margin:6px 0"><b>Not ready to control units yet:</b></p><ul class="rc-sub" style="margin:0 0 8px 18px;padding:0">${d.ready.why.slice(0, 6).map(w => `<li>${w}</li>`).join("")}${d.ready.why.length > 6 ? `<li>…and ${d.ready.why.length - 6} more</li>` : ""}</ul>`) : ""}
      <button data-mx="fit" ${fitS.running ? "disabled" : ""}>Refit now</button></div>`;
    // 2. would do vs today
    const dec = sh.decision || {};
    // A controller "off" is its idle state (nothing to change; nothing is sent), not the unit switched off.
    const md = (mode, sp, ctl) => mode == null ? "—" : mode === "off" ? (ctl ? "no change" : "off") : `${mode} ${sp == null ? "" : Math.round(sp) + "°"}`;
    const unitRows = Object.entries(d.today || {}).map(([u, t]) => `<tr><td>${t.name.replace(/ (air handler|minisplit)$/i, "")}</td>
        <td><b>${md((dec[u] || {}).mode, (dec[u] || {}).setpoint, true)}</b></td><td>${md(t.mode, t.setpoint, true)}${t.control === "room_climate" ? "" : " <small>(thermostat in control)</small>"}</td>
        <td>${md(t.actual_mode, t.actual_setpoint)}</td></tr>`).join("");
    const costs = plan ? Object.entries(plan.by_mode).map(([k, v]) => `${k} ${(+v.cost).toFixed(2)}`).join(" · ") : "";
    const would = `<div class="rc-sect"><h4>Would do vs today</h4>
      ${plan ? `<p class="rc-sub" style="margin:4px 0">Model-based: <b>${sh.house === "off" ? "no change needed" : sh.house}</b>${sh.pending ? ` (confirming ${sh.pending})` : ""} · plan cost ${costs} · ${when(plan.at)}, ${plan.ms} ms.
        Today's controller: <b>${d.house_today === "off" ? "no change needed" : d.house_today}</b>.</p>` : `<p class="rc-sub">No plan yet${m ? " (first one within 5 minutes)" : ": needs a model"}.</p>`}
      <table class="rc-tbl"><tr><th>Unit</th><th>Model would</th><th>Today's controller</th><th>Unit now</th></tr>${unitRows}</table>
      <p class="rc-sub" style="margin:6px 0 0">Comfort leans cool: band ${f1(d.comfort.band_below)}° below to ${f1(d.comfort.band_above)}° above target, running warm costs ${d.comfort.warm_weight}×, heating aims ${d.comfort.aim_offset}° low.</p></div>`;
    // 3. accuracy
    const acc = sh.accuracy || {}, hz = sh.horizons || [30, 60, 120];
    const cell = v => {
      if (!v) return `<td>—</td>`;
      const cls = v.rmse < 0.8 * v.base_rmse ? "ok" : v.rmse < v.base_rmse ? "warn" : "bad";
      return `<td class="${cls}" title="${v.n} predictions · bias ${v.bias > 0 ? "+" : ""}${v.bias}°">${v.rmse.toFixed(2)}° <small>vs ${v.base_rmse.toFixed(2)}°</small></td>`;
    };
    const accRows = Object.keys(acc).sort().map(r => `<tr><td>${name(r)}</td>${hz.map(h => cell(acc[r][String(h)])).join("")}</tr>`).join("");
    const accuracy = `<div class="rc-sect"><h4>Prediction accuracy · last 24 h</h4>
      <p class="rc-sub" style="margin:4px 0">Typical error predicting each room ahead from what the units, weather and sun actually did, vs assuming nothing changes. Green: clearly better; amber: a little; red: worse.</p>
      ${accRows ? `<table class="rc-tbl"><tr><th>Room</th>${hz.map(h => `<th>${h} min</th>`).join("")}</tr>${accRows}</table>` : `<p class="rc-sub">Collecting: the first scores arrive 30 minutes after the model starts tracking.</p>`}</div>`;
    // 4. history charts
    const chart = this._histHtml(this._mHist, name);
    // 5. learned
    let learned = "";
    if (m) {
      const units = Object.keys(m.units || {});
      const rep = meta.report || {}, hold = rep.holdout || {};
      const rows = Object.entries(m.rooms).map(([r, p]) => {
        const top = Object.entries(p.coupling || {}).filter(([, c]) => c > 0.005).sort((a, b) => b[1] - a[1]).slice(0, 2)
          .map(([j, c]) => `${name(j)} ${c.toFixed(2)}`).join(", ") || "—";
        const hc = units.map(u => `<td>${f1((p.heat || {})[u])} / ${f1((p.cool || {})[u])}</td>`).join("");
        const h = hold[r] || {};
        const ok = h.model_rmse_F != null && h.no_change_rmse_F != null && h.model_rmse_F < h.no_change_rmse_F;
        return `<tr><td>${name(r)}</td>${hc}<td>${(p.loss || 0).toFixed(3)}</td><td>${f1(p.solar)}</td><td style="white-space:normal;min-width:9em">${top}</td>
          <td class="${h.model_rmse_F == null ? "" : ok ? "ok" : "bad"}">${h.model_rmse_F == null ? "—" : `${h.model_rmse_F}° vs ${h.no_change_rmse_F}°`}</td></tr>`;
      }).join("");
      const skipped = Object.entries(rep).filter(([, v]) => v && v.skipped).map(([r, v]) => `${name(r)}: ${v.skipped}`).join("; ");
      const fits = (d.fits || []).slice(-7).reverse().map(f => `<tr><td>${when(f.fitted)}</td><td>${f.rows}</td><td class="${f.usable ? "ok" : "warn"}">${f.usable ? "usable" : "learning"}</td></tr>`).join("");
      learned = `<div class="rc-sect"><h4>What the model learned</h4>
        <p class="rc-sub" style="margin:4px 0">°F per hour each unit moves each room at full heat / full cool; loss to outdoors per hour; sun °F/h in full sun; strongest links to other rooms (per hour); and its 2-hour check on held-out data vs "nothing changes".</p>
        <div style="overflow-x:auto"><table class="rc-tbl"><tr><th>Room</th>${units.map(u => `<th>${name(u).replace(/ (air handler|minisplit)$/i, "")}</th>`).join("")}<th>Loss</th><th>Sun</th><th>Links</th><th>2 h check</th></tr>${rows}</table></div>
        ${skipped ? `<p class="rc-sub" style="margin:6px 0 0">Not fitted yet: ${skipped}</p>` : ""}
        ${fits ? `<h4 style="margin-top:12px">Recent fits</h4><table class="rc-tbl"><tr><th>When</th><th>Minutes</th><th></th></tr>${fits}</table>` : ""}</div>`;
    }
    return status + would + accuracy + chart + learned;
  }

  // One time-series chart. series: [{pts: [[ms, v]], color, width, dash, dots, step}]; band: [[ms, lo, hi]];
  // strips: [{label, segs: [[ms0, ms1, color]]}] drawn as thin rows above the plot (e.g. heat/cool modes).
  _tsChart({ t0, t1, now, series = [], band = null, line = null, strips = [], marks = [], fmt = v => v.toFixed(0), zero = false, H = 210 }) {
    const W = 640, L = 46, R = 10, B = 22, SH = 11, T = 8 + strips.length * (SH + 3);
    const vals = [];
    series.forEach(sr => sr.pts.forEach(p => { if (p[0] >= t0 && p[0] <= t1 && p[1] != null) vals.push(p[1]); }));
    if (band) band.forEach(p => { if (p[0] >= t0 && p[0] <= t1) vals.push(p[1], p[2]); });
    if (!vals.length) return `<p class="rc-sub">Nothing to plot yet.</p>`;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (zero) lo = 0;
    const span = Math.max(hi - lo, zero ? 0.5 : 3), pad = span * 0.08;
    lo = zero ? 0 : lo - pad; hi = lo + span + 2 * pad;
    const x = t => L + (W - L - R) * (t - t0) / (t1 - t0), y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
    let g = "";
    const ystep = (hi - lo) / 4;
    for (let k = 0; k <= 4; k++) {
      const v = lo + k * ystep;
      g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="currentColor" stroke-opacity=".08"/>
        <text x="${L - 5}" y="${y(v) + 4}" text-anchor="end" font-size="10" fill="currentColor" fill-opacity=".55">${fmt(v)}</text>`;
    }
    const hrs = (t1 - t0) / 3600e3, tick = hrs <= 9 ? 1 : hrs <= 27 ? 4 : 12;
    const first = Math.ceil(t0 / (tick * 3600e3)) * tick * 3600e3;
    for (let t = first; t <= t1; t += tick * 3600e3) {
      const d = new Date(t), lab = tick >= 12 ? d.toLocaleString([], { weekday: "short", hour: "numeric" }) : d.toLocaleTimeString([], { hour: "numeric" });
      g += `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B}" stroke="currentColor" stroke-opacity=".05"/>
        <text x="${x(t)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity=".55">${lab}</text>`;
    }
    let bandSvg = "";
    if (band && band.length > 1) {
      const top = band.map(p => `${x(p[0]).toFixed(1)},${y(p[2]).toFixed(1)}`), bot = band.slice().reverse().map(p => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`);
      bandSvg = `<polygon points="${top.concat(bot).join(" ")}" fill="#3de0a5" fill-opacity=".09"/>`;
    }
    const path = (sr) => {
      const p = sr.pts.filter(q => q[0] >= t0 && q[0] <= t1 && q[1] != null);
      if (sr.dots) return p.map(([t, v]) => `<circle cx="${x(t).toFixed(1)}" cy="${y(v).toFixed(1)}" r="${sr.r || 2.4}" fill="${sr.color}"/>`).join("");
      if (p.length < 2) return "";
      let d = `M${x(p[0][0]).toFixed(1)} ${y(p[0][1]).toFixed(1)}`;
      for (let i = 1; i < p.length; i++) d += sr.step ? `H${x(p[i][0]).toFixed(1)}V${y(p[i][1]).toFixed(1)}` : `L${x(p[i][0]).toFixed(1)} ${y(p[i][1]).toFixed(1)}`;
      return `<path d="${d}" fill="none" stroke="${sr.color}" stroke-width="${sr.width || 2}" ${sr.dash ? `stroke-dasharray="${sr.dash}"` : ""} stroke-linejoin="round"/>`;
    };
    const strip = strips.map((st, i) => {
      const yy = 6 + i * (SH + 3);
      return `<text x="${L - 5}" y="${yy + SH - 2}" text-anchor="end" font-size="9" fill="currentColor" fill-opacity=".6">${st.label}</text>` +
        st.segs.filter(sg => sg[1] > t0 && sg[0] < t1).map(sg => `<rect x="${x(Math.max(t0, sg[0])).toFixed(1)}" y="${yy}" width="${Math.max(0.5, x(Math.min(t1, sg[1])) - x(Math.max(t0, sg[0]))).toFixed(1)}" height="${SH}" fill="${sg[2]}" fill-opacity=".75"/>`).join("");
    }).join("");
    const nowLine = now >= t0 && now <= t1 ? `<line x1="${x(now)}" x2="${x(now)}" y1="${T}" y2="${H - B}" stroke="currentColor" stroke-opacity=".35"/>
      <text x="${x(now) + 3}" y="${T + 10}" font-size="10" fill="currentColor" fill-opacity=".6">now</text>` : "";
    const hline = line ? `<line x1="${L}" x2="${W - R}" y1="${y(line)}" y2="${y(line)}" stroke="#3de0a5" stroke-opacity=".5" stroke-dasharray="2 4"/>` : "";
    const markSvg = marks.filter(mk => mk.t > t0 && mk.t < t1).map(mk => `<line x1="${x(mk.t)}" x2="${x(mk.t)}" y1="${T}" y2="${H - B}" stroke="#ffd166" stroke-opacity=".55" stroke-dasharray="3 3"/>
      <text x="${x(mk.t) - 4}" y="${T + 10}" text-anchor="end" font-size="9.5" fill="#ffd166" fill-opacity=".85">${mk.left || ""}</text>
      <text x="${x(mk.t) + 4}" y="${T + 10}" font-size="9.5" fill="#ffd166" fill-opacity=".85">${mk.right || ""}</text>`).join("");
    return `<svg class="rc-chart" viewBox="0 0 ${W} ${H}">${strip}${g}${bandSvg}${hline}${markSvg}${nowLine}${series.map(path).join("")}</svg>`;
  }

  _histHtml(h, name) {
    if (!h) return "";
    const ms = iso => Date.parse(iso), now = ms(h.now), t0 = now - h.hours * 3600e3, t1 = now + 2 * 3600e3;
    const S = h.samples || [];
    const MODE = { heat: "#ff7a45", cool: "#3fa9ff" };
    const segs = (get) => {  // [[t0, t1, color]] runs of a mode over the samples
      const out = [];
      S.forEach((smp, i) => {
        const md = get(smp), c = MODE[md] || "#667085", ta = ms(smp.t), tb = i + 1 < S.length ? ms(S[i + 1].t) : ta + 300e3;
        if (out.length && out[out.length - 1][2] === c && out[out.length - 1][1] >= ta - 1) out[out.length - 1][1] = tb;
        else out.push([ta, tb, c]);
      });
      return out;
    };
    const ranges = `<div class="rc-seg">${[[6, "6 h"], [24, "24 h"], [72, "3 days"]].map(([v, l]) =>
      `<button data-mx="range" data-h="${v}" class="${(this._mHours || 24) === v ? "on" : ""}">${l}</button>`).join("")}</div>`;
    // rooms with readings in the window
    const rooms = [...new Set(S.flatMap(x => Object.keys(x.r || {})))].sort();
    if (!rooms.includes(this._mRoom)) this._mRoom = rooms.includes("kitchen") ? "kitchen" : rooms[0];
    const rid = this._mRoom, P = (h.points || {})[rid] || {};
    const roomTabs = `<div class="rc-seg">${rooms.map(r => `<button data-mx="room" data-r="${r}" class="${r === rid ? "on" : ""}">${name(r)}</button>`).join("")}</div>`;
    const act = S.filter(x => x.r[rid]).map(x => [ms(x.t), x.r[rid][0]]);
    const band = S.filter(x => x.r[rid] && x.r[rid][1] != null).map(x => [ms(x.t), x.r[rid][1] - h.band[0], x.r[rid][1] + h.band[1]]);
    const proj = ((h.projection || {}).temps || {})[rid] || [], plan = (h.plan || {})[rid] || [];
    const startPt = act.length ? [act[act.length - 1]] : [];
    const BT = ((h.backtest || {}).points || {})[rid] || {};
    const split = (h.backtest || {}).split;
    const btMode = this._mBt || "60";
    const tempChart = this._tsChart({ t0, t1, now, band, fmt: v => v.toFixed(1) + "°",
      marks: split ? [{ t: ms(split), left: "trained on ←", right: "→ tested on" }] : [],
      series: [
      { pts: act, color: "#e6edf7", width: 2 },
      { pts: (BT[btMode] || []).map(([t, v]) => [ms(t), v]), color: "#ffb454", width: 1.3 },
      { pts: (P["120"] || []).map(([t, v]) => [ms(t), v]), color: "#b48cff", dots: true },
      { pts: (P["60"] || []).map(([t, v]) => [ms(t), v]), color: "#3fa9ff", dots: true },
      { pts: startPt.concat(proj.map(([t, v]) => [ms(t), v])), color: "#9aa7b8", dash: "2 4", width: 2 },
      { pts: startPt.concat(plan.map(([t, v]) => [ms(t), v])), color: "#3de0a5", dash: "6 4", width: 2 }] });
    // prediction error by hour
    const A = (h.accuracy || {})[rid] || {};
    const hourly = k => (A[k] || []).map(([t, r]) => [ms(t) + 1800e3, r]);
    const accChart = this._tsChart({ t0, t1: now, now, zero: true, H: 160, fmt: v => v.toFixed(1) + "°", series: [
      { pts: (A["120"] || []).map(([t, , b]) => [ms(t) + 1800e3, b]), color: "#667085", dash: "4 4" },
      { pts: hourly("60"), color: "#3fa9ff" }, { pts: hourly("120"), color: "#b48cff" },
      { pts: hourly("60"), color: "#3fa9ff", dots: true, r: 2 }, { pts: hourly("120"), color: "#b48cff", dots: true, r: 2 }] });
    // commands per unit
    const units = Object.keys(h.units || {});
    if (!units.includes(this._mUnit)) this._mUnit = units[0];
    const uid = this._mUnit;
    const unitTabs = `<div class="rc-seg">${units.map(u => `<button data-mx="unit" data-u="${u}" class="${u === uid ? "on" : ""}">${(h.units[u] || u).replace(/ (air handler|minisplit)$/i, "")}</button>`).join("")}</div>`;
    const U = i => S.filter(x => x.u[uid] && x.u[uid][i] != null).map(x => [ms(x.t), x.u[uid][i]]);
    const cmdChart = this._tsChart({ t0, t1: now, now, fmt: v => v.toFixed(0) + "°",
      strips: [{ label: "model", segs: segs(x => (x.u[uid] || [])[4]) }, { label: "today", segs: segs(x => (x.u[uid] || [])[2]) },
               { label: "unit", segs: segs(x => (x.u[uid] || [])[0]) }],
      series: [{ pts: U(1), color: "#e6edf7", step: true, width: 2 }, { pts: U(3), color: "#3de0a5", step: true, dash: "5 3" },
               { pts: U(5), color: "#b48cff", step: true, width: 2.5 }] });
    // comfort score
    const C = h.comfort || [];
    const comfortChart = this._tsChart({ t0, t1: now, now, zero: true, H: 170, fmt: v => v.toFixed(1), series: [
      { pts: C.filter(c => c[1] != null).map(c => [ms(c[0]), c[1]]), color: "#e6edf7", width: 2 },
      { pts: C.filter(c => c[2] != null).map(c => [ms(c[0]), c[2]]), color: "#b48cff", width: 2 }] });
    const lg = items => `<div class="rc-legend" style="justify-content:flex-start;gap:14px;flex-wrap:wrap">${items.map(([c, t, k]) => `<span><b style="color:${c}">${k || "—"}</b> ${t}</span>`).join("")}</div>`;
    return `<div class="rc-sect"><h4>History</h4>${ranges}
      <h4 style="margin-top:10px">Temperature: predicted vs actual, and the next 2 hours</h4>${roomTabs}${tempChart}
      <div class="rc-seg" style="margin-top:6px">${[["60", "Backtest 1 h ahead"], ["120", "Backtest 2 h ahead"]].map(([k, l]) =>
        `<button data-mx="bt" data-k="${k}" class="${btMode === k ? "on" : ""}">${l}</button>`).join("")}</div>
      ${lg([["#e6edf7", "actual"], ["#ffb454", `latest model run over the logged past, ${btMode === "60" ? "1 h" : "2 h"} ahead (backtest)`],
            ["#3fa9ff", "model said 1 h before (live)", "●"], ["#b48cff", "said 2 h before (live)", "●"],
            ["#3de0a5", "next 2 h if the model were in charge", "- -"], ["#9aa7b8", "next 2 h if settings stay", "··"], ["#3de0a5", "comfort band", "▒"]])}
      <h4 style="margin-top:14px">Prediction error by hour · ${name(rid)}</h4>${accChart}
      ${lg([["#3fa9ff", "1 h ahead"], ["#b48cff", "2 h ahead"], ["#667085", "'nothing changes', 2 h", "- -"]])}
      <p class="rc-sub" style="margin:2px 0 0">Typical miss (°F) of predictions scored in each hour. Below the grey line = better than assuming nothing changes.</p>
      <h4 style="margin-top:14px">Commands</h4>${unitTabs}${cmdChart}
      ${lg([["#b48cff", "model would set"], ["#3de0a5", "today's controller set", "- -"], ["#e6edf7", "unit actually at"],
            ["#ff7a45", "heat", "■"], ["#3fa9ff", "cool", "■"], ["#667085", "off / no change", "■"]])}
      <h4 style="margin-top:14px">Comfort score (lower is better)</h4>${comfortChart}
      ${lg([["#e6edf7", "actual, over the following 2 h (today's controller)"], ["#b48cff", "model's prediction for its own plan"]])}
      <p class="rc-sub" style="margin:2px 0 0">Priority-weighted misses outside the comfort band, per hour (the cost both controllers minimize). The actual line stops 2 h before now, since those 2 hours aren't over yet.</p></div>`;
  }

  /* ---------- schedule editor (modal) ---------- */
  _openSchedule(room, kind = "temp") {
    this._closeSchedule();
    const k = SCHED_KINDS[kind];
    const days = DAYS.map((_, d) => (room[k.key] || []).filter(e => e.days.includes(d))
      .map(e => ({ m: toMin(e.start), t: e[k.field] })).sort((a, b) => a.m - b.m));
    // Every day gets an explicit 12:00 AM point (fixed in time) so the ends of the curve don't depend on other points.
    // Seeded with the temperature already in force, so existing behavior is unchanged. A day's end is the next day's start.
    const starts = days.map((_, d) => RoomMapCard._carryFor(days, d, k.def(room)));
    days.forEach((pts, d) => { if (!pts.length || pts[0].m !== 0) pts.unshift({ m: 0, t: starts[d] }); });
    this._ed = { room, k, days, day: todayIdx(), sel: null, selEnd: false, drag: false };
    const m = this._modal = document.createElement("div");
    m.className = "rc-modal";
    m.innerHTML = `<style>${CSS}</style><div class="rc-sheet" role="dialog" aria-label="${k.title}">
      <header><h3>${k.title} · ${room.name}</h3><button data-x="cancel" aria-label="Close">✕</button></header>
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
    if (this._mTimer) { clearInterval(this._mTimer); this._mTimer = null; }
    this._mOpen = false;
  }

  // Temperature in force at the start of the edited day: last point of the nearest earlier day that has any.
  static _carryFor(days, day, def) {
    for (let back = 1; back <= 7; back++) {
      const p = days[(day - back + 7) % 7];
      if (p.length) return p[p.length - 1].t;
    }
    return def;
  }
  _carry() { return RoomMapCard._carryFor(this._ed.days, this._ed.day, this._ed.k.def(this._ed.room)); }

  _edRender() {
    const ed = this._ed, m = this._modal; if (!ed) return;
    const k = ed.k, W = 640, H = 320, L = 46, R = 14, T = 14, B = 30;
    const x = mn => L + (W - L - R) * mn / 1440, y = t => T + (H - T - B) * (1 - (t - k.lo) / (k.hi - k.lo));
    const pts = ed.days[ed.day], endP = ed.days[(ed.day + 1) % 7][0];
    let grid = "";
    for (let t = k.lo; t <= k.hi; t += k.grid)
      grid += `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="currentColor" stroke-opacity=".12"/>
               <text x="${L - 8}" y="${y(t) + 4}" text-anchor="end" font-size="12" fill="currentColor" fill-opacity=".6">${k.fmt(t)}</text>`;
    ["12a", "3a", "6a", "9a", "12p", "3p", "6p", "9p", "12a"].forEach((l, i) => {
      grid += `<line y1="${T}" y2="${H - B}" x1="${x(i * 180)}" x2="${x(i * 180)}" stroke="currentColor" stroke-opacity=".12"/>
               <text x="${x(i * 180)}" y="${H - 10}" text-anchor="middle" font-size="12" fill="currentColor" fill-opacity=".6">${l}</text>`;
    });
    let line = `M${x(0)} ${y(pts[0].t)}`;
    pts.slice(1).forEach(p => { line += `H${x(p.m)}V${y(p.t)}`; });
    line += `H${x(1440)}V${y(endP.t)}`;  // step to tomorrow's start temp at midnight
    const area = `${line}V${y(k.lo)}H${x(0)}Z`;
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
      tip = `<text x="${tx}" y="${ty}" text-anchor="middle" font-size="15" font-weight="600" fill="currentColor">${ed.selEnd ? "End of day" : fmtTime(ed.sel.m)} · ${k.fmt(ed.sel.t)}</text>`;
    }
    m.querySelector("svg").innerHTML = `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent" data-plot="1"/>
      <g pointer-events="none">${grid}<path d="${area}" fill="#3fa9ff" fill-opacity=".15"/>
      <path d="${line}" fill="none" stroke="#3fa9ff" stroke-width="3" stroke-linejoin="round"/></g>${dots}<g pointer-events="none">${tip}</g>`;
    m.querySelector('[data-slot="days"]').innerHTML = DAYS.map((n, d) =>
      `<button data-x="day" data-d="${d}" class="${d === ed.day ? "on" : ""}">${n}</button>`).join("");
    const sel = ed.sel, fixed = sel && (ed.selEnd || sel.m === 0), dis = fixed ? "disabled" : "";
    m.querySelector('[data-slot="readout"]').innerHTML = sel
      ? `<div class="rc-row"><div class="rc-step"><button data-x="t" data-d="-1">−</button><span>${k.fmt(sel.t)}</span><button data-x="t" data-d="1">+</button></div>
           <div class="rc-step"><button data-x="m" data-d="-15" ${dis}>−15m</button><span style="min-width:5.5em">${ed.selEnd ? "End of day" : fmtTime(sel.m)}</span><button data-x="m" data-d="15" ${dis}>+15m</button></div>
           <button data-x="del" ${dis}>Delete</button></div>
         ${fixed ? `<p class="rc-sub" style="margin:0">${ed.selEnd ? `End of day is the start of ${DAYS[(ed.day + 1) % 7]}: only its temperature can change.` : "Start of day: only its temperature can change."}</p>` : ""}`
      : `<p class="rc-sub" style="margin:8px 0">Tap the chart to add a point · drag a point to change its time and temperature.
           ${k.hint}.</p>`;
  }

  _svgPt(e, svg) {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    return p.matrixTransform(svg.getScreenCTM().inverse());
  }
  _fromChart(p) {
    const k = this._ed.k, W = 640, H = 320, L = 46, R = 14, T = 14, B = 30;
    const m = clamp(Math.round((p.x - L) / (W - L - R) * 1440 / 15) * 15, 0, 1425);
    const t = clamp(Math.round(k.lo + (1 - (p.y - T) / (H - T - B)) * (k.hi - k.lo)), k.lo, k.hi);
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
        const k = ed.k, sched = this._toSchedule(ed.days, k.field);
        this._closeSchedule();
        this._set({ [k.key]: sched });
        return;
      }
      case "day": ed.day = d; ed.sel = null; ed.selEnd = false; break;
      case "t": ed.sel.t = clamp(ed.sel.t + d, ed.k.lo, ed.k.hi); break;
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

  // Per-day point lists -> schedule entries, merging days that share an identical (time, value) point.
  _toSchedule(days, field = "temp") {
    const by = new Map();
    days.forEach((pts, d) => pts.forEach(p => {
      const key = `${p.m}|${p.t}`;
      if (!by.has(key)) by.set(key, { days: [], start: toHHMM(p.m), [field]: p.t });
      by.get(key).days.push(d);
    }));
    return [...by.values()].sort((a, b) => toMin(a.start) - toMin(b.start));
  }
}
// Any render/click handler that throws is caught and reported rather than leaving the card blank.
["_render", "_click", "_panel", "_edRender", "_diagUpdate", "_modelUpdate"].forEach(m => {
  const f = RoomMapCard.prototype[m];
  if (typeof f === "function") RoomMapCard.prototype[m] = function (...a) { try { return f.apply(this, a); } catch (e) { this._fail(m, e); } };
});
// The load probe may re-import this file if the first load stalled, so registering twice must be harmless.
if (!customElements.get("room-map-card")) {
  customElements.define("room-map-card", RoomMapCard);
  window.customCards = (window.customCards || []).concat([{ type: "room-map-card", name: "Room map card" }]);
}
