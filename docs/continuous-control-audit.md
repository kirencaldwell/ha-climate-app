# Audit: fixed-threshold switching logic, and continuous alternatives (2026-10-18)

Parked for later. Written after the Oct 16 and Oct 17 nights' heat/cool limit cycles and the Oct 18 midday Bedroom 3
overheating.

## Where the fixed thresholds are

- **Room comfort cost** (feeds the heat/cool vote): zero inside a band 1.5 °F below to 1.0 °F above target, quadratic
  outside it (warm side 2×); hard floor 63 / ceiling 72 °F (weight 20); a room counts as the worse of now and a 45-min
  trend projection (capped at 3 °F); heating aims 0.5 °F low.
- **House mode**: leave "off" at demand ≥ 1.0; switch only if the other side beats the current one by 1.5× + 0.5, for
  10 min of confirmation, after ≥ 20 min in the mode; "nothing needed" below 0.05. Exclusions: rooms reading their own
  unit's sensor; rooms coasting after their unit stops (20 min, ≤ 2× the run, ends when the room moves 1.5 °F). A
  unit's rooms net their demand together.
- **Outer loop**: stop when predicted 0.5 °F past target; restart 0.6 °F short; stop command 2 °F behind the room; at
  least one 0.5 °C step while clearly short; push caps (2 °F default, 4 air handler, 1 Bedroom 3); integral cap 2 °F;
  no trend-based restart while coasting; setpoint dead zone 0.7 °F, at most every 5 min.
- **Fan**: quiet / low / medium by thirds of the push; auto when 2 °F short; ≥ 10 min between changes.
- **Other**: experiment guards (ceiling − 0.5, floor); dropout bridging 15 min; schedule glide 1 h.
- **The units' own** (not ours): ~1 °C start/stop hysteresis, minimum output.

## What failed

Every bad night came from the comfort cost and the house-mode vote: a short transient in one room (Bedroom 3's
coast after a run) went through the 45-min projection, became a huge vote (217), cleared the 10-min confirmation and
flipped the house, over and over. Midday sun pushed Bedroom 3's real reading over its band, and the fixed
confirm/min-time rules (and the coast exclusion) delayed the response. The own-sensor and coast exclusions are both
patches for the vote reacting to instantaneous and extrapolated readings.

## Continuous alternatives

| Now | Continuous replacement | Expected effect |
|---|---|---|
| Mode switching: ratio, margin, confirm, minimum time, start threshold | Accumulated cost of staying: integrate (other mode's advantage), leaking when it isn't; switch when it exceeds one switch cost (the MPC's) | Biggest win: transients barely accumulate; a big sustained need switches sooner; one parameter instead of five |
| 45-min trend projection with max() | Low-pass filtered reading, or the model's prediction once trusted | Removes inflated votes (217, 600) |
| Comfort dead zone | Smooth asymmetric cost around the aim (warm side 2×) | Votes ramp instead of jumping from zero; needs the accumulated-cost switch |
| Coast and own-sensor exclusions | Unnecessary with the two above | Fewer special cases |
| Outer loop stop/restart thresholds | One smooth saturating law push(miss) from −2 °F to the cap; the unit's own hysteresis does on/off | Uncertain: removes a relay stacked on the unit's relay, but the unit idles only ~1 °C past setpoint. Sim first. |
| Fan thirds | Keep (discrete actuator); cut points could come from the smooth push | Small |
| 0.5 °C grid, dead zone, interval, floor/ceiling, schedule glide | Keep (hardware, safety) | — |

Inherently discrete: heat vs cool for the whole house (one compressor), and each unit on/off below its minimum
output. The aim is to choose those from cost over time rather than instantaneous thresholds, as the MPC already does.

## Recommended order (when picked up)

1. Accumulated-cost mode switching + filtered readings; then remove the coast and own-sensor exclusions. Validate on the
   Oct 16–18 Bedroom 3 traces and in the sim.
2. Smooth comfort cost (with 1).
3. Sim-only experiment: continuous outer-loop law vs today's thresholds; adopt only if cycling and overshoot don't
   get worse.
