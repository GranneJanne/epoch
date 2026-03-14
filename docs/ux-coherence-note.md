# Epoch UX Coherence Note

## Focus, selection, active run

- Focus is the panel or graph that receives keyboard actions.
- Selection is the currently highlighted item inside the focused source panel.
- On Home, the `Runs` panel is the source of truth for run selection.
- The `Run Details` panel on Home never keeps an independent hidden selection.
- Active runs are a run state, not a focus state. They stay visually distinct even when not selected.

## Home panel roles

- `Runs` is the primary navigation panel.
- `Run Details` reflects the run selected in `Runs`.
- `Alerts` is supporting context.
- `Processes` remains an action surface for process attach, but it is secondary to `Runs` for run navigation.
- `System` is present as a compact support strip rather than a competing primary panel.

## Enter behavior

- Enter always activates the selected item in the focused panel.
- On Home `Runs`, Enter opens the selected run.
- On Home `Run Details`, Enter opens the same run currently selected in `Runs`.
- On Home `Processes`, Enter attaches the selected process.
- Search, rename, tag edit, and delete confirmation keep Enter scoped to the active input flow.

## Live vs browse behavior

- `LIVE` means the Run Detail view is following incoming data.
- `BROWSING` means the user zoomed or moved away from the newest point, so live-follow is paused.
- `SNAPSHOT` means the selected run is historical and not accepting live updates.
- `g` jumps all run-detail viewports back to live.
- `Space` toggles live-follow on and off for the Run Detail surface.

## Multi-run Run Detail behavior

- The selected run is the primary run.
- A second run can be marked from Home `Runs` as an overlay candidate.
- Opening Run Detail with an overlay candidate shows both runs together in the graph area.
- The header and Run Detail context strip identify the primary run, overlay run, and current mode.
- If a metric is missing for both opened runs, the graph shows a deliberate empty state instead of noisy placeholders.

## Graph mode behavior

- `sparkline` stays compact and single-series.
- `line` keeps the existing chart style.
- `dense` uses a heavier chart rendering for better readability under live updates.
- Multi-run overlays render on chart-based graphs so both runs remain visible together.
