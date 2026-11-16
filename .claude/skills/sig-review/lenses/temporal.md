# Lens: temporal relations

Report: `07-temporal.md`. Apply the rules of `../SKILL.md`.

Review the temporal semantics of the target: does the system interpret calendars, intervals, cores, events and the time relations between nodes correctly, and traverse and align them as specified?

`docs/02_discovery.md` §2.10, `docs/06_graph_and_storage.md` §6.1 (the temporal plane), `docs/07_question_answering.md` §7.1–§7.4 and the time rows of `docs/11_reference.md` are the contract. Do not assume a time library, a timezone or an interval algebra until the code or those chapters confirm it. The series estimators (the bin z, the smoother, the event gate, `time_score`) are the `math` lens. Walks that are not time are the `graph` lens. A dropped field is `data`. The screen is `ux`. This lens owns the calendar and the relations.

## What to check

### 1. Calendars and intervals

Time's coordinate system is the dataset's calendar, not a universal clock.

- Is the grid (`day`, `week`, `month`, `quarter`, or `auto` as §2.10 chooses it) the same from the upload through the series, the hubs and the answer?
- Are bins named on the dataset's zone (wall clock; UTC when `tz` is empty) and then read on that calendar, not re-parsed as UTC?
- Are intervals half-open `[start, end)`, including a core's last instant and a period asked as dates?
- Does a reader turn a period's name back into dates when the trajectory already stores `[t_start, t_end]` per bin?
- Are an ISO week, its year and the month that holds it (the Thursday rule) kept as one fact?
- Are a period written in the question, the period bar and `as_of` kept distinct, and does a period in the text win over the bar?

### 2. Temporal relations

Check the relations actually implemented:

- overlap of a core with a period (active in it), the core being the span §2.10 defines;
- an event's bin inside a period (`PEAKED_AT`, `DROPPED_AT`), one edge per pattern, period and kind;
- a core's ends (`STARTS_AT`, `ENDS_AT`);
- the period above (`WITHIN`: day → week → month → quarter), one chain;
- alignment of an event layer to a target's span (`align`).

A core overlaps a period, and an event's bin lies inside it, only as read on that pattern's own calendar. Who is active in a period comes from the trajectories (`DualGraph.members`), never from an edge. `WITHIN` is not walked. Two datasets share a hub because the period is the same calendar name; they do not join tables, and datasets on different grids meet only in the month or quarter above.

### 3. Traversal and alignment

- The temporal plane is not a path. A walk that follows `PEAKED_AT`, `STARTS_AT` or `WITHIN` to reach an insight is a defect unless a chapter says that walk exists.
- Membership of a period (a hub's, or any other) is `members`: core overlap, else an event inside, at any level.
- A question's periods leave the text before tokens are matched, and a token that only looks like a period stays.
- A layer's events are counted in the window the run declared (`align=span` or the layer's own span), and that window is the one later text names.
- A pattern with no series has no trajectory; a dataset whose series file is not the one its record signs has no temporal plane. Neither is filled in from a neighbour.

### 4. Edge cases

Depending on the change:

- ISO week 53, and a week whose Thursday falls in the next month or the next year;
- a period whose bounds cross a clock change;
- empty, partial, thin and unmeasured bins, and a bin without spread;
- an epoch placeholder among real dates;
- a core of one bin (one hub, both ends);
- a period that ends before it starts, or a token that names no date (`2031-W54`);
- the same week on two zones (a zoned week is not the UTC week before it);
- a shared hub whose bounds carry the first dataset's offset;
- a time record with no `series_digest` (the old chain model): no plane, and the chapter says so.

Do not demand handling of cases the chapters exclude (a flash shorter than the upload is not mined as its own insight; month names in words are not read), but check that the exclusion still holds.

### 5. The same interval

A grid, a zone or a half-open bound must not change between two readers of the same trajectory. "Current" is judged from `as_of` (the latest span end when none is given), and the core's last instant is not inside it. Two readers that apply different overlap rules to the same bins are this lens. A field that never arrived, or a call made in the wrong order, is not.

## Expected result

For each finding:

- the temporal relation or operation concerned;
- the calendar, grid and interval rule, if established;
- where the error arises;
- a concrete timeline (bins, a zone, a period asked) on which the result is wrong;
- the correct fix;
- a test or control dates that tell the right result from the wrong one.

Behaviour that depends on an unstated calendar or an undescribed contract is an uncertainty, not a confirmed defect.
