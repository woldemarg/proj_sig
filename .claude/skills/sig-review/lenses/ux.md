# Lens: interface focus and flow

Report: `08-ux.md`. Apply the rules of `../SKILL.md`.

Review what a person sees and the path they take. Does the change stay focused, move in one natural order, and keep the quiet look the console already has?

`docs/08_interface.md` is the contract: §8.2 for the screens and the look, §8.3 for one visual language, §8.5 for the words. Whether a number, a place or a period is true is that lens. Whether the request that leaves the page is the one the server answered is the `system` lens. Judge only the surface.

Do not invent a design system. The look is already specified: neutral greys, one blue accent for links, focus, selection and switches, primary buttons in the text colour, icon buttons without a frame, radii of 3–6 px, no gradients or glows. Data colours (themes, shifts, layers, the map) keep their meaning. A control the chapter places behind developer mode (`data-dev`, off by default) is not clutter on the person's screens.

## What to check

### 1. Focus

The person's screens are *Інсайти*, *Мапа* (once geo data is there) and *Час* (once the graph holds time). The graph and the sphere are the developer's.

- Does a person's screen show one job, or a second control, badge or number that repeats one already there?
- Does developer chrome (levels, p, prompts, the legend, raw records) stay behind the mode?
- Does a new colour, ring or marker compete with the data colours or with the answer markers (seed, evidence, path)?
- Is there still one action that clears an answer (`clearAnswer`)?

### 2. Natural flow

Walk the path the chapter gives, and the path the change adds:

add data → watch the batch → read the finding → ask → follow the citation onto the same thing in the view that is open.

- Drawer order stays: what it shows, how strong, how far to trust it, when, where, then links. A new block that jumps ahead of that, or repeats a block above it, is a break.
- The dataset filter, the period stretch and a map zone shown on screen are the ones the question on screen is about. A view that paints a stretch the other views have dropped is a break. Whether the server applied that stretch is not this lens.
- A click lands on the thing it names (drawer, map, time axis, evidence card). An answer for a turn already left paints nothing.
- Empty, running, failed, degraded and hidden each have one obvious next step, in the words the chapter uses. A control that does nothing, and an error with no words, are findings.

### 3. The look

- New UI uses the tokens in `style.css`. One meaning has one label and one marker in every view that shows it (the legend key, where the graph and the sphere both draw it).
- Chrome is Ukrainian; a data literal stays as the data holds it.
- A term's tooltip is that term's meaning (`TERMS`), not a second definition.
- At the widths §8.2 names, panels stack and none covers another. `prefers-reduced-motion` still suppresses animation, and a focus still moves without a flight.
- Keyboard: a control a person can click can be reached and fired as §8.2 says.

### 4. What is not a finding

A different taste, a missing animation, a denser layout the chapter already specifies, or a developer's field visible with the mode on. Report a defect only when the chapter's focus, order, words or look is broken, and name the sentence it breaks.

## Expected result

For each finding:

- the surface and the step in the path;
- what the person sees, and what §8.2, §8.3 or §8.5 says instead;
- the smallest change that restores that (usually one control removed, one order put back, one token reused);
- how to see the fix: the screen, the width and the state (empty, error, narrow, developer mode off).

A preference with no sentence in the chapter behind it is not a finding.
