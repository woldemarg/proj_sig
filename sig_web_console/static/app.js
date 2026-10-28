/* SIG UI — the shell: state, vocabulary, the API helper, health, views' wiring (docs/08_interface.md); datasets.js, graph.js, chat.js and geo.js hold the surfaces. */
"use strict";

const S = { datasets: {}, cy: null, hl: null, poll: null, lastReady: null, options: null, view: "graph", sphere: null, sphereSeq: 0, graphSeq: 0, regionsIn: false, regionCount: 0, nodes: [], sort: { k: "weight", asc: false }, chat: [], busy: false };
const STAGE_TEXT = {
  UPLOADED: "У черзі", VALIDATING: "Читання файлу", PROFILING: "Опис колонок", DISCOVERING: "Пошук сегментів",
  VALIDATING_INSIGHTS: "Перевірка інсайтів", EMBEDDING: "Індексування інсайтів", UPDATING_ONTOLOGY: "Пошук тем",
  BUILDING_GRAPH: "Побудова графа", PERSISTING: "Збереження", READY: "Готово",
};
const STAGES = Object.keys(STAGE_TEXT);  // in processing order
const TERMINAL = new Set(["READY", "FAILED", "SKIPPED"]);
const DECLARED = "джерело, надійність, показники, глосарій і документ";  // what a dataset's metadata digest signs: every mismatch notice names them
const ERROR_HELP = {
  no_candidates: ["Немає за чим групувати", "У таблиці немає придатних колонок, щоб групувати рядки (текстових колонок із кількома значеннями).", "Позначте колонки з кодами (ID, прапорці) у «Вважати категоріями» або розбийте числову колонку на смуги."],
  invalid_schema: ["Таблицю не можна проаналізувати", "Потрібні щонайменше одна колонка для групування, одна числова колонка і достатньо рядків.", "Перевірте файл або позначте колонки з кодами як категорії."],
  no_numeric_targets: ["Немає числових колонок", "SIG вимірює зсуви числових колонок і не знайшов жодної придатної.", "Перевірте, чи числа не збережено як текст."],
  no_viable_insights: ["Стійких знахідок немає", "Сегменти перевірено, але жоден не пройшов перевірок тесту медіани, стабільності й сили зсуву.", "Спробуйте більше рядків або інші колонки для групування."],
  unsupported_file: ["Файл не підтримується", "Використайте CSV, TSV або Parquet (у межах ліміту завантаження).", ""],
  unreadable_file: ["Не вдалося прочитати файл", "Розбір не вдався — перевірте роздільник і кодування.", ""],
  invalid_options: ["Параметр не прийнято", "Колонки з параметрів немає або вона не числова, значення має неправильний формат, або зону мапи не вдалося прочитати.", "Перевірте назви колонок і формат значень або комірки чи форму зони."],
  metadata_mismatch: ["Метадані змінено", `Запис датасету більше не збігається з цифровим відбитком своїх метаданих: ${DECLARED} ігноруються, а їхні зміни відхиляються.`, "Завантажте файл ще раз із потрібними метаданими."],
  representation_mismatch: ["Робочий простір несумісний", "Цей робочий простір побудовано з іншою моделлю опису інсайтів або іншими вагами.", "Скиньте робочий простір, щоб змінити модель або ваги."],
  duplicate_patterns: ["Уже завантажено", "Ці інсайти вже є в графі.", ""],
  interrupted: ["Перервано", "Обробку зупинено до завершення; нічого часткового не збережено.", "Завантажте файл ще раз."],
  chat_timeout: ["Відповіді вчасно немає.", "Граф і мовна модель не відповіли в межах ліміту часу.", "Спитайте про один датасет чи меншу зону або вимкніть пояснення, щоб отримати самі свідчення."],
};
const CHAT_TIMEOUT_S = 800;  // a little over nginx's 780 s for /api/chat/
// chat content is Ukrainian; data literals (column names, values) stay exactly as the data holds them.
// The welcome card's questions per demo dataset (by filename); any other geo dataset gets GeoMap.suggestions().
const SUGGESTIONS = {
  "retail_synthetic.csv": [
    "Чому margin нижчий для phones у US?",
    "Що пов'язано з вищим return_rate?",
    "Розкажи про margin для laptops у EU",
    "Де руйнується кореляція між discount і margin?",
  ],
  "california_housing.csv": [
    "Чому median house value нижчий у зонах INLAND?",
    "Де median_house_value вищий?",
    "Що особливого в зонах NEAR OCEAN?",
    "Де найбільше households?",
  ],
};
// The legend (index.html data-key) -> the graph elements of each entry; a sphere trace carries its entry's key as
// `meta` (sphereTraces), and the toggles (.toggle[data-key]) show or hide their entry in both views.
const MARKS = {
  anchor: 'node[kind="Attractor"]', up: 'node[kind="Pattern"][ptype != "covariance"][direction > 0]',
  down: 'node[kind="Pattern"][ptype != "covariance"][direction < 0]', cov: 'node[kind="Pattern"][ptype = "covariance"]',
  lattice: 'edge[type="SPECIALIZES"]', contrast: 'edge[type="CONTRASTS"]', sibling: 'edge[type="SIBLING"]', colocated: 'edge[type="CO_LOCATED"]',
  latent: 'edge[type="RELATED_TO"], edge[type="CO_OCCURS"]', activates: 'edge[type="ACTIVATES"]',
  schema: 'node[kind="Dimension"], node[kind="Metric"], edge[type="HAS_SCOPE"], edge[type="TARGETS"]',
  region: 'node[kind="Region"], edge[type="IN_REGION"], edge[type="PARENT_OF"]',
  temporal: 'node[kind="TemporalNode"], edge[type="PEAKED_AT"], edge[type="DROPPED_AT"]',
};
const NO_VECTORS = ["schema", "region", "temporal"];  // legend entries the sphere cannot draw
// trust (docs/06_graph_and_storage.md §6.2) and scale (docs/07_question_answering.md §7.4): their words are TERMS'
// (reliability_A…, credibility_1…, integrity_ok…, scale_…), their chip classes these
const CREDIBILITY = { 1: "ok", 2: "warn", 5: "bad", 6: "" };
const INTEGRITY = { ok: "ok", mismatch: "bad", unverified: "warn" };
// the legend's entries (index.html data-key) -> the terms whose meaning is their tooltip, then what the drawing's line says
const LEGEND_TERMS = { anchor: ["theme"], cov: ["covariance"], lattice: ["SPECIALIZES", "стрілка веде до ширшого інсайту"],
  latent: ["RELATED_TO", "CO_OCCURS", "дуга — пов'язані теми, пунктир — спільні"], contrast: ["CONTRASTS", "пунктир"], sibling: ["SIBLING", "крапки"],
  colocated: ["CO_LOCATED", "пунктир"], activates: ["ACTIVATES", "пунктир — слабка належність: лише для охоплення, пошук нею не йде"],
  region: ["region", "IN_REGION", "PARENT_OF", "пунктир — у регіоні, стрілка — до ширшого регіону"], schema: ["HAS_SCOPE", "TARGETS"],
  temporal: ["temporal_hub", "PEAKED_AT", "DROPPED_AT", "стрілка — до періоду піку, пунктирна — спаду"],
  seed: ["role_seed"], cross: ["transversal_only"] };
// The terms the console names, from insight_contracts/wording.py (tests/test_wording.py keeps the two equal): key ->
// [name, meaning]; a relation's key is its edge type, "<TYPE> (in)" incoming, its name labelling the other end.
const TERMS = {
  "insight": ["інсайт", "перевірена знахідка: частина даних, у якій метрика або зв'язок двох метрик помітно інші, ніж у всіх даних"],
  "theme": ["тема", "картина, що повторюється в інсайтах з різних частин даних, наприклад «discount вище, margin нижче»"],
  "scope": ["де", "умови, що виділяють частину даних, про яку інсайт"],
  "metric": ["метрика", "числова колонка, зсув якої описує інсайт"],
  "sd": ["sd", "різниця медіан у робастних стандартних відхиленнях: 0,6745 · Δмедіани / MAD, зі знаком, |z| не більше 10"],
  "weight": ["сила свідчення", "від нижньої межі до 1: зважене геометричне середнє сили зсуву, стабільності, обсягу і зміни зв'язків; слабкий бік не компенсується сильним"],
  "support": ["рядки", "скільки рядків даних охоплює інсайт і яку частку всіх рядків"],
  "p_adjusted": ["скоригована p", "результат тесту медіани з поправкою на кількість перевірок; поріг відбору, а не точна ймовірність"],
  "stability": ["стабільність", "1 − min(CV, 0,9), де CV — коефіцієнт варіації суми |z| головних зсувів на 20 бутстреп-вибірках; від 0,1 до 1, 0 коли зсуву немає; ближче до 1 — менше коливання"],
  "drivers": ["супутні чинники", "колонки, чиї значення в цій частині даних розподілені помітно інакше (тест хі-квадрат); вони можуть пояснювати зсув"],
  "covariance": ["зміна зв'язку", "кореляція двох метрик тут інша, ніж у всіх даних"],
  "aliases": ["інші умови", "інші умови, що виділяють ті самі рядки"],
  "reliability": ["надійність джерела", "як завантажувач оцінив джерело: від A (цілком надійне) до E (ненадійне); F — оцінити не можна"],
  "reliability_A": ["цілком надійне", "оцінка A надійності джерела"],
  "reliability_B": ["здебільшого надійне", "оцінка B надійності джерела"],
  "reliability_C": ["досить надійне", "оцінка C надійності джерела"],
  "reliability_D": ["переважно ненадійне", "оцінка D надійності джерела"],
  "reliability_E": ["ненадійне", "оцінка E надійності джерела"],
  "reliability_F": ["надійність не оцінено", "оцінка F: надійність джерела оцінити не можна"],
  "credibility": ["підтвердження", "чи має знахідку інше незалежне джерело; рахується з порівняння джерел, не зі статистики"],
  "credibility_1": ["підтверджено іншим джерелом", "інше джерело має ту саму знахідку в тому самому місці або з тією самою метрикою"],
  "credibility_2": ["не підтверджено", "інше джерело має події в цьому місці або ту саму метрику, але такої знахідки не має"],
  "credibility_5": ["суперечить іншому джерелу", "інше джерело показує протилежний зсув"],
  "credibility_6": ["немає іншого джерела", "жодне інше джерело цього не охоплює, тож перевірити нема чим"],
  "integrity": ["цілісність", "чи збігаються записані числа зі своїм цифровим відбитком"],
  "integrity_ok": ["цілісність підтверджено", "числа збігаються з відбитком, записаним разом із ними"],
  "integrity_mismatch": ["цілісність порушено", "числа змінено після запису; знахідку виключено з відповідей"],
  "integrity_unverified": ["цілісність не перевірено", "записано до появи відбитків"],
  "trend": ["динаміка", "як рівень знахідки змінювався в часі, від першого до останнього виміряного періоду її датасету"],
  "trend_rising": ["зростає", "від першого до останнього виміряного періоду рівень знахідки помітно піднявся"],
  "trend_falling": ["спадає", "від першого до останнього виміряного періоду рівень знахідки помітно опустився"],
  "trend_flat": ["без змін", "за весь проміжок рівень знахідки помітно не змінився, піків і спадів у ньому немає"],
  "trend_spike": ["сплеск", "рівень знахідки повернувся туди, де був, але між тим мав пік або спад"],
  "time_current": ["зараз", "рівень знахідки в останньому виміряному періоді, після згладжування"],
  "time_unmeasured": ["не виміряно", "у жодному періоді не було досить рядків знахідки, щоб виміряти її рівень"],
  "time_faded": ["нижче порога", "в останньому виміряному періоді відхилення знахідки слабше за поріг, з якого воно вважається суттєвим"],
  "time_carrier": ["зосереджено в", "частина знахідки, рядки якої в цьому періоді змістилися найбільше порівняно з рештою знахідки — понад звичну різницю між ними"],
  "time_echo": ["як у ширшої знахідки", "ширша знахідка, що охоплює всі рядки цієї, має той самий пік чи спад у тому самому періоді: це одна подія"],
  "temporal_hub": ["період часу", "день, тиждень, місяць чи квартал, у якому одна чи кілька знахідок мали пік або спад; той самий для всіх датасетів"],
  "PEAKED_AT": ["пік у періоді", "у цьому періоді рівень знахідки був помітно вищим, ніж до і після"],
  "PEAKED_AT (in)": ["пік знахідки", "знахідка, рівень якої в цьому періоді був помітно вищим, ніж до і після"],
  "DROPPED_AT": ["спад у періоді", "у цьому періоді рівень знахідки був помітно нижчим, ніж до і після"],
  "DROPPED_AT (in)": ["спад знахідки", "знахідка, рівень якої в цьому періоді був помітно нижчим, ніж до і після"],
  "scale_regional": ["тримається на ширшому рівні", "той самий зсув є і на ширшому рівні мапи цього завантаження"],
  "scale_scale-dependent": ["змінюється з масштабом", "на ширшому рівні мапи цього завантаження зсув протилежний"],
  "scale_not_found_coarser": ["на ширшому рівні не підтверджено", "на ширшому рівні окремої такої знахідки немає; це не означає, що вона лише місцева"],
  "scale_insufficient": ["один рівень", "ширший рівень мапи не аналізувався"],
  "role_seed": ["відповідає питанню", "знайдено прямо за словами питання"],
  "role_structural": ["пов'язана знахідка", "знайдено через вужчу, ширшу, протилежну знахідку чи знахідку в тому самому місці"],
  "role_transversal": ["через тему", "знайдено через спільну тему: та сама картина в іншій частині даних"],
  "role_vertical_up": ["ширший рівень", "та сама картина на ширшому рівні мапи"],
  "role_vertical_down": ["детальніший рівень", "та сама картина на детальнішому рівні мапи"],
  "transversal_only": ["інший сегмент", "знайдено через тему; жодної спільної умови з питанням"],
  "located": ["лише за місцем", "знайдено лише тому, що лежить у тій самій частині мапи; це не означає, що знахідки пов'язані"],
  "ACTIVATES": ["належить до теми", "інсайт — один із прикладів цієї теми"],
  "ACTIVATES (in)": ["інсайт теми", "інсайт, що є прикладом цієї теми"],
  "RELATED_TO": ["пов'язана тема", "теми, чиї інсайти найбільше схожі одна на одну за змістом; не сусідство на мапі"],
  "RELATED_TO (in)": ["пов'язана тема", "теми, чиї інсайти найбільше схожі одна на одну за змістом; не сусідство на мапі"],
  "CO_OCCURS": ["спільна тема", "теми, що мають спільні інсайти"],
  "CO_OCCURS (in)": ["спільна тема", "теми, що мають спільні інсайти"],
  "SPECIALIZES": ["ширший інсайт", "ширша знахідка, частина якої — ця: ті самі умови без однієї"],
  "SPECIALIZES (in)": ["вужчий інсайт", "вужча частина цієї знахідки: її умови і ще одна"],
  "CONTRASTS": ["протилежний зсув", "частини даних перетинаються, а зсув у них протилежний"],
  "CONTRASTS (in)": ["протилежний зсув", "частини даних перетинаються, а зсув у них протилежний"],
  "SIBLING": ["сусідній варіант", "та сама ширша частина даних, інше значення однієї колонки"],
  "SIBLING (in)": ["сусідній варіант", "та сама ширша частина даних, інше значення однієї колонки"],
  "CO_LOCATED": ["те саме місце", "на мапі в тому самому місці; це не означає, що знахідки пов'язані"],
  "CO_LOCATED (in)": ["те саме місце", "на мапі в тому самому місці; це не означає, що знахідки пов'язані"],
  "IN_REGION": ["у регіоні", "регіон мапи, де лежить інсайт"],
  "IN_REGION (in)": ["інсайт у регіоні", "інсайт, що лежить у цьому регіоні"],
  "PARENT_OF": ["ширший регіон", "регіон ширшого рівня мапи, що містить цей"],
  "PARENT_OF (in)": ["вужчий регіон", "регіон детальнішого рівня всередині цього"],
  "HAS_SCOPE": ["колонка умови", "колонка, за якою виділено частину даних"],
  "HAS_SCOPE (in)": ["інсайт за колонкою", "інсайт, умови якого використовують цю колонку"],
  "TARGETS": ["метрика інсайту", "метрика, зсув якої описує інсайт"],
  "TARGETS (in)": ["інсайт про метрику", "інсайт, що описує зсув цієї метрики"],
  "cell": ["комірка", "шестикутна клітинка мапи; детальніший рівень — менші комірки"],
  "region": ["регіон", "комірка ширшого рівня мапи, що об'єднує детальніші"],
  "zone": ["зона", "виділена вами ділянка мапи, про яку питаєте"],
  "segment": ["сегмент", "частина таблиці, виділена умовами"],
  "points": ["події", "точки з файлу, що потрапили в комірки інсайту"],
  "cells_extent": ["комірки", "комірки мапи, з яких складається інсайт на своєму рівні"],
  "cells_events": ["комірки подій", "комірки цього рівня, що тримають події інсайту, знайденого на іншому рівні"],
  "cells_in_view": ["комірки у вікні", "комірки у вікні мапи, де є події інсайту"],
  "window_lift": ["частота у вікні", "у скільки разів частіше події інсайту трапляються у вікні мапи (комірки з центром у рамці), ніж очікувано за його часткою на всій мапі; згладжено 30 подіями"],
  "place_lift": ["частота в регіоні", "у скільки разів частіше події інсайту трапляються в цьому регіоні, ніж очікувано за часткою регіону на всій мапі"],
  "compactness": ["суцільність", "частка зв'язків сусідства (кільце 1) між комірками інсайту, що лишаються всередині нього: 1 — суцільний блок, 0 — розкидані комірки"],
  "n_eff": ["ефективний обсяг", "ефективний обсяг для тесту медіани: кількість комірок з непорожнім значенням метрики, зменшена за схожістю сусідніх комірок (Moran's I > 0)"],
  "moran_i": ["схожість сусідів", "наскільки сусідні комірки інсайту схожі за метрикою (Moran's I): більше 0 — схожі, 0 і менше — ні"],
  "sac": ["поправка на сусідство", "скільки перевірених знахідок значущі без поправки → з поправкою на схожість сусідніх комірок (просторову автокореляцію)"],
  "mass": ["вага теми", "сума сил свідчення її інсайтів"],
  "evidence_mass": ["вага з відповідністю", "сума сил свідчення її інсайтів, кожна помножена на те, наскільки інсайт відповідає темі"],
  "dispersion": ["розкид", "наскільки інсайти теми відрізняються від неї: 0 — усі однакові"],
  "weight_effect": ["множник зсуву", "множник сили свідчення від величини зсуву, від 0 до 1"],
  "weight_support": ["множник обсягу", "корисність обсягу, поділена на її найбільше можливе значення, від 0 до 1"],
  "weight_emm": ["множник зміни зв'язків", "бонус за зміну кореляцій, від 0,5 до 1: ніколи не обнуляє силу свідчення"],
  "sd_score": ["оцінка зсуву", "сума зсувів головних метрик з поправкою на стабільність"],
  "emm_score": ["оцінка зміни зв'язків", "середньоквадратична зміна рангових кореляцій між метриками, зменшена для малих частин даних"],
  "volume_utility": ["корисність обсягу", "√p · (1 − p) для частки рядків p: найбільша, коли інсайт охоплює третину даних"]
};
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const levelName = (level) => (level === "table" ? "таблиця" : String(level).replace(/^res /, "рівень "));  // the server's level codes, as shown
const word = (key) => (TERMS[key] || [key])[0];  // a term's name; an unknown key reads as itself
// a term as the console names it, its meaning in the tooltip: the one writer of the dictionary's tooltips (docs/08_interface.md §8.5)
const term = (key) => (TERMS[key] ? `<span class="term" title="${esc(TERMS[key][1])}">${esc(TERMS[key][0])}</span>` : esc(key));
const reached = (it) => (it.located ? "located" : `role_${it.role}`);  // how an answer reached an item: by location alone, else its role (docs/07 §7.4)
const cred = (level) => [CREDIBILITY[level] ?? "warn", TERMS[`credibility_${level}`] ? word(`credibility_${level}`) : "не оцінено"];
const RELATION = { overlaps: "ділянки перетинаються", adjacent: "ділянки торкаються", same_region: "той самий регіон, ділянки окремо" };
const LEGEND = { ...MARKS, seed: ".hl-seed", ev: ".hl-evidence", cross: ".hl-cross", path: ".hl-edge, .hl-anchor" };  // + the answer's marks

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (x, d = 2) => (x === null || x === undefined || Number.isNaN(Number(x))) ? "–" : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
const signed = (x, d = 2) => (x > 0 ? "+" : "") + num(x, d);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const human = (s) => String(s || "").replace(/_/g, " ");
const PLURAL = new Intl.PluralRules("uk");
const form = (n, one, few, many) => ({ one, few }[PLURAL.select(n)] ?? many);  // 1 / 2–4 / 5+
const plural = (n, one, few, many) => `${num(n, 0)} ${form(n, one, few, many)}`;
const on = (root, selector, fn) =>  // one click listener on a container for every ``selector`` its renders produce
  root.addEventListener("click", (e) => { const el = e.target.closest(selector); if (el && root.contains(el)) fn(el, e); });
const chip = (cls, text, title = "") => `<span class="chip ${cls}"${title ? ` title="${esc(title)}"` : ""}>${esc(text)}</span>`;
const KEYED = '[role="button"]:not(button), tr[data-id], th[data-k]';  // clickable without being buttons: tabindex 0
const MOTION = matchMedia("(prefers-reduced-motion: reduce)");
const calm = () => MOTION.matches;  // no camera flights or pulses then
const nbRow = (id, text, note = "") => `<div class="nb" data-id="${esc(id)}" tabindex="0" role="button"><span>${esc(text)}</span><span class="muted">${esc(note)}</span></div>`;
const picture = (r) => `${plural(r.patterns, "інсайт", "інсайти", "інсайтів")} · ${plural(r.datasets, "датасет", "датасети", "датасетів")} · ${plural(r.themes, "тема", "теми", "тем")}`;  // what a region holds

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(typeof body.detail === "string" ? body.detail : body.detail ? JSON.stringify(body.detail) : res.statusText), { status: res.status });
  return body;
}

// theme + health
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("sig-theme", t); } catch (e) { /* private mode */ }
  if (S.cy) S.cy.style(graphStyle());
  renderSphere();  // same palette in 3D
}

const CPU_HINT = "Модель опису інсайтів працює на CPU: завантаження повільніші. На Mac запустіть сервіс графа на хості, де він бере GPU (MPS; README, Mac: GPU); з GPU NVIDIA запускайте стек із compose.gpu.yaml.";

async function refreshHealth() {
  try {
    const [h, chat] = await Promise.all([api("/api/health"), api("/api/chat/health").catch(() => ({}))]);
    const llm = chat.llm || {};
    const llmCls = llm.reachable ? (llm.model_available ? "ok" : "warn") : "bad";
    const model = String(llm.model || "LLM").split("/").pop();
    const llmTxt = llm.reachable ? (llm.model_available ? model : `${model}: моделі немає`) : `${model} недоступна`;
    const cpu = h.embedding_device === "cpu";  // amber: on a Mac, Docker sees no GPU (README, Mac: GPU)
    const emb = h.embedding ? `${String(h.embedding.model_id).split("/").pop()} · ${h.embedding_device}` : `модель · ${h.embedding_device}`;
    $("health").innerHTML =
      `<span class="pill ${llmCls}" title="Мовна модель · ${esc(llm.base_url || "")}"><i class="dot"></i>${esc(llmTxt)}</span>` +
      `<span class="pill ${h.neo4j.enabled ? "ok" : ""} opt" title="Дзеркало графа в Neo4j"><i class="dot"></i>${h.neo4j.enabled ? "Neo4j " + esc(h.neo4j.database) : "Neo4j вимкнено"}</span>` +
      `<span class="pill ${cpu ? "warn " : ""}opt" title="${cpu ? CPU_HINT : "Модель опису інсайтів і пристрій"}"><i class="dot"></i>${esc(emb)}</span>`;
    $("llm-label").textContent = `пояснення ${model}`;
    $("reset-btn").hidden = !h.problem;  // a degraded workspace: only a reset makes it usable again
    if (h.problem) setUploadMsg(`Робочим простором користуватися не можна: ${h.problem}`, true);
  } catch (e) { $("health").innerHTML = `<span class="pill bad"><i class="dot"></i>Сервер недоступний</span>`; }
}

function setLegend(show) {
  $("legend").hidden = !show;
  $("legend-btn").setAttribute("aria-pressed", String(show));
  try { localStorage.setItem("sig-legend", show ? "on" : "off"); } catch (e) { /* private mode */ }
  if (S.cy) S.cy.resize();
}

function initResize() {
  /* drag the border of the datasets rail or of the chat; widths are remembered, a double-click resets */
  const app = document.querySelector(".app");
  try { for (const col of ["rail", "chat"]) { const w = localStorage.getItem(`sig-${col}-w`); if (w) app.style.setProperty(`--${col}-w`, w); } } catch (e) { /* private mode */ }
  document.querySelectorAll(".col-resize").forEach((h) => {
    const col = h.dataset.col, other = document.querySelector(col === "rail" ? ".chat" : ".rail");
    const save = () => { try { localStorage.setItem(`sig-${col}-w`, app.style.getPropertyValue(`--${col}-w`)); } catch (e) { /* private mode */ } };
    h.addEventListener("pointerdown", (e) => {
      e.preventDefault(); h.setPointerCapture(e.pointerId); h.classList.add("on"); document.body.classList.add("resizing");
      const box = app.getBoundingClientRect(), max = box.width - other.getBoundingClientRect().width - 380;
      const move = (ev) => {
        const w = col === "rail" ? ev.clientX - box.left : box.right - ev.clientX;
        app.style.setProperty(`--${col}-w`, `${Math.round(Math.min(Math.max(w, col === "rail" ? 220 : 300), max))}px`);
        if (S.cy) S.cy.resize();
      };
      const up = () => {
        h.classList.remove("on"); document.body.classList.remove("resizing");
        h.removeEventListener("pointermove", move); h.removeEventListener("pointerup", up); save();
      };
      h.addEventListener("pointermove", move); h.addEventListener("pointerup", up);
    });
    h.addEventListener("dblclick", () => { app.style.removeProperty(`--${col}-w`); save(); if (S.cy) S.cy.resize(); });
  });
}

// wiring
function clearAnswer() {
  /* the one "clear the answer" action: its highlight in the graph and on the sphere, the map's focus and evidence */
  S.hl = null; clearHighlight(); $("clear-btn").hidden = true; renderSphere(); GeoMap.clear();
}

function init() {
  document.querySelectorAll(".key[data-key]").forEach((el) => {  // the legend's tooltips: the terms' meanings
    const keys = LEGEND_TERMS[el.dataset.key];
    if (keys) el.title = keys.map((k) => (TERMS[k] ? `${word(k)}: ${TERMS[k][1]}` : k)).join("\n");
  });
  $("theme-btn").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  // upload: click or drop
  const drop = $("drop");
  $("metadata").addEventListener("change", () => { $("metadata-name").textContent = ($("metadata").files[0] || {}).name || ""; });  // the native control speaks the browser's language
  $("file").addEventListener("change", () => { const f = $("file").files[0]; $("file-name").textContent = f ? f.name : ""; $("upload-btn").disabled = !f; });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { $("file").files = e.dataTransfer.files; $("file").dispatchEvent(new Event("change")); } });
  const syncTable = () => { $("table").disabled = !$("geo").value.trim(); };  // the base level is one of a geo upload's
  $("geo").addEventListener("input", syncTable);
  window.addEventListener("pageshow", syncTable);  // a value the browser restored fires no input
  $("upload-form").addEventListener("submit", (e) => { e.preventDefault(); upload(); });
  for (const [id, path] of [["demo-btn", "/api/demo"], ["demo-geo-btn", "/api/demo/geo"]]) {
    $(id).addEventListener("click", async () => {  // both wait for the request: a double click queues one demo
      const demos = [$("demo-btn"), $("demo-geo-btn")];
      demos.forEach((b) => { b.disabled = true; });
      setUploadMsg("Готую демо-датасет…");
      try { await api(path, { method: "POST" }); setUploadMsg("Демо в черзі — поступ видно нижче."); refreshBatches(); } catch (err) { setUploadMsg(err.message, true); }
      demos.forEach((b) => { b.disabled = false; });
    });
  }
  $("reset-btn").addEventListener("click", async () => {
    if (!confirm("Скинути робочий простір?\n\nУсі датасети, інсайти й теми буде видалено; дані доведеться завантажити знову.")) return;
    try { await api("/api/reset", { method: "POST" }); setUploadMsg("Робочий простір скинуто."); } catch (err) { setUploadMsg(err.message, true); }
    S.lastReady = null; refreshBatches(); refreshHealth();
  });
  // canvas
  document.querySelectorAll("#view-switch button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  document.querySelectorAll(".toggle[data-key]").forEach((b) => b.addEventListener("click", () => {
    b.classList.toggle("on");
    if (b.dataset.key === "region") return toggleRegions();
    applyVisibility(); highlight(S.hl);
  }));
  $("layout").addEventListener("change", () => { applyVisibility(); runLayout(); });
  $("dataset-filter").addEventListener("change", () => refreshBatches());  // the filter is part of its key
  $("fit-btn").addEventListener("click", () => {
    const gd = S.view === "sphere" && sphereDiv();
    if (gd) Plotly.relayout(gd, { "scene.camera.eye": { x: 1.1, y: 1.1, z: 1.1 }, "scene.camera.center": { x: 0, y: 0, z: 0 } });
    else if (S.cy) S.cy.fit(S.cy.elements(":visible"), 40);
  });
  document.querySelectorAll("#legend .key[data-key]").forEach((k) => {
    if (!k.matches("button")) k.tabIndex = 0;  // a plain key is reached by Tab too: its spotlight on focus
    for (const [on_, off] of [["mouseenter", "mouseleave"], ["focus", "blur"]]) {
      k.addEventListener(on_, () => spotlight(k.dataset.key));
      k.addEventListener(off, () => spotlight(null));
    }
  });
  $("legend-btn").addEventListener("click", () => setLegend($("legend").hidden));
  $("clear-btn").addEventListener("click", clearAnswer);
  $("drawer-close").addEventListener("click", closeDrawer);
  $("table-search").addEventListener("input", renderTable);
  document.querySelectorAll("#ins-table th").forEach((th) => th.addEventListener("click", () => {
    S.sort = { k: th.dataset.k, asc: S.sort.k === th.dataset.k ? !S.sort.asc : false }; renderTable();
  }));
  // what the views render, clicked: bound once on their containers, so a render only writes HTML
  const thread = $("thread");
  on(thread, ".chip-q", (c) => ask(c.textContent));
  on(thread, ".ask-cancel", (el) => turnOf(el).ctl.abort("cancel"));
  on(thread, "a.cite", (a) => reveal(a.dataset.pid));
  on(thread, ".ev", (el) => { const turn = turnOf(el), item = turn.qa.view.evidence.items[Number(el.dataset.i)]; reveal(item.pattern_id, { isolate: { turn, item, el } }); });
  on(thread, ".bot-tabs button", toggleTab);
  on(thread, ".insight-card .nb", (el) => reveal(el.dataset.id, { walk: true }));
  on(thread, ".insight-card .ic-theme", (el) => { inspect(el.dataset.id); GeoMap.openTheme(el.dataset.id); });
  on(thread, ".insight-card .ic-ask", (el) => {  // asked with its place: the cells it is drawn by on the open map, as a zone
    const p = turnOf(el).insight.node.props, cells = GeoMap.placeOf(p.id);
    ask(askAbout(p), cells.length ? { cells } : false);  // never with a zone selected for other questions
  });
  on(thread, ".insight-card .ic-graph", async (el) => {  // a filter on another dataset would hide its node
    const p = turnOf(el).insight.node.props, sel = $("dataset-filter");
    if (sel.value && sel.value !== p.dataset_id) { sel.value = ""; await refreshBatches(); }
    setView("graph"); reveal(p.id);
  });
  on($("batches"), ".ds-del", (el, e) => { e.stopImmediatePropagation(); deleteDataset(el.dataset.del, el.dataset.name); });
  on($("batches"), ".ds.clickable", (el) => { const sel = $("dataset-filter"); sel.value = sel.value === el.dataset.ds ? "" : el.dataset.ds; refreshBatches(); });
  on($("ins-table"), "tbody tr[data-id]", (tr) => inspect(tr.dataset.id));
  on($("inspector"), ".nb", async (el) => {  // the drawer follows the row; on the map an insight is focused there when drawn, as reveal does
    const id = el.dataset.id, onMap = S.view === "map" && id.startsWith("P-");
    if (S.view === "graph" && id.startsWith("R:") && !layerState().region) {  // a region asked for: the Map regions switch goes on
      const sw = document.querySelector('.toggle[data-key="region"]');
      sw.classList.add("on"); await toggleRegions();
      if (S.cy.getElementById(id).empty()) { sw.classList.remove("on"); await toggleRegions(); }  // one this view does not draw: as it was
    }
    inspect(id, onMap && !GeoMap.has(id) ? GeoMap.offMap(id) : undefined);  // after the regions: it selects the node
    if (onMap) return GeoMap.has(id) ? GeoMap.focusPattern(id) : GeoMap.unfocus();
    focusNodes([id]);
  });
  on($("mw-body"), ".mw-th", (b) => { W.fold[b.dataset.id] = !b.parentElement.classList.contains("open"); renderWindow(); });
  on($("mw-body"), ".mw-i", (b) => GeoMap.pick(b.dataset.pid));
  on($("mw-body"), ".mw-all", () => { W.all = !W.all; renderWindow(); });
  // chat
  $("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
  $("question").addEventListener("input", autosize);
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("question").value); } });
  $("new-chat").addEventListener("click", () => { S.chat.forEach((t) => t.ctl && t.ctl.abort("new chat")); S.chat = []; clearAnswer(); renderThread(); });
  document.addEventListener("keydown", (e) => {  // Enter or Space on a clickable row or link acts as a click
    if ((e.key !== "Enter" && e.key !== " ") || e.repeat || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || !e.target.matches(KEYED)) return;
    e.preventDefault(); e.target.click();
  });
  document.addEventListener("keydown", (e) => {  // Esc closes one layer a press: the cell popup, the map's focus, the drawer; a text field keeps its own
    if (e.key !== "Escape" || e.target.matches("textarea, input:not([type]), input[type=text], input[type=search]")) return;
    if (GeoMap.closePopup()) return;
    if (S.view === "map" && GeoMap.focused()) { GeoMap.unfocus(); return; }
    closeDrawer();
  });
  initResize();
  new ResizeObserver(([e]) => $("map-view").style.setProperty("--bottom", `${e.contentRect.height}px`)).observe(document.querySelector(".map-bottom"));  // the map's top panels end above its foot
  try { if (localStorage.getItem("sig-legend") === "off") setLegend(false); } catch (e) { /* private mode */ }
  renderThread(); refreshHealth(); refreshBatches();
  setInterval(refreshHealth, 30000);
  setInterval(() => tick(document), 1000);  // a running stage's and a pending question's time, between renders
}

document.addEventListener("DOMContentLoaded", init);
