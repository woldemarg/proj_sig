"""How SIG talks to people (docs/08_interface.md §8.5): the Ukrainian name and one-line meaning of every term the
console and the chat show, and the readings of an insight, a theme and a region built from them.

Beside ``insight_contracts.text``, not in it: that module's English phrases build the embedded canonical document
(docs/04_representation.md §4.2), so changing them re-embeds every workspace; this one is presentation only. Plain
words for someone who reads the map and the chat, not the graph: no "latent", "attractor", "edge" or "node". The
console keeps a copy of ``TERMS`` (``TERMS`` in ``app.js``) that ``tests/test_wording.py`` holds equal to this one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from insight_contracts.insight import Condition, Insight, PhenomenonThresholds, Shift
from insight_contracts.text import covariance_change, format_value, headline, humanize

# key -> (name, meaning). A relation's key is its edge type, "<TYPE> (in)" for the incoming direction (the drawer's
# neighbour groups and a reversed path step use the same keys); its name labels the node at the other end.
TERMS: dict[str, tuple[str, str]] = {
    "insight": ("інсайт", "перевірена знахідка: частина даних, у якій метрика або зв'язок двох метрик помітно інші, ніж у всіх даних"),
    "theme": ("тема", "картина, що повторюється в інсайтах з різних частин даних, наприклад «discount вище, margin нижче»"),
    "scope": ("де", "умови, що виділяють частину даних, про яку інсайт"),
    "metric": ("метрика", "числова колонка, зсув якої описує інсайт"),
    "sd": ("sd", "різниця медіан у робастних стандартних відхиленнях: 0,6745 · Δмедіани / MAD, зі знаком, не більше 10"),
    "weight": ("сила свідчення", "від 0,05 до 1: сила зсуву, стабільність, обсяг і зміна зв'язків разом; слабкий бік не компенсується сильним"),
    "support": ("рядки", "скільки рядків даних охоплює інсайт і яку частку всіх рядків"),
    "p_adjusted": ("скоригована p", "результат тесту медіани з поправкою на кількість перевірок; поріг відбору, а не точна ймовірність"),
    "stability": (
        "стабільність",
        "стійкість величини зсуву на 20 бутстреп-вибірках: 1 − min(CV, 0,9), від 0,1 до 1, 0 коли SD = 0; ближче до 1 — менше коливання",
    ),
    "drivers": ("супутні чинники", "інші колонки, знайдені під час перевірки, які можуть пояснювати зсув"),
    "covariance": ("зміна зв'язку", "кореляція двох метрик тут інша, ніж у всіх даних"),
    "aliases": ("інші умови", "інші умови, що виділяють ті самі рядки"),
    # trust: three separate things (docs/13_trust.md)
    "reliability": ("надійність джерела", "як завантажувач оцінив джерело: від A (цілком надійне) до E (ненадійне); F — оцінити не можна"),
    "reliability_A": ("цілком надійне", "оцінка A надійності джерела"),
    "reliability_B": ("здебільшого надійне", "оцінка B надійності джерела"),
    "reliability_C": ("досить надійне", "оцінка C надійності джерела"),
    "reliability_D": ("переважно ненадійне", "оцінка D надійності джерела"),
    "reliability_E": ("ненадійне", "оцінка E надійності джерела"),
    "reliability_F": ("надійність не оцінено", "оцінка F: надійність джерела оцінити не можна"),
    "credibility": ("підтвердження", "чи має знахідку інше незалежне джерело; рахується з порівняння джерел, не зі статистики"),
    "credibility_1": ("підтверджено іншим джерелом", "інше джерело має ту саму знахідку в тому самому місці або з тією самою метрикою"),
    "credibility_2": ("не підтверджено", "інше джерело має події в цьому місці або ту саму метрику, але такої знахідки не має"),
    "credibility_5": ("суперечить іншому джерелу", "інше джерело показує протилежний зсув"),
    "credibility_6": ("немає іншого джерела", "жодне інше джерело цього не охоплює, тож перевірити нема чим"),
    "integrity": ("цілісність", "чи збігаються записані числа зі своїм цифровим відбитком"),
    "integrity_ok": ("цілісність підтверджено", "числа збігаються з відбитком, записаним разом із ними"),
    "integrity_mismatch": ("цілісність порушено", "числа змінено після запису; знахідку виключено з відповідей"),
    "integrity_unverified": ("цілісність не перевірено", "записано до появи відбитків"),
    # how a finding relates to coarser levels of its map
    "scale_regional": ("тримається на ширшому рівні", "той самий зсув є і на ширшому рівні мапи цього завантаження"),
    "scale_scale-dependent": ("змінюється з масштабом", "на ширшому рівні мапи цього завантаження зсув протилежний"),
    "scale_not_found_coarser": (
        "на ширшому рівні не підтверджено",
        "на ширшому рівні окремої такої знахідки немає; це не означає, що вона лише місцева",
    ),
    "scale_insufficient": ("один рівень", "ширший рівень мапи не аналізувався"),
    # how an answer's finding was reached
    "role_seed": ("відповідає питанню", "знайдено прямо за словами питання"),
    "role_structural": ("пов'язана знахідка", "знайдено через вужчу, ширшу, протилежну знахідку чи знахідку в тому самому місці"),
    "role_transversal": ("через тему", "знайдено через спільну тему: та сама картина в іншій частині даних"),
    "role_vertical_up": ("ширший рівень", "та сама картина на ширшому рівні мапи"),
    "role_vertical_down": ("детальніший рівень", "та сама картина на детальнішому рівні мапи"),
    "transversal_only": ("інший сегмент", "знайдено через тему; жодної спільної умови з питанням"),
    "located": ("лише за місцем", "знайдено лише тому, що лежить у тій самій частині мапи; це не означає, що знахідки пов'язані"),
    # relations: the name labels the other end
    "ACTIVATES": ("належить до теми", "інсайт — один із прикладів цієї теми"),
    "ACTIVATES (in)": ("інсайт теми", "інсайт, що є прикладом цієї теми"),
    "RELATED_TO": ("пов'язана тема", "теми, чиї інсайти найбільше схожі одна на одну за змістом; не сусідство на мапі"),
    "RELATED_TO (in)": ("пов'язана тема", "теми, чиї інсайти найбільше схожі одна на одну за змістом; не сусідство на мапі"),
    "CO_OCCURS": ("спільна тема", "теми, що мають спільні інсайти"),
    "CO_OCCURS (in)": ("спільна тема", "теми, що мають спільні інсайти"),
    "SPECIALIZES": ("вужчий за", "ця знахідка — вужча частина тієї: її умови і ще одна"),
    "SPECIALIZES (in)": ("вужчий інсайт", "вужча частина цієї знахідки: її умови і ще одна"),
    "CONTRASTS": ("протилежний зсув", "частини даних перетинаються, а зсув у них протилежний"),
    "CONTRASTS (in)": ("протилежний зсув", "частини даних перетинаються, а зсув у них протилежний"),
    "SIBLING": ("сусідній варіант", "та сама ширша частина даних, інше значення однієї колонки"),
    "SIBLING (in)": ("сусідній варіант", "та сама ширша частина даних, інше значення однієї колонки"),
    "CO_LOCATED": ("те саме місце", "на мапі в тому самому місці; це не означає, що знахідки пов'язані"),
    "CO_LOCATED (in)": ("те саме місце", "на мапі в тому самому місці; це не означає, що знахідки пов'язані"),
    "IN_REGION": ("у регіоні", "регіон мапи, де лежить інсайт"),
    "IN_REGION (in)": ("інсайт у регіоні", "інсайт, що лежить у цьому регіоні"),
    "PARENT_OF": ("ширший регіон", "регіон ширшого рівня мапи, що містить цей"),
    "PARENT_OF (in)": ("вужчий регіон", "регіон детальнішого рівня всередині цього"),
    "HAS_SCOPE": ("колонка умови", "колонка, за якою виділено частину даних"),
    "HAS_SCOPE (in)": ("інсайт за колонкою", "інсайт, умови якого використовують цю колонку"),
    "TARGETS": ("метрика інсайту", "метрика, зсув якої описує інсайт"),
    "TARGETS (in)": ("інсайт про метрику", "інсайт, що описує зсув цієї метрики"),
    # the map
    "cell": ("комірка", "шестикутна клітинка мапи; детальніший рівень — менші комірки"),
    "region": ("регіон", "комірка ширшого рівня мапи, що об'єднує детальніші"),
    "zone": ("зона", "виділена вами ділянка мапи, про яку питаєте"),
    "segment": ("сегмент", "частина таблиці, виділена умовами"),
    "points": ("події", "точки з файлу, що потрапили в комірки інсайту"),
    "cells_extent": ("комірки", "комірки мапи, з яких складається інсайт на своєму рівні"),
    "cells_footprint": ("комірки сліду", "комірки цього рівня, що тримають 80 % подій інсайту, знайденого на іншому рівні"),
    "cells_in_view": ("комірки у вікні", "комірки у вікні мапи, де є події інсайту"),
    "window_lift": (
        "частота у вікні",
        "у скільки разів частіше події інсайту трапляються у вікні мапи (комірки з центром у рамці), "
        "ніж очікувано за його часткою на всій мапі; згладжено 30 подіями",
    ),
    "place_lift": (
        "частота в регіоні",
        "у скільки разів частіше події інсайту трапляються в цьому регіоні, ніж очікувано за часткою регіону на всій мапі",
    ),
    "compactness": (
        "суцільність",
        "частка зв'язків сусідства (кільце 1) між комірками інсайту, що лишаються всередині нього: 1 — суцільний блок, 0 — розкидані комірки",
    ),
    "n_eff": (
        "ефективний обсяг",
        "ефективний обсяг для тесту медіани: кількість комірок з непорожнім значенням метрики, зменшена за схожістю сусідніх комірок (Moran's I > 0)",
    ),
    "moran_i": ("схожість сусідів", "наскільки сусідні комірки інсайту схожі за метрикою (Moran's I): більше 0 — схожі, 0 і менше — ні"),
    "sac": (
        "поправка на сусідство",
        "скільки перевірених знахідок значущі без поправки → з поправкою на схожість сусідніх комірок (просторову автокореляцію)",
    ),
    # a theme
    "mass": ("вага теми", "сума сил свідчення її інсайтів"),
    "evidence_mass": ("вага з відповідністю", "сума сил свідчення її інсайтів, кожна помножена на те, наскільки інсайт відповідає темі"),
    "dispersion": ("розкид", "наскільки інсайти теми відрізняються від неї: 0 — усі однакові"),
    # the score breakdown
    "weight_effect": ("внесок зсуву", "частка сили свідчення від величини зсуву, від 0 до 1"),
    "weight_support": ("внесок обсягу", "частка сили свідчення від кількості рядків, від 0 до 1"),
    "weight_emm": ("внесок зміни зв'язків", "бонус за зміну кореляцій, від 0,5 до 1"),
    "sd_score": ("оцінка зсуву", "сума зсувів головних метрик з поправкою на стабільність"),
    "emm_score": ("оцінка зміни зв'язків", "середньоквадратична зміна кореляцій між метриками"),
    "volume_utility": ("корисність обсягу", "√(p · (1 − p)) для частки рядків p: найбільша для половини даних"),
}

# the cells' neighbourhood classes and bands (subgroup_miner/spatial.py), in plain words
LISA = {
    "hot spot": "гаряча точка (багато подій тут і поруч)",
    "cold spot": "холодна точка (мало подій тут і поруч)",
    "high outlier": "багато подій тут, мало поруч",
    "low outlier": "мало подій тут, багато поруч",
    "no cluster": "без скупчення подій",
}
QUARTERS = {
    "q1": "серед найтихіших (нижня чверть)",
    "q2": "нижче середнього",
    "q3": "вище середнього",
    "q4": "серед найжвавіших (верхня чверть)",
    "missing": "без подій",
}
_ORDINAL = ("першій", "другій", "третій", "четвертій", "п'ятій", "шостій", "сьомій", "восьмій", "дев'ятій", "десятій")  # locative
_DEGREE = ((1.0, "слабко"), (2.0, "помірно"), (3.0, "помітно"))  # the buckets of text.magnitude_word
_COVARIANCE = {"strengthens": "сильнішає", "weakens": "слабшає", "reverses": "змінює знак"}
_DERIVED = (  # a geo dataset's cell columns -> their reading, given the source column's words
    (re.compile(r"(.+)_active_days"), "кількість різних днів з подіями в комірці (за {})"),
    (re.compile(r"(.+)_late_share"), "частка подій комірки в другій половині періоду (за {})"),
    (re.compile(r"(.+)_median"), "типове (медіанне) значення {} у подіях комірки"),
    (re.compile(r"(.+)_mode"), "найчастіше значення {} серед подій комірки"),
)


def name(key: str) -> str:
    """The Ukrainian name of a term; an unknown key reads as itself (a new edge type or label never raises)."""
    return TERMS.get(key, (key, ""))[0]


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def column_words(column: str, glossary: Mapping[str, str], geo: bool) -> str:
    """A column in words: the uploader's glossary first, then (geo cells only) the derived column's reading, else the
    humanized name. ``glossary`` is the declared one, never the derived English ``profile.geo.glossary``."""
    if column in glossary:
        return glossary[column]
    if geo:
        if column == "points":
            return "кількість подій у комірці"
        if column.startswith("neighbours_") and not column.endswith("_band"):
            return "середня кількість подій у шести сусідніх комірках"
        if column.startswith("share_"):
            rest = column.removeprefix("share_")
            # ponytail: a source column the glossary does not name splits at the last "_"; a level holding "_" misreads
            source = next((c for c in sorted(glossary, key=len, reverse=True) if rest.startswith(c + "_")), rest.rpartition("_")[0])
            if source:
                return f"частка подій комірки, де {column_words(source, glossary, False)} — {rest[len(source) + 1 :]}"
        for pattern, template in _DERIVED:
            if m := pattern.fullmatch(column):
                return template.format(column_words(m.group(1), glossary, False))
    return humanize(column)


def condition_words(condition: Condition, glossary: Mapping[str, str], geo: bool, bands: Mapping[str, int]) -> str:
    """One ``attribute = value`` in words: the glossary's phrase for the pair, a cell's neighbourhood class or band, a
    quantile band 'у i-й з k смуг' (``bands``: band column -> k), else ``<column words> = <value>`` with the value as written."""
    attribute, value = condition.attribute, condition.value
    if phrase := glossary.get(condition.expr):
        return phrase
    if geo and attribute == "lisa_points" and value in LISA:
        return LISA[value]
    if geo and attribute.startswith("neighbours_") and attribute.endswith("_band") and value in QUARTERS:
        return f"сусідні комірки {QUARTERS[value]}"
    words = column_words(attribute.removesuffix("_band") if attribute.endswith("_band") else attribute, glossary, geo)
    if attribute.endswith("_band") and re.fullmatch(r"q\d+", value):
        i, k = int(value[1:]), bands.get(attribute)
        ordinal = _ORDINAL[i - 1] if i <= len(_ORDINAL) else f"{i}-й"
        where = f"{ordinal} чверті" if k == 4 else f"{ordinal} з {k} смуг" if k else f"смузі {i}"
        return f"{words} у {where} (від найменших значень)"
    return f"{words} = {value}"


def shift_words(z: float) -> str:
    """Direction and degree of a robust shift, gender-neutral: 'помітно вище' (the buckets of ``text.magnitude_word``)."""
    degree = next((word for limit, word in _DEGREE if abs(z) < limit), "різко")
    return f"{degree} {'вище' if z > 0 else 'нижче'}"


def _shift(s: Shift, glossary: Mapping[str, str], geo: bool, first: bool) -> str:
    overall = " загалом" if first else ""
    return (
        f"{column_words(s.metric, glossary, geo)}: {shift_words(s.robust_z)}, "
        f"медіана {format_value(s.local_median)} проти {format_value(s.global_median)}{overall} ({s.robust_z:+.2f} sd)"
    )


def covariance_words(cov: Mapping[str, Any], glossary: Mapping[str, str], geo: bool) -> str:
    """'зв'язок між a і b слабшає: кореляція +0.88 загалом → +0.73 тут' (the predicates of ``text.covariance_change``)."""
    word, _ = covariance_change(dict(cov))
    a, b = (column_words(c, glossary, geo) for c in sorted(cov["pair"]))
    return f"зв'язок між {a} і {b} {_COVARIANCE[word]}: кореляція {cov['global_corr']:+.2f} загалом → {cov['local_corr']:+.2f} тут"


def scope_words(ins: Insight, glossary: Mapping[str, str] | None = None, bands: Mapping[str, int] | None = None) -> str:
    """Where the insight holds, in words: its conditions joined by 'і'."""
    geo = bool(ins.provenance.get("spatial"))
    return " і ".join(condition_words(c, glossary or {}, geo, bands or {}) for c in ins.conditions)


def phenomenon_words(ins: Insight, thresholds: PhenomenonThresholds, glossary: Mapping[str, str] | None = None) -> str:
    """What the insight shows, without where: the target's shift and the next material one (at most two, as the
    evidence lists them), or the correlation change. Any other kind reads by its shifts; a record without one, as its
    headline."""
    glossary, geo = glossary or {}, bool(ins.provenance.get("spatial"))
    if ins.phenomenon_type == "covariance" and ins.covariance.get("pair"):
        return covariance_words(ins.covariance, glossary, geo)
    shifts = thresholds.phenomenon_shifts(ins)[:2] or [s for s in ins.shifts if s.metric == ins.target]
    return "; ".join(_shift(s, glossary, geo, first=i == 0) for i, s in enumerate(shifts)) or headline(ins)


def reading(ins: Insight, thresholds: PhenomenonThresholds, glossary: Mapping[str, str] | None = None, bands: Mapping[str, int] | None = None) -> str:
    """An insight in one sentence: 'Де <scope> — <phenomenon>.' ``glossary`` is the dataset's declared one (empty
    without it), ``bands`` the quantile-band columns' band counts."""
    return f"Де {scope_words(ins, glossary, bands)} — {phenomenon_words(ins, thresholds, glossary)}."


def theme_reading(signature: Sequence[Mapping[str, Any]], n_patterns: int, distinct_scopes: int) -> str:
    """A theme in one sentence from its two strongest signature entries (docs/05_latent_anchors.md §5.8)."""
    parts = []
    for entry in signature[:2]:
        label, up = entry["component"], entry["value"] > 0
        if label.startswith("correlation between "):
            pair = label.removeprefix("correlation between ").replace(" and ", " і ")
            parts.append(f"зв'язок {pair} {'сильнішає' if up else 'слабшає'}")
        else:
            parts.append(f"{label} {'вище' if up else 'нижче'}")
    found = f"{n_patterns} {_plural(n_patterns, 'інсайт', 'інсайти', 'інсайтів')} у {distinct_scopes} "
    found += _plural(distinct_scopes, "частині", "частинах", "частинах") + " даних"
    return f"Повторювана картина: {', '.join(parts)} — {found}." if parts else f"Тема без інсайтів: {found}."


def region_reading(region_id: str, picture: Mapping[str, int]) -> str:
    """A map region in one sentence: its H3 cell as a literal and what it holds (``vertical.picture``); no place name."""
    _, resolution, cell = region_id.split(":", 2)
    p, d, t = picture["patterns"], picture["datasets"], picture["themes"]
    return (
        f"Регіон {cell} (рівень {resolution}): {p} {_plural(p, 'інсайт', 'інсайти', 'інсайтів')} "
        f"з {d} {_plural(d, 'датасету', 'датасетів', 'датасетів')}, {t} {_plural(t, 'тема', 'теми', 'тем')}."
    )


def lift_words(kind: str, lift: float) -> str:
    """A lift in words: 'window' (the map window's, smoothed by its prior) or 'place' (a region's, unsmoothed)."""
    if abs(lift - 1) < 0.05:
        words = "як на всій мапі"
    else:
        words = f"у {max(lift, 1 / lift):.1f} раза {'частіше' if lift > 1 else 'рідше'}, ніж очікувано за часткою на всій мапі"
    return words + (" (згладжено)" if kind == "window" else "")
