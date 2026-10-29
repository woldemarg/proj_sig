"""How SIG talks to people (docs/08_interface.md §8.5): the Ukrainian name and one-line meaning of every term the
console and the chat show, and the readings of an insight, a theme, a region and a lift built from them.

Beside ``insight_contracts.text``, not in it: that module's English phrases build the embedded canonical document
(docs/04_representation.md §4.2), so changing them re-embeds every workspace; this one is presentation only. Plain
words for someone who reads the map and the chat, not the graph: no "latent", "attractor", "edge" or "node". The
console keeps a copy of ``TERMS`` (``TERMS`` in ``app.js``) that ``tests/test_wording.py`` holds equal to this one.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Any

from insight_contracts.insight import Condition, Insight, PhenomenonThresholds, Shift
from insight_contracts.text import covariance_change, format_p, format_value, humanize, magnitude_word

# key -> (name, meaning). A relation's key is its edge type, "<TYPE> (in)" for the incoming direction (the drawer's
# neighbour groups and a reversed path step use the same keys); its name labels the node at the other end.
TERMS: dict[str, tuple[str, str]] = {
    "insight": ("інсайт", "перевірена знахідка: частина даних, у якій метрика або зв'язок двох метрик помітно інші, ніж у всіх даних"),
    "theme": ("тема", "картина, що повторюється в інсайтах з різних частин даних, наприклад «discount вище, margin нижче»"),
    "scope": ("де", "умови, що виділяють частину даних, про яку інсайт"),
    "metric": ("метрика", "числова колонка, зсув якої описує інсайт"),
    "sd": ("sd", "різниця медіан у робастних стандартних відхиленнях: 0,6745 · Δмедіани / MAD, зі знаком, |z| не більше 10"),
    "weight": (
        "сила свідчення",
        "від нижньої межі до 1: зважене геометричне середнє сили зсуву, стабільності, обсягу і зміни зв'язків; слабкий бік не компенсується сильним",
    ),
    "support": ("рядки", "скільки рядків даних охоплює інсайт і яку частку всіх рядків"),
    "p_adjusted": ("скоригована p", "результат тесту медіани з поправкою на кількість перевірок; поріг відбору, а не точна ймовірність"),
    "stability": (
        "стабільність",
        "1 − min(CV, 0,9), де CV — коефіцієнт варіації суми |z| головних зсувів на 20 бутстреп-вибірках; "
        "від 0,1 до 1, 0 коли зсуву немає; ближче до 1 — менше коливання",
    ),
    "drivers": (
        "супутні чинники",
        "колонки, чиї значення в цій частині даних розподілені помітно інакше (тест хі-квадрат); вони можуть пояснювати зсув",
    ),
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
    # time: a finding's series over the calendar periods of its dataset (docs/02_discovery.md §2.10)
    "trend": ("динаміка", "як рівень знахідки змінювався в часі, від першого до останнього виміряного періоду її датасету"),
    "trend_rising": ("зростає", "від першого до останнього виміряного періоду рівень знахідки помітно піднявся"),
    "trend_falling": ("спадає", "від першого до останнього виміряного періоду рівень знахідки помітно опустився"),
    "trend_flat": ("без змін", "за весь проміжок рівень знахідки помітно не змінився, піків і спадів у ньому немає"),
    "trend_spike": ("сплеск", "рівень знахідки повернувся туди, де був, але між тим мав пік або спад"),
    "time_current": ("зараз", "рівень знахідки в останньому виміряному періоді, після згладжування"),
    "time_unmeasured": ("не виміряно", "у жодному періоді не було досить рядків знахідки, щоб виміряти її рівень"),
    "time_faded": ("нижче порога", "в останньому виміряному періоді відхилення знахідки слабше за поріг, з якого воно вважається суттєвим"),
    "time_carrier": (
        "зосереджено в",
        "частина знахідки, рядки якої в цьому періоді змістилися найбільше порівняно з рештою знахідки — понад звичну різницю між ними",
    ),
    "time_echo": (
        "як у ширшої знахідки",
        "ширша знахідка, що охоплює всі рядки цієї, має той самий пік чи спад у тому самому періоді: це одна подія",
    ),
    "temporal_hub": (
        "період часу",
        "день, тиждень, місяць чи квартал, у якому знахідки мали пік або спад чи починали або закінчували своє ядро, і кожен ширший період над ним; той самий для всіх датасетів",
    ),
    "PEAKED_AT": ("пік у періоді", "у цьому періоді рівень знахідки був помітно вищим, ніж до і після"),
    "PEAKED_AT (in)": ("пік знахідки", "знахідка, рівень якої в цьому періоді був помітно вищим, ніж до і після"),
    "DROPPED_AT": ("спад у періоді", "у цьому періоді рівень знахідки був помітно нижчим, ніж до і після"),
    "DROPPED_AT (in)": ("спад знахідки", "знахідка, рівень якої в цьому періоді був помітно нижчим, ніж до і після"),
    "time_core": ("ядро", "найкоротший проміжок послідовних періодів, у якому лежить 80 % рядків знахідки з датою: де вона живе в часі"),
    "time_active": ("діє в періоді", "знахідка, ядро якої перетинає цей період"),
    "as_of": ("станом на", "дата, від якої рахують свіжість знахідок у відповіді: остання дата даних"),
    "period": (
        "період",
        "дні, обрані на смузі періоду: у відповіді вище знахідки, виміряні в них, і найвище — з піком чи спадом у них; решта лишається нижче",
    ),
    "STARTS_AT": ("починається в періоді", "перший період ядра знахідки"),
    "STARTS_AT (in)": ("початок ядра знахідки", "знахідка, ядро якої починається в цьому періоді"),
    "ENDS_AT": ("закінчується в періоді", "останній період ядра знахідки"),
    "ENDS_AT (in)": ("кінець ядра знахідки", "знахідка, ядро якої закінчується в цьому періоді"),
    "WITHIN": (
        "входить у період",
        "ширший період календаря, що містить цей: день у тижні, тиждень у місяці, якому належить більшість його днів, місяць у кварталі",
    ),
    "WITHIN (in)": ("вужчий період", "період детальнішого рівня календаря всередині цього"),
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
    "SPECIALIZES": ("ширший інсайт", "ширша знахідка, частина якої — ця: ті самі умови без однієї"),
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
    "cells_events": ("комірки подій", "комірки цього рівня, що тримають події інсайту, знайденого на іншому рівні"),
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
    # the score breakdown: the factors of the weight's geometric mean, each from 0 to 1
    "weight_effect": ("множник зсуву", "множник сили свідчення від величини зсуву, від 0 до 1"),
    "weight_support": ("множник обсягу", "корисність обсягу, поділена на її найбільше можливе значення, від 0 до 1"),
    "weight_emm": ("множник зміни зв'язків", "бонус за зміну кореляцій, від 0,5 до 1: ніколи не обнуляє силу свідчення"),
    "sd_score": ("оцінка зсуву", "сума зсувів головних метрик з поправкою на стабільність"),
    "emm_score": ("оцінка зміни зв'язків", "середньоквадратична зміна рангових кореляцій між метриками, зменшена для малих частин даних"),
    "volume_utility": ("корисність обсягу", "√p · (1 − p) для частки рядків p: найбільша, коли інсайт охоплює третину даних"),
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
    "q2": "нижче середини",
    "q3": "вище середини",
    "q4": "серед найжвавіших (верхня чверть)",
    "missing": "без подій",
}
_ORDINAL = ("першій", "другій", "третій", "четвертій", "п'ятій", "шостій", "сьомій", "восьмій", "дев'ятій", "десятій")  # locative
_DEGREE = {"mild": "слабко", "moderate": "помірно", "strong": "помітно", "extreme": "різко"}  # text.magnitude_word's buckets
_COVARIANCE = {"strengthens": "сильнішає", "weakens": "слабшає", "reverses": "змінює знак"}
# a geo dataset's cell column suffix -> its reading; the source column's words stay in the nominative, as a declared phrase is
_DERIVED = {
    "_active_days": "кількість днів з подіями в комірці ({})",
    "_late_share": "частка подій комірки в другій половині періоду ({})",
    "_median": "{} (медіана подій комірки)",
    "_mode": "{} (найчастіше серед подій комірки)",
}


JARGON = (  # analyst words a plain-language answer leaves out (system-prompt rule 10, narration.jargon): English whole words,
    # Ukrainian stems
    *("sd", "MAD", "median", "quartile", "q1", "q2", "q3", "q4", "LISA", "subgroup", "seed", "transversal", "latent", "anchor"),
    *("медіан", "квартил", "підгруп", "якор", "якір", "латентн", "трансверсальн", "атрактор"),
)


# a calendar period as the grid names one and a question may write it, in any case: 2031-W13, 2031-Q1 or Q1 2031, 2031-03-24, 2031-03
PERIOD = re.compile(
    r"(?<![\w-])(?:(\d{4})-W(\d{2})|(\d{4})-Q([1-4])|Q([1-4])\s+(\d{4})|(\d{4})-(\d{2})-(\d{2})|(\d{4})-(\d{2}))(?![\d-])", re.IGNORECASE
)


def name(key: str) -> str:
    """The Ukrainian name of a term; an unknown key reads as itself (a new edge type or label never raises)."""
    return TERMS.get(key, (key, ""))[0]


def plural(n: int, one: str, few: str, many: str) -> str:
    """The Ukrainian form of a noun after the number ``n``: 1 інсайт, 2 інсайти, 5 інсайтів."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def source_column(column: str, declared: Collection[str]) -> str | None:
    """The declared column a dataset column stands for: itself, a band's (``<c>_band``), a geo cell's derivation
    (``<c>_median`` / ``_mode`` / ``_late_share`` / ``_active_days``, ``share_<c>_<level>``); None when it is none of them."""
    if column in declared:
        return column
    if column.endswith("_band"):  # a band of a column or of a cell's derivation
        return source_column(column.removesuffix("_band"), declared)
    if column.startswith("neighbours_"):  # a counter's neighbourhood lag
        return source_column(column.removeprefix("neighbours_"), declared)
    for suffix in _DERIVED:
        if column.endswith(suffix) and column.removesuffix(suffix) in declared:
            return column.removesuffix(suffix)
    if column.startswith("share_"):
        # ponytail: the longest declared prefix wins; a declared "region" reads share_region_code_UA when region_code is
        # not declared, as the profile keeps no share column's source to tell them apart
        rest = column.removeprefix("share_")
        return next((c for c in sorted(declared, key=len, reverse=True) if rest.startswith(c + "_")), None)
    return None


def time_words(facts: Mapping[str, Any]) -> str:
    """A finding's time line in words (the evidence-only answer, the console's chip): its metric's trend (or that it was
    never measured), its current level and an event — its metric's strongest, else another metric's, named, with the
    condition that carries it; a broader finding's event when it has none of its own
    (``graph_query_engine.temporal.time_facts``)."""
    if facts["trend"] is None:
        return f"{name('trend')}: {name('time_unmeasured')}"
    parts = [f"{name('trend')}: {name('trend_' + facts['trend'])}"]
    if facts["current"] is not None:
        level = name("time_faded") if facts["faded"] else shift_words(facts["current"])
        parts.append(f"{name('time_current')}: {level} ({facts['current']:+.2f} sd)")
    if events := facts["events"] or facts.get("echoes", []):
        event = next((e for e in events if e["metric"] == facts["metric"]), events[0])
        other = "" if event["metric"] == facts["metric"] else f" ({humanize(event['metric'])})"
        carrier = f", {name('time_carrier')}: {humanize(c['attribute'])} — {c['value']}" if (c := event.get("carrier")) else ""
        echo = f" ({name('time_echo')})" if "echo_of" in event else ""
        parts.append(f"{name(event['kind'])} {event['period']}{other}{carrier}{echo}")
    return "; ".join(parts)


def column_words(column: str, glossary: Mapping[str, str], geo: bool) -> str:
    """A column in words: the uploader's glossary first, then (geo cells only) the derived column's reading, else the
    humanized name. ``glossary`` is the declared one, never the derived English ``profile.geo.glossary``."""
    if column in glossary:
        return glossary[column]
    if geo:
        if column == "points":
            return "кількість подій у комірці"
        if column.startswith("neighbours_") and not column.endswith("_band"):
            return f"{column_words(column.removeprefix('neighbours_'), glossary, geo)} (середнє шести сусідніх комірок)"
        if column.startswith("share_"):
            rest = column.removeprefix("share_")
            # ponytail: a source column the glossary does not name splits at the last "_"; a level holding "_" misreads
            source = source_column(column, glossary) or rest.rpartition("_")[0]
            if source:
                return f"частка подій комірки, де {column_words(source, glossary, False)} — {rest[len(source) + 1 :]}"
        for suffix, template in _DERIVED.items():
            if column.endswith(suffix):
                return template.format(column_words(column.removesuffix(suffix), glossary, False))
    return humanize(column)


def condition_words(condition: Condition, glossary: Mapping[str, str], geo: bool, bands: Mapping[str, int]) -> str:
    """One ``attribute = value`` in words: the glossary's phrase for the pair (a cell's mode column reads its source
    column's pair), a cell's neighbourhood class or band (named by its counter), a quantile band 'у i-й з k смуг'
    (``bands``: band column -> k), a blank ('missing', the ingestion's sentinel) 'не вказано', else
    ``<column words> = <value>`` with the value as written."""
    attribute, value = condition.attribute, condition.value
    pair = f"{attribute.removesuffix('_mode')}={value}" if geo and attribute.endswith("_mode") else condition.expr
    if phrase := glossary.get(condition.expr) or glossary.get(pair):
        return phrase
    if geo and attribute == "lisa_points" and value in LISA:
        return LISA[value]
    if geo and attribute.startswith("neighbours_") and attribute.endswith("_band") and value in QUARTERS:
        counter = attribute.removeprefix("neighbours_").removesuffix("_band")
        return f"сусідні комірки {QUARTERS[value]} ({column_words(counter, glossary, geo)})"  # a cell has one band per counter
    words = column_words(attribute.removesuffix("_band"), glossary, geo)
    if attribute.endswith("_band") and value[:1] == "q" and value[1:].isdigit():
        i, k = int(value[1:]), bands.get(attribute)
        ordinal = _ORDINAL[i - 1] if i <= len(_ORDINAL) else f"{i}-й"
        where = f"{ordinal} чверті" if k == 4 else f"{ordinal} з {k} смуг" if k else f"смузі {i}"
        return f"{words} у {where} (від найменших значень)"
    return f"{words} не вказано" if value == "missing" else f"{words} = {value}"


def short_conditions(ins: Insight, glossary: Mapping[str, str], bands: Mapping[str, int]) -> list[str]:
    """Each condition in the fewest words, for a list row: its value where it reads ``<column> = <value>``, else its
    phrase (a cell's class or band, a declared pair, a blank), which a code alone would not say."""
    geo = bool(ins.provenance.get("spatial"))
    out = []
    for c in ins.conditions:
        words = condition_words(c, glossary, geo, bands)
        out.append(c.value if words.endswith(f" = {c.value}") else words)
    return out


def shift_words(z: float) -> str:
    """Direction and degree of a robust shift, gender-neutral: 'помітно вище' (the buckets of ``text.magnitude_word``, on
    the shift as it is shown, to two decimals: -0.998 shows as -1.00 sd and reads помірно)."""
    return f"{_DEGREE[magnitude_word(round(z, 2))]} {'вище' if z > 0 else 'нижче'}"


def _shift_head(s: Shift, glossary: Mapping[str, str], geo: bool) -> str:
    return f"{column_words(s.metric, glossary, geo)}: {shift_words(s.robust_z)}"


def _shift(s: Shift, glossary: Mapping[str, str], geo: bool, first: bool) -> str:
    overall = " загалом" if first else ""
    return f"{_shift_head(s, glossary, geo)}, медіана {format_value(s.local_median)} проти {format_value(s.global_median)}{overall} ({s.robust_z:+.2f} sd)"


def _covariance_head(cov: Mapping[str, Any], glossary: Mapping[str, str], geo: bool) -> str:
    word, _ = covariance_change(cov)
    a, b = (column_words(c, glossary, geo) for c in sorted(cov["pair"]))
    return f"зв'язок метрик {a} і {b} {_COVARIANCE[word]}"


def covariance_words(cov: Mapping[str, Any], glossary: Mapping[str, str], geo: bool) -> str:
    """'зв'язок метрик a і b слабшає: кореляція +0.88 загалом → +0.73 тут' (the predicates of ``text.covariance_change``)."""
    return f"{_covariance_head(cov, glossary, geo)}: кореляція {cov['global_corr']:+.2f} загалом → {cov['local_corr']:+.2f} тут"


def reading(
    ins: Insight, thresholds: PhenomenonThresholds, glossary: Mapping[str, str] | None = None, bands: Mapping[str, int] | None = None
) -> dict[str, str]:
    """An insight in words: ``where`` its conditions joined by 'і', ``what`` the target's shift and the next material one
    (at most two, as the evidence lists them) or the correlation change, ``headline`` the first clause of ``what`` (the
    target and its degree, or the correlation change), which a list row shows, ``gist`` the clauses of ``what`` without
    their numbers, which the drawer leads with above its table of shifts. ``glossary`` is the dataset's declared one
    (empty without it), ``bands`` the quantile-band columns' band counts. A surface that shows the scope already prints
    ``what`` alone; ``sentence`` joins the two."""
    glossary, geo = glossary or {}, bool(ins.provenance.get("spatial"))
    if ins.phenomenon_type == "covariance":
        headline, what = _covariance_head(ins.covariance, glossary, geo), covariance_words(ins.covariance, glossary, geo)
        gist = headline
    else:
        shifts = thresholds.phenomenon_shifts(ins)[:2]
        headline = _shift_head(shifts[0], glossary, geo) if shifts else ""
        what = "; ".join(_shift(s, glossary, geo, first=i == 0) for i, s in enumerate(shifts))
        gist = "; ".join(_shift_head(s, glossary, geo) for s in shifts)
    where = " і ".join(condition_words(c, glossary, geo, bands or {}) for c in ins.conditions)
    return {"where": where, "what": what, "headline": headline, "gist": gist}


def sentence(words: Mapping[str, str]) -> str:
    """A reading as one sentence, without its full stop: 'Де <where> — <what>'."""
    return f"Де {words['where']} — {words['what']}"


def theme_reading(props: Mapping[str, Any], glossary: Mapping[str, str] | None = None) -> str:
    """A theme in one sentence from its Attractor props: its two strongest signature entries, its insights and scopes
    (docs/05_latent_anchors.md §5.8); ``glossary`` the declared one of its insights' datasets (``graph.theme_glossary``),
    so its metrics read in the uploader's words (a signature label is a humanised column name) and a cell's derived
    columns as in its insights."""
    n_patterns, distinct_scopes, parts = props["n_patterns"], props["distinct_scopes"], []

    def words(label: str) -> str:
        # ponytail: the humanised label is the column with "_" read as " "; a share level holding a space reads it back as "_"
        return column_words(label.replace(" ", "_"), glossary or {}, True)

    for entry in props["signature"][:2]:
        label, up = entry["component"], entry["value"] > 0
        if label.startswith("correlation between "):
            # ponytail: a column name holding " and " splits at its first one
            a, _, b = label.removeprefix("correlation between ").partition(" and ")
            parts.append(f"зв'язок метрик {words(a)} і {words(b)} {'сильнішає' if up else 'слабшає'}")
        else:
            parts.append(f"{words(label)} {'вище' if up else 'нижче'}")
    found = f"{n_patterns} {plural(n_patterns, 'інсайт', 'інсайти', 'інсайтів')} у {distinct_scopes} "
    found += plural(distinct_scopes, "частині", "частинах", "частинах") + " даних"
    return f"Повторювана картина: {', '.join(parts)} — {found}." if parts else f"Тема без інсайтів: {found}."


def region_reading(picture: Mapping[str, int]) -> str:
    """What a map region holds (``vertical.picture``) in one sentence; the drawer's title names the region, no place name."""
    p, d, t = picture["patterns"], picture["datasets"], picture["themes"]
    return (
        f"{p} {plural(p, 'інсайт', 'інсайти', 'інсайтів')} "
        f"з {d} {plural(d, 'датасету', 'датасетів', 'датасетів')}, {t} {plural(t, 'тема', 'теми', 'тем')}."
    )


def lift_words(kind: str, lift: float) -> str:
    """A lift in words: 'window' (the map window's, smoothed by its prior) or 'place' (a region's, unsmoothed). Both are
    listed only at 1 or more."""
    times = round(lift, 1)
    if times <= 1:
        words = "як на всій мапі"
    else:
        count = f"{int(times)} {plural(int(times), 'раз', 'рази', 'разів')}" if times.is_integer() else f"{times} раза"
        words = f"у {count} частіше, ніж очікувано за часткою на всій мапі"
    return words + (" (згладжено)" if kind == "window" else "")


def value_words(ins: Insight) -> dict[str, str]:
    """What an insight's test numbers say: ``p_adjusted`` its bucket and the exact value, and for a geo insight
    ``n_eff``, the effective sample size of its median test with the cells it counts. ``n_eff`` is the cells holding a
    value of the metric times ``(1 − ρ) / (1 + ρ)``, ``ρ`` Moran's I clipped to [0, 0.99] as
    ``spatial.effective_n_factor`` clips it: a sparse cell holds no value, so ``n_eff`` below the extent's cells says
    nothing of its neighbours by itself, and ``I ≤ 0`` reduces nothing. With ``I > 0`` the counted cells come back from
    the rounded ``n_eff`` and ``I``, so they read as approximate and never exceed the extent's cells."""
    p = ins.p_adjusted
    if p is None:
        out = {"p_adjusted": "не перевірялося (зміна зв'язку)"}
    else:
        exact = f" ({p:.1e})" if 0 < p < 0.05 else " (< 1e-300)" if p == 0 else ""  # above 0.05 the bucket is the number
        out = {"p_adjusted": format_p(p) + exact}
    spatial = ins.provenance.get("spatial") or {}
    if spatial.get("n_eff") is not None:
        moran = spatial.get("moran_i") or 0.0
        rho = min(max(moran, 0.0), 0.99)
        valued = round(spatial["n_eff"] * (1 + rho) / (1 - rho))
        valued = min(valued, int(spatial.get("cells") or valued))  # the rounded n_eff and I can give one cell more than the extent holds
        tail = f"зменшено за схожістю сусідніх комірок (Moran's I {moran:.2f})" if rho > 0 else "без зменшення за автокореляцією"
        cells = plural(valued, "непорожня комірка", "непорожні комірки", "непорожніх комірок")
        out["n_eff"] = f"{spatial['n_eff']:g}: {'≈' if rho > 0 else ''}{valued} {cells} метрики, {tail}"
    return out
