"""``@auto``: the levels read from the points and the error that names a level the cap left out (docs/02_discovery.md §2.9)."""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
import pytest
from perf_benchmark import write_synthetic_points

from insight_graph_service.core.engine import PipelineError
from insight_graph_service.core.settings import PROJECT_ROOT
from subgroup_miner.config import MinerConfig
from subgroup_miner.ingestion import auto_levels, load_dataset
from subgroup_miner.spatial import SpatialError, select_levels


def test_auto_picks_the_levels_from_the_points(points_csv, engine, tmp_path, monkeypatch):
    """``@auto``: the finest resolution with enough cells, half its points in dense cells and few enough cells at its
    region level (its own below the finest region level), then the coarser region levels; the same file the same levels;
    a denser file a finer level; a sparse wide file the level its points fill, not the one its territory does; an upload
    with ``@auto`` is the upload of its levels."""
    cfg, levels = MinerConfig(), (7, 5, 3, 1)
    california = PROJECT_ROOT / "data" / "demo" / "california_housing.csv"
    picked = auto_levels(california, "auto", cfg, levels)
    assert picked["chosen"] == [6, 5] and picked == auto_levels(california, "auto", cfg, levels)
    rng = np.random.default_rng(0)
    finest = {
        n: select_levels(rng.uniform(36, 37, n), rng.uniform(-120, -119.17, n), levels, min_rows=50, min_cell_points=5)["chosen"][0]
        for n in (2_000, 50_000)
    }
    assert finest == {2_000: 6, 50_000: 7}  # the denser file the finer level; 8 has 10,000 cells of its own
    with pytest.raises(SpatialError, match="points in dense cells"):  # many cells, none dense: the refusal names that
        select_levels(rng.uniform(-80, 80, 3_000), rng.uniform(-179, 179, 3_000), levels, min_rows=50, min_cell_points=5)
    town = select_levels(50 + rng.random(5_000) * 0.09, 30 + rng.random(5_000) * 0.14, levels, min_rows=50, min_cell_points=5)
    assert town["chosen"] == [9] and town["candidates"]["10"]["dense_points"] < 0.5  # 10 fits the cap, but its points sit alone
    sparse = pd.read_csv(write_synthetic_points(tmp_path, 30_000))  # 30,000 points over the whole state, most cells empty-ish
    wide_sparse = select_levels(sparse["lat"].to_numpy(), sparse["lon"].to_numpy(), levels, min_rows=50, min_cell_points=5)
    assert wide_sparse["chosen"] == [6, 5] and wide_sparse["candidates"]["6"]["dense_points"] > 0.5  # 4 by the share of cells
    big = np.random.default_rng(1)  # a large export: level 7 dense but 13,500 cells, level 5 only ~310
    picked = select_levels(35 + big.random(80_000) * 3, -121 + big.random(80_000) * 2.5, levels, min_rows=50, min_cell_points=5)
    assert picked["chosen"] == [6] and "GEO_AUTO_MAX_CELLS" in picked["reasons"]["7"] and "GEO_AUTO_MIN_CELLS" in picked["reasons"]["5"]
    box = 35 + big.random(80_000) * 3, -121 + big.random(80_000) * 2.5
    assert select_levels(*box, (5, 3, 1), min_rows=50, min_cell_points=5)["chosen"] == [6]  # finer than every region level: its own cells
    with monkeypatch.context() as m:  # a cap of 300: 6 counts the cells of 5, 4 those of 3
        m.setattr("subgroup_miner.spatial.GEO_AUTO_MAX_CELLS", 300)
        capped = select_levels(*box, levels, min_rows=50, min_cell_points=5)
    assert capped["chosen"] == [4] and capped["capped"] == [7, 6, 5] and capped["reasons"]["6"].endswith("cells at 5 > GEO_AUTO_MAX_CELLS")
    wide = select_levels(34 + big.random(40_000) * 4, -122 + big.random(40_000) * 6, levels, min_rows=50, min_cell_points=5)
    assert wide["candidates"]["6"]["cells"] > 5_000 >= wide["candidates"]["6"]["cap_cells"] == wide["candidates"]["5"]["cells"]
    assert wide["chosen"] == [6, 5] and wide["capped"] == []  # the wide box: 6 counts its IN_REGION regions at 5
    world = select_levels(big.uniform(-80, 80, 250_000), big.uniform(-179, 179, 250_000), levels, min_rows=50, min_cell_points=5)
    assert world["chosen"] == [3] and world["candidates"]["3"]["cells"] > 5_000  # every level over the cap: the coarsest that fits
    rng = np.random.default_rng(48)  # parents of the resolution-10 cells counted 50 cells at 5, the points' own cells 49
    n, size = int(rng.integers(2000, 6000)), float(rng.uniform(0.6, 1.2))
    lat0, lon0 = rng.uniform(32.6, 40.7), rng.uniform(-124.3, -115.4)  # inside CALIFORNIA
    lat, lon = lat0 + rng.random(n) * size, lon0 + rng.random(n) * size
    picked = select_levels(lat, lon, levels, min_rows=50, min_cell_points=5)
    for res, candidate in picked["candidates"].items():  # each count is the cell table build_cells makes at that level
        assert candidate["cells"] == len({h3.latlng_to_cell(a, b, int(res)) for a, b in zip(lat, lon)})
    assert 5 not in picked["chosen"] and picked["candidates"]["5"]["cells"] == 49

    with points_csv.open("rb") as f:
        record, table = engine.upload("points.csv", f, geo="auto@auto", table=True)
    assert record["profile"]["geo"]["levels"]["chosen"] == [7]  # where the selection stays, whatever the batch's end
    done = engine.process(record["batch_id"])
    assert done["level"] == "7" and done["profile"]["geo"]["levels"]["chosen"] == [7] and "geo_levels" not in done
    assert table["level"] == "table" and not table["profile"]  # the selection is the geo levels' alone
    assert done["dataset_id"] == load_dataset(points_csv, engine.settings.miner, geo="auto@7").dataset_id
    pd.DataFrame({"lat": [36.5] * 60, "lon": [-119.2] * 60, "x": range(60)}).to_csv(tiny := tmp_path / "tiny.csv", index=False)
    with tiny.open("rb") as f, pytest.raises(PipelineError) as exc:
        engine.upload("tiny.csv", f, geo="auto@auto")  # one cell at every resolution
    assert exc.value.code == "invalid_options" and len(list(engine.ws.uploads_dir.glob("*"))) == 1  # not kept


def test_an_auto_level_with_nothing_names_the_level_the_cap_left_out(engine, tmp_path, monkeypatch):
    """``@auto``'s finest level that finds nothing while finer ones were left out by ``GEO_AUTO_MAX_CELLS`` alone: the
    batch's error names the nearest of them as the explicit option to try."""
    rng = np.random.default_rng(0)
    lat, lon = 36 + rng.random(40_000) * 0.5, -120 + rng.random(40_000) * 0.5
    frame = pd.DataFrame({"latitude": lat, "longitude": lon, "sector": np.where(lat > 36.25, "north", "south"), "load": rng.normal(10, 2, 40_000)})
    frame.to_csv(path := tmp_path / "noise.csv", index=False)
    monkeypatch.setattr("subgroup_miner.spatial.GEO_AUTO_MAX_CELLS", 400)  # 8 (3,200 cells) and 7 (500) over it, 6 under
    with path.open("rb") as f:
        (record,) = engine.upload("noise.csv", f, geo="latitude,longitude@auto")
    assert record["profile"]["geo"]["levels"]["capped"] == [8, 7]
    done = engine.process(record["batch_id"])
    assert done["level"] == "6" and done["error"]["code"] in {"no_candidates", "no_viable_insights"}
    assert done["error"]["message"].endswith("@auto left latitude,longitude@7 out for GEO_AUTO_MAX_CELLS alone, try it")
