"""Check registry for the diagnostics stage.

Every threshold the stage applies lives here with its provenance, alongside the
mechanism constants it must agree with (imported, never redefined). Changing a
number here is a reviewed decision; the lane modules only read this table.
Severity semantics: GATE hard-fails the stage, FLAG is counted and listed,
REPORT is distributional context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from rra_population_model.postprocess.census_rake.utils import (
    DENSITY_CAP_MIN_CENSUS,
    RF_LAMBDA_PERSONS,
)
from rra_population_model.postprocess.raking_factors.runner import (
    NO_CENSUS_RF_TOLERANCE,
)

Severity = Literal["GATE", "FLAG", "REPORT"]
Lane = Literal["A", "B", "C"]
UnitOfWork = Literal["census", "time_point", "block", "release"]

SEVERITY_RANK: dict[str, int] = {"REPORT": 0, "FLAG": 1, "GATE": 2}

# A11: a census count below this is treated as "census-0" (one- and two-person
# units are enumeration noise, not a statement that nobody lives there).
CENSUS_ZERO_FLOOR = 5.0
# A3 percentile flags need a family large enough for a q99.9 to mean anything.
MIN_FAMILY_UNITS = 30
# B1: per-pixel people bounds are stated for a 40m pixel and scale by area.
PIXEL_AREA_M2: dict[str, float] = {"40": 1600.0, "100": 10_000.0}
# B1: pixels kept per block per time point; the collator takes the global top-N.
TOP_K_PER_BLOCK = 20
TOP_N_PIXELS = 100
# B6/B9: blocks sampled per release for the pixel-trajectory / partial-coverage passes.
SAMPLE_BLOCKS = 200
SAMPLE_SEED = 20260917


@dataclass(frozen=True)
class Check:
    id: str
    lane: Lane
    severity: Severity
    unit_of_work: UnitOfWork
    description: str
    thresholds: dict[str, float] = field(default_factory=dict)
    provenance: str = ""
    enabled: bool = True
    guidance: str = ""  # conceptual framing and how to read the result; rendered into summary.md


CHECKS: dict[str, Check] = {
    c.id: c
    for c in [
        # ---- Lane A: census-stage checks over the per-shape tables / rf_prior ----
        Check("A1", "A", "GATE", "census", "Anchor exactness: gated raked(t_c) sums to gated census per task",
              {"rel_tol": 1e-6, "abs_floor": 1e-6}, "in-task guard, census_rake.utils.compute_shape_values"),
        Check("A2", "A", "GATE", "census", "Growth-density: credited unit above its density_bound (GATE = cap regression); at-anchor exceedances REPORT",
              {}, "35x audit; 411-unit flag set; one-sided cap y <= max(C, bound * n_on). At-anchor exceedances (105k units, "
              "first cluster run) are the accepted established-density class and A3 covers extremes, so they are REPORT rows"),
        Check("A3", "A", "FLAG", "census", "At-anchor occupancy extremes: census / n_on(t_c) absolute, or a multiple of the parent family's median",
              {"anchor_abs": 250.0, "parent_median_mult": 10.0, "min_census": DENSITY_CAP_MIN_CENSUS},
              "audit_density_concentration ANCHOR_ABS_FLAG was 25 people per ON 40m pixel, tuned on US blocks; the first cluster "
              "run flagged 416k units / 460M people at 25, dominated by KOR/JPN high-rise (HKG median unit density 113, q99 236), "
              "so the absolute line sits above HKG's q99 and the relative rule (10x family median, family >= 30) carries outliers"),
        Check("A4", "A", "FLAG", "census", "Density concentration: max_t density / anchor density",
              {"conc": 2.0, "min_census": DENSITY_CAP_MIN_CENSUS}, "audit_density_concentration CONC_FLAG"),
        Check("A5", "A", "FLAG", "census", "Trajectory sanity: max/min swing, gate blinks, one-quarter steps",
              {"swing": 2.0, "max_blinks": 0}, "audit gate-blink + series-start findings"),
        Check("A6", "A", "REPORT", "census", "Gate-loss accounting: census>0 units never gated / ungated at the anchor",
              {}, "DESIGN section 7 accepted behavior"),
        Check("A7", "A", "REPORT", "census", "Capped-credit accounting per time point",
              {}, "density cap adoption; post-OBM diff baseline"),
        Check("A8", "A", "FLAG", "census", "rf_country / lambda-probe roll-up with plausibility band",
              {"median_mult": 5.0},
              "Lambda guard. Band is relative to the median rf_country across censuses (applied in the collator): a rate-unit model "
              "puts every census near 1600, so an absolute band flagged all 206; the informative cases are per-census outliers "
              "(SJM 80k, PNG 12k, BDI/DJI/HKG ~9.5k vs median 1670 on the first cluster run)"),
        Check("A9", "A", "FLAG", "census", "Vanished census units: enumerated shapes absent from every table (FLAG when they carry people)",
              {}, "validity audit F13a; 3,038 vanished units on the first cluster run, 2,900 of them zero-population water blocks -> REPORT"),
        Check("A10", "A", "FLAG", "census", "Prior-dominated AND uncapped pooling parents (both backstops off)",
              {"lambda_persons": RF_LAMBDA_PERSONS}, "DESIGN section 7 (audit 2026-09-15); mirrors the census_rf_main print-only warning"),
        Check("A11", "A", "FLAG", "census", "Phantom stock: census-0 units the model populates (FLAG when base x rf_country reaches the floor)",
              {"census_zero_floor": CENSUS_ZERO_FLOOR},
              "2026-09-17 DEU fit-plot review (204 gemeindefreie Gebiete); people floor added 2026-09-29 after AUS/JPN listed 78k/203k census-0 units"),
        Check("A12", "A", "REPORT", "census", "Zero-mismatch roll-up by cause (a) ungated (b) unmodeled GBD location (c) smearing (d) phantom",
              {}, "2026-09-17 FIN/DEU fit-plot review; validate.metrics reads gbd_raked so unmodeled locations plot as zero"),
        # ---- Lane B: final-surface checks ----
        Check("B1", "B", "GATE", "time_point", "Pixel plausibility: non-finite/negative pixels GATE; people-per-pixel and people-per-m3 exceedances FLAG",
              {"ppp_flag": 500.0, "ppp_list": 1000.0, "ppm3": 0.2, "exceedance_share": 1e-4, "m3_per_person_floor": 5.0},
              "pixel-product lens; ITU density artifacts; bounds for 1600 m2 pixels, 0.2 person/m3 = 5 m3/person. FLAG (applied "
              "in the collator) when exceedance pixels are more than exceedance_share of a location-block's finite pixels or any "
              "pixel passes ppp_list: on the first full run one exceedance pixel anywhere flagged 62% of location-blocks while the "
              "99th-percentile share was 3e-4. ppp_list 5000 -> 1000 (rmbarber, 2026-09-30): the plausibility line; pixels "
              "above it are listed in B1_implausible with a mechanism label, and those with under m3_per_person_floor of "
              "residential volume per person are the physically impossible ones (first run: a 3,894-person Texas pixel with "
              "0.015 m3/person, census mass smeared onto a unit the model gave no stock)"),
        Check("B2", "B", "GATE", "time_point", "National/GBD reconciliation: surface sums vs GBD envelope per location",
              {"rf_tol": NO_CENSUS_RF_TOLERANCE}, "single-write design; raking_factors final guard"),
        Check("B3", "B", "FLAG", "release", "Splice-boundary smoothness: admin-total jumps at census-weight transitions",
              {"rel": 0.05, "abs": 100.0}, "weight-table bug class"),
        Check("B4", "B", "REPORT", "release", "Census population in unmodeled GBD locations wiped by the final factor",
              {}, "rmbarber's list item; roll-up of A12 class (b)"),
        Check("B5", "B", "FLAG", "time_point", "Block raster bounds ordering and grid alignment",
              {}, "admin_rasters gotchas"),
        Check("B6", "B", "FLAG", "block", "Pixel trajectory sample: per-pixel swing, blinks, one-quarter steps",
              {"swing": 2.0, "max_blinks": 0}, "pixel-product lens; A5 is unit-level"),
        Check("B7", "B", "FLAG", "release", "Splice/fringe accounting per GBD location",
              {"fringe_share": 0.05, "ln_factor_ratio": math.log(1.5), "min_coverage": 0.01},
              "2026-09-17 review; needs persisted rf component sums. min_coverage keeps sliver overlaps (a neighbour's census "
              "touching 0.01% of a location) from producing a meaningless implied factor; with it the first run flags Palestine, "
              "Gibraltar, Monaco, Vatican -- real cross-border census coverage"),
        Check("B8", "B", "FLAG", "release", "Census mass outside every GBD mask (coastal ring + whole territories)",
              {"wiped_share": 0.005}, "generalizes B4; needs persisted block census totals"),
        Check("B9", "B", "FLAG", "block", "Multi-census partial coverage: pixels valid in exactly one contributing census",
              {"people_share": 0.001}, "merge-sum never renormalizes weights"),
        Check("B10", "B", "REPORT", "release", "rf2 discontinuity across adjacent GBD locations",
              {"ratio": 1.25}, "value-based splice + per-location rf2"),
        # ---- Lane C: cross-release / external consistency ----
        Check("C1", "C", "REPORT", "release", "Cross-release diff of location and census-parent totals",
              {"top_n": 50}, "release face-validity practice"),
        Check("C2", "C", "REPORT", "census", "Model-at-anchor fidelity by admin level: nationally scaled base prediction vs census, from the tables",
              {}, "smearing investigation; R2 denominator lesson; raked(t_c) equals census by A1 so the model prediction is the informative side"),
        Check("C3", "C", "REPORT", "release", "Held-out multi-census scoring: flat vs v3 wMAPE near the other census dates",
              {"max_offset_quarters": 1}, "validation program; DESIGN section 5"),
    ]
}


# How to read each check: what it measures, what a flag means, what to do. Written for the
# person reading summary.md before a release; the numbers quoted are from the first full runs
# on 2026_09_06.007 and give a sense of scale, not thresholds.
GUIDANCE: dict[str, str] = {
    "A1": (
        "Measures whether, for every census task, the raked population at the census date sums to the census "
        "for units with any model stock. This is the one invariant the census rake promises: established stock "
        "is held flat at the census count. Read: abs_error should be zero to floating precision; any GATE row is a "
        "bookkeeping bug in the mechanism, never a data property. Act: stop and trace the task."
    ),
    "A2": (
        "Measures units credited with post-anchor construction whose implied density (people per populated pixel) "
        "exceeds their one-sided density bound. The cap is y <= max(C, bound x n_on), so a credited exceedance is "
        "impossible unless the cap failed: GATE. REPORT rows are established units already denser than their family "
        "bound at the anchor, an accepted class (about 0.7% of units) that A3 examines. Read: the GATE count must be "
        "zero; the REPORT count is context. Act: GATE means a cap regression."
    ),
    "A3": (
        "Measures census people per populated pixel at the anchor, which is the density the rake imposes on the "
        "surface. Flags above an absolute line or ten times the pooling parent's median. Extremes are one of two "
        "things: a unit whose people land on very little detected stock (institutions, missing buildings), or true "
        "high-rise (Hong Kong's median unit is 113 per pixel). Read: absolute flags cluster in high-rise countries; "
        "relative flags are within-family outliers worth a building-data look. The m3-per-person column in "
        "B1_implausible separates the two. Act: handoff list for building QC; the lines are calibration knobs."
    ),
    "A4": (
        "Measures each unit's maximum density over the window divided by its anchor density, i.e. how much "
        "post-anchor credit concentrates onto a unit's pixels. A ratio of 2-3 is ordinary for small units gaining "
        "construction; a long tail (ratios above 50) means credit is landing on near-empty footprints. Read: the "
        "flagged list ranked by conc. Act: inspect the top units; this is the case for an allocation fallback."
    ),
    "A5": (
        "Measures per-unit trajectory shape: swing (max over min raked population across the window), blinks "
        "(populated, then zero, then populated) and one-quarter steps. Unit totals should be smooth: flat before "
        "the anchor, monotone credit after it. Read: most flags are swing above 2x in small units with construction "
        "credit; the blink count is the cleaner pathology signal because it can only come from the building "
        "detector. Act: compare with B6, which measures the same thing per pixel; both are building-data metrics."
    ),
    "A6": (
        "Measures census people in units that never have a populated pixel (and separately, none at the anchor). "
        "These people cannot be placed within their unit and are left to the GBD rake, which spreads them across the "
        "GBD location. Read: people_never_gated is the headline (574k worldwide on the first run); it concentrates "
        "where the building data misses hamlets. Act: accept, or improve building coverage there."
    ),
    "A7": (
        "Measures, per quarter, the construction credit the density cap removed against the credit applied. "
        "share_removed is how much of the model's construction signal the cap distrusts. Read: a rising removed share "
        "through the window points at detector growth the cap is holding back; per-census rows say where. Act: this is "
        "the baseline to diff after a building-data update."
    ),
    "A8": (
        "Measures each census's national ratio of census count to model prediction (rf_country) against the median "
        "across all censuses. With a rate-unit model the absolute level means nothing, so the band is relative: a "
        "census five times above or below the median has an occupancy scale the model does not share. Read: the first "
        "run flagged SJM, PNG, BDI, DJI and HKG. Act: check the inputs for the flagged censuses."
    ),
    "A9": (
        "Measures enumerated census units that appear in no table because no pixel of the modeling frame falls "
        "inside them. Most are zero-population water or sliver units and are REPORT rows; FLAG rows carry people "
        "(8,992 on the first run, nearly all Mexico). Act: check the modeling frame and rasterization for flagged units."
    ),
    "A10": (
        "Measures pooling parents whose pooled construction rate rests on the prior (predicted mass below "
        "lambda over rf_country) and which have no density bound. Both backstops are off at once, so credit there is "
        "bounded only by the gate and the detected footprint. Read: zero rows is the expectation. Act: inspect any "
        "parent listed before release."
    ),
    "A11": (
        "Measures census-zero units in which the model places residential stock. FLAG when that phantom stock, "
        "scaled to people, reaches the floor. The anchor zeroes these units, so the product is right at the census "
        "date, but construction credit can still flow into them afterwards (phantom growth: 6.3M people by 2026q1 "
        "on the first run, USA 2.9M and MEX 2.7M). Read: phantom_people as a share of the national prediction says "
        "how much of the model's stock is not residential (USA 9%, MEX 30%); post_anchor_credit is the product "
        "effect. Act: residential-mask QC, and a policy decision on credit into census-zero units."
    ),
    "A12": (
        "Rolls up every off-axis point of the census-versus-prediction fit plots by cause: (a) census people with "
        "no populated pixel (gate loss or vanished), (b) census people in a GBD location the pipeline does not model, "
        "wiped by the final factor, (c) census people smeared onto little stock (A3's absolute class), (d) model "
        "stock where the census says nobody lives (A11). Read: each class has a different owner: modeling frame, "
        "GBD hierarchy policy, building data, residential mask. Act: route by class."
    ),
    "B1": (
        "Scans every pixel of the final surface. GATE on any non-finite or negative value. FLAG when exceedance "
        "pixels (above the people-per-pixel line, above the people-per-m3 line, or populated with no building volume) "
        "are more than a small share of a location-block's pixels, or any pixel passes the plausibility line (1,000). "
        "Pixels are the product, and the maximum value is the first thing a user sees. Read: use the companion "
        "artifacts: B1_locations ranks GBD locations, B1_implausible lists every pixel above the line with lon/lat, "
        "m3 of residential volume per person and the mechanism that put it there. Under 5 m3 per person is "
        "physically impossible and means census people landed on stock the residential mask removed (jails, "
        "dormitories); 5-20 m3 per person with a census that agrees with the model is dense but real. Act: covariates "
        "for the impossible class; heights for towers under-volumed; nothing for the real ones."
    ),
    "B2": (
        "Measures, per GBD location and quarter, the surface sum against the GBD envelope the final rake targets. "
        "Deviations of 1e-5 are float noise; above 1e-3 with a complete scan is a pipeline bug. Partial scans report "
        "only. Act: trace the raking factor for the location."
    ),
    "B3": (
        "Measures jumps in a location's quarterly total where the set of contributing censuses changes (multi-census "
        "countries blend anchors by inverse distance). The blend should be continuous. Read: FLAG rows name the "
        "location and the quarter pair. Act: inspect the census weights table for that country."
    ),
    "B4": (
        "Lists census people who live in GBD locations the pipeline does not model (Aland, the Australian external "
        "territories, Svalbard): the census rake places them, the final GBD factor is undefined there and wipes them. "
        "This is a reporting-coverage policy, not a bug. Read: by census. Act: decide whether to merge such "
        "territories into a modeled parent, as was done for Northern Cyprus."
    ),
    "B5": (
        "Checks each block raster's bounds ordering, position inside its block and alignment to the pixel grid. "
        "Read: all PASS is the only acceptable state; a FLAG means an export or projection defect."
    ),
    "B6": (
        "Measures per-pixel trajectories on a stratified sample of blocks: pixels that swing more than 2x, pixels "
        "that blink (populated, zero, populated) and one-quarter steps. This is flicker the unit-level checks cannot "
        "see because units average it away. Read: the first run found a median 35% of ever-populated pixels blinking "
        "and 20% swinging; this is the headline pixel-quality metric for the building-data series. Act: track across "
        "releases; the building-data update is the lever."
    ),
    "B7": (
        "Measures, per GBD location, how the two-rake flow combines census and model: the share of model mass "
        "covered by a census layer, the people filled in from the GBD-raked model surface where no census value "
        "exists (fringe), and the factor the census implies for its area against the national factor the fringe gets. "
        "FLAG when coverage is at least 1%, the fringe is a material share, and the two factors disagree by more than "
        "1.5x. Read: an implied factor near zero with high coverage means a zero-population census unit covers a real "
        "place (the Falklands under Argentina's census); moderate disagreements are neighbouring censuses covering "
        "small locations (Gibraltar, Monaco). Act: census exclusions for the first kind; note the second."
    ),
    "B8": (
        "Measures census people whose pixels fall outside every GBD polygon and are therefore wiped by an undefined "
        "final factor: the coastal ring where census and GBD boundaries disagree, plus whole unmodeled territories. "
        "Read: about 0.05% of census people worldwide on the first run; FLAG if any census loses more than 0.5%. "
        "Act: boundary review where flagged."
    ),
    "B9": (
        "Measures, where two censuses of one country contribute to a quarter, pixels valid in only one of them. "
        "The census layer merge never renormalizes weights, so such a pixel carries half its value at a 50/50 blend. "
        "Read: Italy's 2022 and 2023 censuses cover slightly different pixels. Act: harmonize census footprints or "
        "renormalize the merge."
    ),
    "B10": (
        "Measures the ratio of final raking factors between adjacent GBD locations inside census countries. Different "
        "factors across an internal border show as a seam in the surface. Read: ratios above 1.25 are listed with "
        "names. Act: context for the C1 review; no release action."
    ),
    "C1": (
        "Diffs location totals and census-parent totals against a previous version (requires --compare-version). "
        "Read: the top movers table is the human face-validity review. Act: explain every large mover before release."
    ),
    "C2": (
        "Measures the model's anchor prediction, nationally scaled to the census total, against the census at every "
        "admin level: rmse and correlation on log10 counts plus counts of zero mismatches. This is the fit plot "
        "computed from tables; raked totals equal census by A1, so the prediction is the informative side. Read: "
        "correlation near 0.95 at admin 1 falling toward finer levels is typical; level-3 census-zero/prediction-"
        "positive counts are A11's class. Act: compare across model versions."
    ),
    "C3": (
        "Scores, for multi-census countries, the flat census carried forward against the v3 mechanism's credited "
        "population at the quarter nearest the other census date (the offset is recorded; tables never hold the "
        "other date itself). A regression tripwire on the claim that construction credit helps. Read: near-ties at a "
        "one-quarter offset are expected because credit needs several quarters to accrue; v3 clearly worse than flat "
        "would be a regression. Act: investigate any census pair where v3 loses by a margin."
    ),
}
CHECKS = {check_id: replace(check, guidance=GUIDANCE[check_id]) for check_id, check in CHECKS.items()}


def checks_in_lane(lane: Lane) -> list[Check]:
    return [c for c in CHECKS.values() if c.lane == lane and c.enabled]
