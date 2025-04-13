"""GBD location ids"""

import pandas as pd

NO_ISO_CODE = "ZZZ"

# Region ids

EAST_ASIA = 5
SOUTHEAST_ASIA = 9
OCEANIA = 21
CENTRAL_ASIA = 32
CENTRAL_EUROPE = 42
EASTERN_EUROPE = 56
HIGH_INCOME_ASIA_PACIFIC = 65
AUSTRALASIA = 70
WESTERN_EUROPE = 73
SOUTHERN_LATIN_AMERICA = 96
HIGH_INCOME_NORTH_AMERICA = 100
CARIBBEAN = 104
ANDEAN_LATIN_AMERICA = 120
CENTRAL_LATIN_AMERICA = 124
TROPICAL_LATIN_AMERICA = 134
NORTH_AFRICA_AND_MIDDLE_EAST = 138
SOUTH_ASIA = 159
CENTRAL_SUB_SAHARAN_AFRICA = 167
EASTERN_SUB_SAHARAN_AFRICA = 174
SOUTHERN_SUB_SAHARAN_AFRICA = 192
WESTERN_SUB_SAHARAN_AFRICA = 199

NO_REGION_ID = -1

TO_DROP_PARENTS = [
    # Drop UK UTLAs from these regions
    4618,
    4919,
    4620,
    4621,
    4622,
    4623,
    4624,
    4625,
    4626,
    # Drop the India urban/rural splits from these states
    4841,
    4842,
    4843,
    4844,
    4846,
    4849,
    4850,
    4851,
    4852,
    4853,
    4854,
    4855,
    4856,
    4857,
    4859,
    4860,
    4861,
    4862,
    4863,
    4864,
    4865,
    4867,
    4868,
    4869,
    4870,
    4871,
    4872,
    4873,
    4874,
    4875,
    44538,
    # Drop the Maori/non-Maori split from New Zealand
    72,
]

TO_USE_LSAE_SHAPES = [
    92,  # Spain, removes Canary Islands
    71,  # Australia, removes Ashmore and Cartier Islands and Coral Sea Islands
    # Canada and Greenland intersect in the GBD hierarchy, so swap both.
    101,
    349,
]


class SUPPLEMENT:
    WPP = "wpp"
    UNMODELED = "unmodeled"
    ZERO_POPULATION = "zero_population"


def load_supplmental_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # This is a manual mapping of locations not present in the GBD hierarchy to the
            # GBD region in which they reside. We grab the additional iso3 and location name
            # metadata from the LSAE location hierarchy since they are not present in the GBD
            # hierarchy. We have to manually include the region_id because the LSAE hierarchy
            # does not consider regions in their hierarchy.
            #
            # CATEGORY ONE: `wpp`
            # These locations are present in the WPP data and not modeled by GBD, so WPP is used
            # to produce scalars of location_population / region_population, then applied to the
            # GBD regional data to produce location-level estimates.
            #
            # LOCATION GROUP ONE: Mapped
            # These locations are present in the IHME mapping file:
            # /home/j/DATA/IHME_COUNTRY_CODES/IHME_COUNTRY_CODES_Y2013M07D26.CSV
            # This file has unknown provenance and is obviously quite old, but it is what is
            # primarily usied in GBD scalar production, so we reproduce here. The relevant
            # columns from the mapping file are (location_id, location_name, gbd_region_id).
            # GBD scalar production uses the `iso_num` to map to WPP, but this is 1:1 with the
            # iso3 code, so we use the iso3 code to map to WPP.
            #
            # This location to region mapping has been manually checked for geographic
            # consistency, but needs a round of review with the GBD demographics team.
            #
            (299, "Anguilla", CARIBBEAN, "AIA", SUPPLEMENT.WPP),
            (300, "Aruba", CARIBBEAN, "ABW", SUPPLEMENT.WPP),
            (313, "Cayman Islands", CARIBBEAN, "CYM", SUPPLEMENT.WPP),
            (
                331,
                "Falkland Islands (Islas Malvinas)",
                SOUTHERN_LATIN_AMERICA,
                "FLK",
                SUPPLEMENT.WPP,
            ),
            (332, "Faroe Islands", WESTERN_EUROPE, "FRO", SUPPLEMENT.WPP),
            (338, "French Guiana", CARIBBEAN, "GUF", SUPPLEMENT.WPP),
            (339, "French Polynesia", OCEANIA, "PYF", SUPPLEMENT.WPP),
            (345, "Gibraltar", WESTERN_EUROPE, "GIB", SUPPLEMENT.WPP),
            (350, "Guadeloupe", CARIBBEAN, "GLP", SUPPLEMENT.WPP),
            (352, "Guernsey", WESTERN_EUROPE, "GGY", SUPPLEMENT.WPP),
            (353, "Holy See (Vatican City)", WESTERN_EUROPE, "VAT", SUPPLEMENT.WPP),
            (355, "Isle of Man", WESTERN_EUROPE, "IMN", SUPPLEMENT.WPP),
            (356, "Jersey", WESTERN_EUROPE, "JEY", SUPPLEMENT.WPP),
            (360, "Liechtenstein", WESTERN_EUROPE, "LIE", SUPPLEMENT.WPP),
            (363, "Martinique", CARIBBEAN, "MTQ", SUPPLEMENT.WPP),
            (368, "Montserrat", CARIBBEAN, "MSR", SUPPLEMENT.WPP),
            (372, "New Caledonia", OCEANIA, "NCL", SUPPLEMENT.WPP),
            (387, "Reunion", SOUTHEAST_ASIA, "REU", SUPPLEMENT.WPP),
            (391, "Saint Barthelemy", CARIBBEAN, "BLM", SUPPLEMENT.WPP),
            (394, "Saint Martin", CARIBBEAN, "MAF", SUPPLEMENT.WPP),
            (
                395,
                "Saint Pierre and Miquelon",
                HIGH_INCOME_NORTH_AMERICA,
                "SPM",
                SUPPLEMENT.WPP,
            ),
            (415, "Turks and Caicos Islands", CARIBBEAN, "TCA", SUPPLEMENT.WPP),
            (421, "Virgin Islands, British", CARIBBEAN, "VGB", SUPPLEMENT.WPP),
            (423, "Wallis and Futuna Islands", OCEANIA, "WLF", SUPPLEMENT.WPP),
            (
                424,
                "Western Sahara",
                NORTH_AFRICA_AND_MIDDLE_EAST,
                "ESH",
                SUPPLEMENT.WPP,
            ),
            #
            # LOCATION GROUP TWO: those not present in the IHME mapping file
            # These locations are present in WPP, but missing from the IHME mapping file.
            # In GBD scalar production, they are manually mapped to region ids.
            #
            (364, "Mayotte", EASTERN_SUB_SAHARAN_AFRICA, "MYT", SUPPLEMENT.WPP),
            # The ISO3 code is manually set here, as it's not present in LSAE
            (4641, "Curaçao", CARIBBEAN, "CUW", SUPPLEMENT.WPP),
            #
            # LOCATION GROUP THREE: Corrections
            # These locations are have some aspect of their metadata corrected from the GBD
            # scalar production process.
            #
            # GBD thinks the location id is 392, but LSAE says its 60927. Use LSAE.
            # ISO3 is manually set.
            (
                60927,
                "Saint Helena, Ascension, and Tristan da Cunha",
                WESTERN_SUB_SAHARAN_AFRICA,
                "SHN",
                SUPPLEMENT.WPP,
            ),
            # These are manually mapped in GBD scalar production, to EASTER_SUB_SAHARAN_AFRICA,
            # though the accomanying comment and a quick google maps confirms they are in the
            # Caribbean.
            # ISO3 is manually set.
            (4642, "Sint Maarten", CARIBBEAN, "SXM", SUPPLEMENT.WPP),
            # ISO3 is manually set.
            (
                60922,
                "Bonaire, Sint Eustatius and Saba",
                CARIBBEAN,
                "BES",
                SUPPLEMENT.WPP,
            ),
            #
            # CATEGORY TWO: `unmodeled`
            # These locations are not present in either WPP or GBD, but I have been able to
            # confirm that people live there by checking out wikipedia. These will end up
            # with a `nan` scalar/population in our raking dataset as we cannot use WPP to
            # produce scalars for them.
            #
            # LOCATION GROUP ONE: Mapped
            # These locations are present in the IHME mapping file:
            # /home/j/DATA/IHME_COUNTRY_CODES/IHME_COUNTRY_CODES_Y2013M07D26.CSV
            # despite the fact that they do not have wpp estimates.
            (
                296,
                "Akrotiri and Dhekelia",
                WESTERN_EUROPE,
                NO_ISO_CODE,
                SUPPLEMENT.UNMODELED,
            ),
            (297, "Aland Islands", WESTERN_EUROPE, "ALA", SUPPLEMENT.UNMODELED),
            (318, "Christmas Island", SOUTHEAST_ASIA, "CXR", SUPPLEMENT.UNMODELED),
            (
                319,
                "Cocos (Keeling) Islands",
                SOUTHEAST_ASIA,
                "CCK",
                SUPPLEMENT.UNMODELED,
            ),
            (375, "Norfolk Island", OCEANIA, "NFK", SUPPLEMENT.UNMODELED),
            (382, "Pitcairn Islands", OCEANIA, "PCN", SUPPLEMENT.UNMODELED),
            (
                411,
                "Svalbard and Jan Mayen Islands",
                WESTERN_EUROPE,
                "SJM",
                SUPPLEMENT.UNMODELED,
            ),
            #
            # LOCATION GROUP TWO: Unmapped
            # These locations are missing from the IHME mapping file
            (
                53483,
                "Turkish Republic of Northern Cyprus",
                WESTERN_EUROPE,
                NO_ISO_CODE,
                SUPPLEMENT.UNMODELED,
            ),
            (311, "Canary Islands", WESTERN_EUROPE, "XCA", SUPPLEMENT.UNMODELED),
            #
            # CATEGORY THREE: `zero_population`
            # The locations are not present in either WPP or GBD, but I have been able to
            # confirm that they have no population.  These will end up with a `0`
            # scalar/population
            #
            # ISO3 manually set
            (
                60348,
                "Terres australes et antarctiques françaises",
                EASTERN_SUB_SAHARAN_AFRICA,
                "ATF",
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                60921,
                "Antarctica",
                SOUTHERN_LATIN_AMERICA,
                "ATA",
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                60923,
                "Bouvet Island",
                SOUTHERN_LATIN_AMERICA,
                "BVT",
                SUPPLEMENT.ZERO_POPULATION,
            ),
            # ISO3 manually set
            (
                60924,
                "Heard Island and McDonald Islands",
                AUSTRALASIA,
                "HMD",
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                60925,
                "British Indian Ocean Territory",
                SOUTH_ASIA,
                "IOT",
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                60926,
                "South Georgia and the South Sandwich Islands",
                SOUTHERN_LATIN_AMERICA,
                "SGS",
                SUPPLEMENT.ZERO_POPULATION,
            ),
            # ISO3 manually set
            (
                60928,
                "United States Minor Outlying Islands",
                HIGH_INCOME_NORTH_AMERICA,
                "UMI",
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                60930,
                "Clipperton Island",
                CENTRAL_LATIN_AMERICA,
                NO_ISO_CODE,
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                60931,
                "Paracel Islands",
                EAST_ASIA,
                NO_ISO_CODE,
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                93924,
                "Spratly Islands",
                SOUTHEAST_ASIA,
                NO_ISO_CODE,
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                94026,
                "Ashmore and Cartier Islands",
                AUSTRALASIA,
                NO_ISO_CODE,
                SUPPLEMENT.ZERO_POPULATION,
            ),
            (
                94027,
                "Coral Sea Islands Territory",
                AUSTRALASIA,
                NO_ISO_CODE,
                SUPPLEMENT.ZERO_POPULATION,
            ),
        ],
        columns=[
            "location_id",
            "location_name",
            "region_id",
            "ihme_loc_id",
            "category",
        ],
    )
