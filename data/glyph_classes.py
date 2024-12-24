# GLYPH_CLASSES = ["fonts", "lamed", "yud", "nikud", "numbers", "noise", "points"] # 00_font_classifier.pth

GLYPH_CLASSES = [
    "00_alef",
    "01_bet",
    "03_gimal",
    "04_dalet",
    "05_he",
    "06_vav",
    "07_zayin",
    "08_chet",
    "09_tet",
    "10_yud",
    "11_caf",
    "12_lamed",
    "13_mem",
    "14_nun",
    "15_samech",
    "16_ayin",
    "17_pe",
    "18_tzadi",
    "19_kuf",
    "20_resh",
    "21_shin",
    "22_tav",
    "23_caf_sofit",
    "24_mem_sofit",
    "25_nun_sofit",
    "26_pe_sofit",
    "27_tzadi_sofit",
    "30_broken",
    "31_fused",
    "40_comma",
    "50_numbers",
    "60_english",
    "70_noise",
    "71_points",
    "80_nikud",
    "81_teamim",
]


def categorize_glyph_class(glyph_class):
    # Define the font categories
    regular_height_glyphs = [
        "00_alef",
        "01_bet",
        "03_gimal",
        "04_dalet",
        "05_he",
        "06_vav",
        "07_zayin",
        "08_chet",
        "09_tet",
        "11_caf",
        "12_lamed",
        "13_mem",
        "14_nun",
        "15_samech",
        "16_ayin",
        "17_pe",
        "18_tzadi",
        "19_kuf",
        "20_resh",
        "21_shin",
        "22_tav",
        "24_mem_sofit",
    ]
    under_the_line_glyphs = [
        "23_caf_sofit",
        "25_nun_sofit",
        "26_pe_sofit",
        "27_tzadi_sofit",
    ]
    irregular_size_glyphs = ["30_broken", "31_fused"]
    nikud_glyphs = ["80_nikud", "81_teamim"]

    # Check which list the font belongs to and return the corresponding value
    if glyph_class in regular_height_glyphs:
        return 0
    elif glyph_class == "12_lamed":
        return 1
    elif glyph_class in under_the_line_glyphs:
        return 2
    elif glyph_class in irregular_size_glyphs:
        return 3
    elif glyph_class == "10_yud":
        return 4
    elif glyph_class in nikud_glyphs:
        return 4
    elif glyph_class == "71_points":
        return 6
    else:
        return -1
