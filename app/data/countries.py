"""
Canonical country list for business-card country normalization.

╔══════════════════════════════════════════════════════════════════════════╗
║  ACTION REQUIRED — REPLACE THIS LIST WITH THE CLIENT'S EXACT 265 NAMES    ║
║                                                                          ║
║  The business-card reader fuzzy-matches the OCR'd country text against    ║
║  CANONICAL_COUNTRIES and returns the matched entry VERBATIM. For the      ║
║  client's records to line up, every string here must be byte-for-byte    ║
║  identical to the names in their database (spelling, casing, spacing,    ║
║  punctuation).                                                            ║
║                                                                          ║
║  The list below is the standard ISO 3166-1 English short-name set        ║
║  (~249 entries) provided as a working default. Swap it for the client's  ║
║  authoritative 265-row export, then update COUNTRY_ALIASES targets to    ║
║  match the new canonical strings.                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical names — one entry per country, exactly as stored downstream.
# ---------------------------------------------------------------------------
CANONICAL_COUNTRIES: list[str] = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola",
    "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados",
    "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei",
    "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile",
    "China", "Colombia", "Comoros", "Congo", "Costa Rica",
    "Croatia", "Cuba", "Cyprus", "Czechia", "Democratic Republic of the Congo",
    "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador",
    "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France",
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana",
    "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hungary", "Iceland",
    "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan",
    "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kosovo",
    "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives",
    "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia",
    "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
    "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway",
    "Oman", "Pakistan", "Palau", "Palestine", "Panama",
    "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
    "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal",
    "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia",
    "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea",
    "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname",
    "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan",
    "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu",
    "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela",
    "Vietnam", "Yemen", "Zambia", "Zimbabwe",
    # ISO-listed territories / dependencies commonly seen on business cards
    "Hong Kong", "Macau", "Puerto Rico", "Greenland", "Gibraltar",
    "Bermuda", "Cayman Islands", "British Virgin Islands", "Guam", "Aruba",
    "Curacao", "Faroe Islands", "French Polynesia", "New Caledonia", "Jersey",
    "Guernsey", "Isle of Man", "Monaco",
]

# ---------------------------------------------------------------------------
# Aliases — abbreviations / common variants the fuzzy matcher handles poorly.
# Checked BEFORE fuzzy matching. Every value MUST exist in CANONICAL_COUNTRIES.
# Keys are matched case-insensitively after whitespace/punctuation collapse.
# ---------------------------------------------------------------------------
COUNTRY_ALIASES: dict[str, str] = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.a": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s": "United States",
    "united states of america": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "u.k": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "u.a.e": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "ksa": "Saudi Arabia",
    "k.s.a.": "Saudi Arabia",
    "kingdom of saudi arabia": "Saudi Arabia",
    "saudi": "Saudi Arabia",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "korea, republic of": "South Korea",
    "korea (south)": "South Korea",
    "s. korea": "South Korea",
    "dominican rep": "Dominican Republic",
    "dominican rep.": "Dominican Republic",
    "dom rep": "Dominican Republic",
    "dprk": "North Korea",
    "korea, democratic people's republic of": "North Korea",
    "prc": "China",
    "people's republic of china": "China",
    "roc": "Taiwan",
    "republic of china": "Taiwan",
    "russian federation": "Russia",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "czech republic": "Czechia",
    "burma": "Myanmar",
    "cape verde": "Cabo Verde",
    "swaziland": "Eswatini",
    "macedonia": "North Macedonia",
    "vatican": "Vatican City",
    "holy see": "Vatican City",
    "east timor": "Timor-Leste",
    "hk": "Hong Kong",
    "uae.": "United Arab Emirates",
}
