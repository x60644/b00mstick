"""
b00mstick — coach_photos.py (local scraper, run on demand)
==========================================================
Builds data/coach_photos.json: {team: {coach_name: {photo, title}}} by
reading each club's /team/coaches-roster/ page. Headshots are the
standardized static.clubs.nfl.com uploads and only appear in the RAW
page HTML, so this parses HTML directly (stdlib only, no bs4).

Scrapes EVERY coach on the page, not just HC/OC — preseason playcallers
are often position coaches (e.g. QB coaches), and keeping the full staff
makes the name lookup in score_week/score_preseason robust.

Failures (bot protection etc.) are reported at the end for manual URL
entry; the app's gold-initials fallback covers any gaps.

Run:  python coach_photos.py            # all 32 teams
      python coach_photos.py KC PHI     # just these teams
Doubles as a mid-season self-heal after coaching changes — rerun and
recommit the JSON.
"""

import json
import re
import sys
import urllib.request
from html.parser import HTMLParser

DOMAINS = {
    "ARI": "azcardinals.com", "ATL": "atlantafalcons.com", "BAL": "baltimoreravens.com",
    "BUF": "buffalobills.com", "CAR": "panthers.com", "CHI": "chicagobears.com",
    "CIN": "bengals.com", "CLE": "clevelandbrowns.com", "DAL": "dallascowboys.com",
    "DEN": "denverbroncos.com", "DET": "detroitlions.com", "GB": "packers.com",
    "HOU": "houstontexans.com", "IND": "colts.com", "JAX": "jaguars.com",
    "KC": "chiefs.com", "LA": "therams.com", "LAC": "chargers.com",
    "LV": "raiders.com", "MIA": "miamidolphins.com", "MIN": "vikings.com",
    "NE": "patriots.com", "NO": "neworleanssaints.com", "NYG": "giants.com",
    "NYJ": "newyorkjets.com", "PHI": "philadelphiaeagles.com", "PIT": "steelers.com",
    "SEA": "seahawks.com", "SF": "49ers.com", "TB": "buccaneers.com",
    "TEN": "tennesseetitans.com", "WAS": "commanders.com",
}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")
TITLE_RE = re.compile(r"coach|coordinator|assistant|specialist", re.I)

class RosterParser(HTMLParser):
    """Flattens the page into an ordered stream of headshot imgs and text."""
    def __init__(self):
        super().__init__()
        self.tokens = []
    def handle_starttag(self, tag, attrs):
        if tag != "img":
            return
        a = dict(attrs)
        # lazy-loaded pages put a base64 placeholder in src and the real
        # URL in data-src / data-srcset — take whichever holds the CDN url
        src = ""
        for key in ("src", "data-src", "data-srcset", "srcset"):
            v = a.get(key) or ""
            if "static.clubs.nfl.com" in v:
                src = v.split()[0].rstrip(",")   # srcset: first candidate
                break
        if src:
            if src.startswith("//"):
                src = "https:" + src
            self.tokens.append(("img", src, (a.get("alt") or "").strip()))
    def handle_data(self, data):
        s = " ".join(data.split())
        if s:
            self.tokens.append(("txt", s, ""))

def scrape_team(team, domain):
    html, last = "", None
    for path in ("/team/coaches-roster/", "/team/coaches/"):
        try:
            req = urllib.request.Request(f"https://www.{domain}{path}",
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                html = r.read().decode("utf-8", errors="replace")
            break
        except Exception as e:
            last = e
    if not html:
        raise last
    p = RosterParser()
    p.feed(html)
    coaches = {}
    for i, (kind, src, alt) in enumerate(p.tokens):
        if kind != "img":
            continue
        # name = alt text when present; title = nearest following title-looking text
        name, title = alt, ""
        window = [t for t in p.tokens[i + 1:i + 8] if t[0] == "txt"]
        for _, s, _ in window:
            if TITLE_RE.search(s) and len(s) < 60:
                title = s
                break
        if not name:
            for _, s, _ in window:
                if not TITLE_RE.search(s) and 4 < len(s) < 40:
                    name = s
                    break
        if name and name not in coaches:
            coaches[name] = {"photo": src, "title": title}
    return coaches

def main():
    only = [t.upper() for t in sys.argv[1:]]
    teams = {t: d for t, d in DOMAINS.items() if not only or t in only}
    try:
        with open("data/coach_photos.json") as f:
            out = json.load(f)          # keep prior results / manual entries
    except FileNotFoundError:
        out = {}
    failed = []
    for team, domain in sorted(teams.items()):
        try:
            coaches = scrape_team(team, domain)
            if not coaches:
                raise ValueError("0 coaches parsed")
            out[team] = coaches
            print(f"  {team}: {len(coaches)} coaches")
        except Exception as e:
            failed.append(team)
            print(f"  {team}: FAILED ({type(e).__name__}: {e})")
    with open("data/coach_photos.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote data/coach_photos.json ({len(out)} teams)")
    if failed:
        print(f"NEEDS MANUAL ENTRY (bot protection?): {', '.join(failed)}\n"
              "  -> add entries by hand to data/coach_photos.json; the app's\n"
              "     initials fallback covers them until then.")

if __name__ == "__main__":
    main()
