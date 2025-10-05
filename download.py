import os
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import time


CSV_FILE = "SB_publication_PMC.csv"
OUTPUT_DIR = "data/full_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = "data/pmc_sections.csv"


df = pd.read_csv(CSV_FILE)
print(f"Total publications in CSV: {len(df)}")


def get_pmc_id(url):
    # URL format: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4136787/
    parsed = urlparse(url)
    if "ncbi.nlm.nih.gov" in parsed.netloc and "/PMC" in parsed.path:
        return parsed.path.split("/PMC")[-1].strip("/")
    return None

df['PMC_ID'] = df['Link'].apply(get_pmc_id)
df = df.dropna(subset=['PMC_ID'])
df = df.drop_duplicates(subset=['PMC_ID'])
print(f"Unique PMC articles to download: {len(df)}")


def fetch_pmc_xml(pmc_id):
    out_file = os.path.join(OUTPUT_DIR, f"PMC{pmc_id}.xml")
    if os.path.exists(out_file):
        print(f"[SKIP] Already downloaded: PMC{pmc_id}")
        return out_file

    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}&rettype=fulltext&retmode=xml"
    try:
        r = requests.get(fetch_url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"[OK] Downloaded PMC{pmc_id}")
            time.sleep(0.3)
            return out_file
        else:
            print(f"[ERROR] {r.status_code} for PMC{pmc_id}")
    except Exception as e:
        print(f"[ERROR] PMC{pmc_id}: {e}")
    return None

xml_files = []
for pmc_id in df['PMC_ID']:
    xml_file = fetch_pmc_xml(pmc_id)
    if xml_file:
        xml_files.append(xml_file)

print(f"Total XMLs downloaded: {len(xml_files)}")

output = []

for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError:
        print(f"[SKIP] Malformed XML: {xml_file}")
        continue

    pmc_id = os.path.basename(xml_file).replace(".xml", "")
    abstract = ""
    introduction = ""
    results = ""
    conclusion = ""

    # Extract abstract
    for ab in root.findall(".//abstract"):
        abstract += " ".join([t.text for t in ab.findall(".//p") if t.text])

    # Extract sections by title
    for sec in root.findall(".//sec"):
        title = sec.find("title")
        text = " ".join([p.text for p in sec.findall(".//p") if p.text])
        if not text:
            continue
        if title is not None and title.text:
            t = title.text.lower()
            if "introduction" in t:
                introduction += text
            elif "result" in t:
                results += text
            elif "conclusion" in t or "discussion" in t:
                conclusion += text

    output.append({
        "PMC_ID": pmc_id,
        "Abstract": abstract,
        "Introduction": introduction,
        "Results": results,
        "Conclusion": conclusion
    })


df_sections = pd.DataFrame(output)
df_sections.to_csv(OUTPUT_CSV, index=False)
print(f" Sections extracted and saved to {OUTPUT_CSV}")
